# spec-pdmux verify-tax optimization prototype (zero-overhead memo item 5):
# mark the TARGET model's linear-weight reads L2 EVICT-FIRST
# (cudaAccessPropertyStreaming) so the verify weight stream — which cannot hit
# in L2 anyway (16 GB/rank per pass at 32B-TP4 vs 40 MB L2) — stops evicting
# the co-located drafter's resident lines on the shared L2.
#
# Mechanism: an env-gated forward_pre_hook on every LinearBase module of the
# TARGET model sets a cudaStreamAttributeAccessPolicyWindow over that layer's
# weight (hitProp=missProp=Streaming, hitRatio=1.0, clamped to the device's
# maxAccessPolicyWindowSize) on the CURRENT stream before the matmul launches.
# During verify-graph capture on the LARGE green-ctx stream the policy is
# baked into the captured kernel nodes (probe P1, l2_evict_probe.py: setting
# the attribute during capture is legal and the graph replays correctly), so
# steady-state replay pays ZERO CPU cost. Decode-shaped inputs only: prefill
# forwards (rows > SGLANG_SPEC_PDMUX_L2_ROWS_MAX) reset the window to Normal —
# prefill GEMMs reuse weight tiles from L2 across block rows, streaming them
# would regress prefill.
#
# Gated by SGLANG_SPEC_PDMUX_L2_EVICT_FIRST=1 (default off; stock byte-safe).
import ctypes
import logging

import torch

logger = logging.getLogger(__name__)

_ATTR_ACCESS_POLICY_WINDOW = 1  # cudaStreamAttributeAccessPolicyWindow
_ACCESS_NORMAL, _ACCESS_STREAMING = 0, 1


class _AccessPolicyWindow(ctypes.Structure):
    _fields_ = [
        ("base_ptr", ctypes.c_void_p),
        ("num_bytes", ctypes.c_size_t),
        ("hitRatio", ctypes.c_float),
        ("hitProp", ctypes.c_int),
        ("missProp", ctypes.c_int),
    ]


class _StreamAttrValue(ctypes.Union):
    _fields_ = [("accessPolicyWindow", _AccessPolicyWindow),
                ("pad", ctypes.c_byte * 64)]


_rt = None
_max_window = 0


def _init_rt():
    global _rt, _max_window
    if _rt is None:
        _rt = ctypes.CDLL("libcudart.so")
        v = ctypes.c_int(0)
        # cudaDevAttrMaxAccessPolicyWindowSize = 109
        _rt.cudaDeviceGetAttribute(ctypes.byref(v), 109, torch.cuda.current_device())
        _max_window = v.value
    return _rt


def _set_window(base_ptr, num_bytes, prop):
    v = _StreamAttrValue()
    v.accessPolicyWindow = _AccessPolicyWindow(
        ctypes.c_void_p(base_ptr) if base_ptr else None,
        min(num_bytes, _max_window), 1.0, prop, prop)
    _rt.cudaStreamSetAttribute(
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        _ATTR_ACCESS_POLICY_WINDOW, ctypes.byref(v))


def _pre_hook(module, args, rows_max):
    # Prefill is TorchDynamo-compiled in SGLang v2; tracing a ctypes Structure
    # ctor blows up (and a graph break per linear would wreck prefill anyway).
    # Skipping under trace is exactly right: the window is only ever wanted on
    # the DECODE path, whose graphs are captured EAGERLY -> the policy still
    # bakes into the verify graph's kernel nodes. Compiled prefill keeps L2
    # Normal, which is the behaviour the rows_max reset branch existed for.
    if torch.compiler.is_compiling():
        return
    x = args[0] if args else None
    if x is not None and x.dim() >= 2 and x.shape[0] > rows_max:
        _set_window(None, 0, _ACCESS_NORMAL)  # prefill-shaped: keep L2 normal
        return
    w = getattr(module, "weight", None)
    if w is not None and w.is_cuda:
        _set_window(w.data_ptr(), w.numel() * w.element_size(), _ACCESS_STREAMING)


def install_l2_evict_first_hooks(model, rows_max=512):
    """Register the evict-first pre-hook on every LinearBase of `model`.
    Returns the hook count. Caller gates on env + target-worker."""
    from sglang.srt.layers.linear import LinearBase

    _init_rt()
    n = 0
    for m in model.modules():
        if isinstance(m, LinearBase):
            m.register_forward_pre_hook(
                lambda mod, a, _rm=rows_max: _pre_hook(mod, a, _rm))
            n += 1
    logger.info(
        "[spec-pdmux] L2 evict-first weight-stream hooks on %d linear "
        "modules (window clamp %d MiB, prefill rows_max %d)",
        n, _max_window >> 20, rows_max)
    return n
