from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, fields
from typing import Dict, Tuple

import torch

from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)

# Process-wide pool keyed by (namespace, name, numel, dtype, device); see
# share_input_buffer.
_PoolKey = Tuple[str, str, int, torch.dtype, torch.device]
_forward_input_buffer_pool: Dict[_PoolKey, torch.Tensor] = {}


def share_input_buffer(
    name: str, new_buffer: torch.Tensor, namespace: str = ""
) -> torch.Tensor:
    """Coalesce a buffer by ``(namespace, name, size, dtype, device)`` into
    the process-wide input-buffer pool.

    Distinct callers that request the same field ``name`` with the same
    size/dtype/device share one physical allocation (and therefore one
    ``data_ptr``): the first registrant's buffer becomes canonical and every
    later identical request is returned as a view aliased onto it. Requests
    that differ in size get their own allocation — they never reuse or displace
    an existing entry — so the sharing *structure* is independent of
    registration order and no already-captured buffer is ever repointed.

    This pool is process-wide and governs *every* ``share_buffers()`` caller —
    including graph runners not yet on the registry (the speculative draft /
    draft-extend / frozen-kv-mtp / multi-layer-eagle runners), which register
    identically-named ``input_ids`` / ``positions`` / ``out_cache_loc`` /
    ``mrope_positions``. Cross-runner sharing is safe ONLY while the forwards
    that use the buffers are sequential / mutually exclusive (they are filled
    immediately before each replay).

    ``namespace`` partitions the pool for callers that BREAK that premise:
    under --enable-spec-pdmux (M2.2+) the draft-side graphs replay on the
    SMALL green-ctx stream concurrently with the target's verify graph on the
    large stream, so the draft/draft-extend runners register their statics
    under a dedicated namespace instead of aliasing the target runner's
    (measured hazard, M2.5: the target-verify and draft-extend runners have
    identical keys for input_ids / positions / out_cache_loc / seq_lens /
    req_pool_indices / next_token_logits_buffer — both use num_tokens_per_bs
    = num_draft_tokens and the same max_bs — and verify(X)'s fills overwrote
    the concurrently-replaying extend graph's baked inputs and logits output;
    tau 3.12 -> ~2 at c=2). Draft-side runners still share buffers among
    THEMSELVES: all draft-phase work is serialized on the one small stream.
    """
    key: _PoolKey = (
        namespace,
        name,
        new_buffer.numel(),
        new_buffer.dtype,
        new_buffer.device,
    )
    canonical = _forward_input_buffer_pool.get(key, None)
    if canonical is None:
        _forward_input_buffer_pool[key] = new_buffer
        canonical = new_buffer
    else:
        logger.debug(
            "[input-buffer-pool] alias: ns=%r '%s' numel=%d dtype=%s "
            "device=%s data_ptr=0x%x",
            namespace,
            name,
            new_buffer.numel(),
            new_buffer.dtype,
            new_buffer.device,
            canonical.data_ptr(),
        )
    return canonical.as_strided(new_buffer.size(), new_buffer.stride())


def share_input_buffers_in(obj, namespace: str = "") -> None:
    """Pool every tensor buffer on ``obj`` (dataclass / ``SimpleNamespace``)
    through the process-wide pool, in place. No-op on NPU; recurses into dict /
    dataclass buffer fields (``pp_proxy_tensors`` / ``ngram_embedding_info``)."""
    if is_npu():
        return

    for name, buffer in list(vars(obj).items()):
        if buffer is None:
            continue
        if dataclasses.is_dataclass(buffer):
            buffer = vars(buffer)
        if isinstance(buffer, dict):
            for sub_name, sub_buffer in buffer.items():
                assert isinstance(
                    sub_buffer, torch.Tensor
                ), f"Field {name}.{sub_name} is expected to be a torch.Tensor, but got {type(sub_buffer)}."
                buffer[sub_name] = share_input_buffer(
                    f"{name}.{sub_name}", sub_buffer, namespace
                )
        else:
            assert isinstance(
                buffer, torch.Tensor
            ), f"Field {name} is expected to be a torch.Tensor, a dict of torch.Tensor, or a dataclass of torch.Tensor, but got {type(buffer)}."
            setattr(obj, name, share_input_buffer(name, buffer, namespace))


@dataclass
class ForwardInputBuffers:

    def _share_one_buffer(
        self, name: str, new_buffer: torch.Tensor, namespace: str = ""
    ) -> torch.Tensor:
        return share_input_buffer(name, new_buffer, namespace)

    def share_buffers(self, namespace: str = ""):
        # disable share input buffer on npu due to accuracy issue
        if is_npu():
            return

        for f in fields(self):
            name = f.name
            buffer = getattr(self, name)

            if buffer is None:
                continue

            if dataclasses.is_dataclass(buffer):
                buffer = vars(buffer)

            if isinstance(buffer, dict):
                for sub_name, sub_buffer in buffer.items():
                    assert isinstance(
                        sub_buffer, torch.Tensor
                    ), f"Field {name}.{sub_name} is expected to be a torch.Tensor, but got {type(sub_buffer)}."
                    new_buffer = self._share_one_buffer(
                        f"{name}.{sub_name}", sub_buffer, namespace
                    )
                    buffer[sub_name] = new_buffer
            else:
                assert isinstance(
                    buffer, torch.Tensor
                ), f"Field {name} is expected to be a torch.Tensor, a dict of torch.Tensor, or a dataclass of torch.Tensor, but got {type(buffer)}."
                new_buffer = self._share_one_buffer(name, buffer, namespace)
                setattr(self, name, new_buffer)
