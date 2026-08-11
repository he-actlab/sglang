# Adapted from qwen2.py
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
)
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.qwen2 import Qwen2MLP, Qwen2Model
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, get_bool_env_var, is_cuda, is_hip, is_npu
from sglang.srt.utils.draft_extend_surface_probe import (
    draft_extend_projection_scope,
)

Qwen3Config = None

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

_has_fused_qk_norm_mrope = False
if _use_aiter:
    try:
        from aiter import fused_qk_norm_mrope_3d_cache_pts_quant_shuffle

        _has_fused_qk_norm_mrope = True
        logger.info("aiter fused_qk_norm_mrope_3d kernel available")
    except ImportError:
        pass

if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope

    from sglang.srt.hardware_backend.npu.cmo import get_cmo_stream, wait_cmo_stream


class _Qwen3DrafterTmaDispatch:
    """Per-draft-model adapter for the exact fixed-width TMA projections."""

    def __init__(self, device_index: int) -> None:
        from sglang.jit_kernel.cutedsl_drafter_tma_gemm import (
            can_run_drafter_tma_model_projection,
            drafter_tma_model_projection,
            precompile_drafter_tma_model_projections,
        )

        precompile_drafter_tma_model_projections(device_index)
        self._can_run = can_run_drafter_tma_model_projection
        self._run = drafter_tma_model_projection

    @staticmethod
    def supports_linear(linear: nn.Module) -> bool:
        weight = getattr(linear, "weight", None)
        return bool(
            getattr(linear, "tp_size", None) == 1
            and isinstance(
                getattr(linear, "quant_method", None), UnquantizedLinearMethod
            )
            and getattr(linear, "bias", None) is None
            and not getattr(linear, "gather_output", False)
            and getattr(linear, "input_is_parallel", True)
            and not getattr(linear, "use_dp_attention_reduce", False)
            and isinstance(weight, torch.Tensor)
            and weight.is_cuda
            and weight.dtype == torch.bfloat16
            and weight.ndim == 2
            and weight.is_contiguous()
            and weight.data_ptr() % 16 == 0
        )

    def __call__(
        self, linear: nn.Module, activation: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self.supports_linear(linear):
            return None
        weight = linear.weight
        if not self._can_run(activation, weight):
            return None
        return self._run(activation, weight)


class _Qwen3DrafterSm80GateUp32Dispatch:
    """Selected exact-M32 A100 gate-up kernel with per-call fallthrough."""

    _CONFIG = "n64_s5"
    _MKN = (32, 1024, 6144)

    def __init__(self, device_index: int) -> None:
        from sglang.jit_kernel.drafter_sm80_bf16_gate_up32 import (
            _jit_drafter_sm80_bf16_gate_up32_module,
            drafter_sm80_bf16_gate_up32,
        )

        with torch.cuda.device(device_index):
            _jit_drafter_sm80_bf16_gate_up32_module()
            self._workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
        self._run = drafter_sm80_bf16_gate_up32

    @staticmethod
    def supports_linear(linear: nn.Module) -> bool:
        weight = getattr(linear, "weight", None)
        return bool(
            getattr(linear, "tp_size", None) == 1
            and isinstance(
                getattr(linear, "quant_method", None), UnquantizedLinearMethod
            )
            and getattr(linear, "bias", None) is None
            and not getattr(linear, "gather_output", False)
            and getattr(linear, "input_is_parallel", True)
            and not getattr(linear, "use_dp_attention_reduce", False)
            and isinstance(weight, torch.Tensor)
            and weight.is_cuda
            and weight.dtype == torch.bfloat16
            and weight.ndim == 2
            and weight.is_contiguous()
            and weight.data_ptr() % 16 == 0
        )

    def __call__(
        self, linear: nn.Module, activation: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self.supports_linear(linear):
            return None
        weight = linear.weight
        shape_mkn = (
            int(activation.shape[0]) if activation.ndim == 2 else -1,
            int(activation.shape[1]) if activation.ndim == 2 else -1,
            int(weight.shape[0]),
        )
        if (
            shape_mkn != self._MKN
            or not activation.is_cuda
            or activation.device != weight.device
            or activation.dtype != torch.bfloat16
            or not activation.is_contiguous()
            or activation.data_ptr() % 16
        ):
            return None
        output = torch.empty(
            (self._MKN[0], self._MKN[2]),
            dtype=torch.bfloat16,
            device=activation.device,
        )
        return self._run(
            self._CONFIG,
            output,
            activation,
            weight,
            self._workspace,
        )


class _Qwen3DrafterSm80Qkv32Dispatch(_Qwen3DrafterSm80GateUp32Dispatch):
    """Selected exact-M32 A100 QKV kernel with per-call fallthrough."""

    _CONFIG = "n128_s6"
    _MKN = (32, 1024, 4096)

    def __init__(self, device_index: int) -> None:
        from sglang.jit_kernel.drafter_sm80_bf16_qkv32 import (
            _jit_drafter_sm80_bf16_qkv32_module,
            drafter_sm80_bf16_qkv32,
        )

        with torch.cuda.device(device_index):
            _jit_drafter_sm80_bf16_qkv32_module()
            self._workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
        self._run = drafter_sm80_bf16_qkv32


class _Qwen3DrafterSm80Down32Dispatch(_Qwen3DrafterSm80GateUp32Dispatch):
    """Confirmed exact-M32 A100 down kernel with stable split-K workspace."""

    _CONFIG = "splitk4_n128_s5"
    _MKN = (32, 3072, 1024)

    def __init__(self, device_index: int) -> None:
        from sglang.jit_kernel.drafter_sm80_bf16_out_down32 import (
            DOWN_SPLITK4_WORKSPACE_BYTES,
            _jit_drafter_sm80_bf16_out_down32_module,
            drafter_sm80_bf16_down32,
        )

        with torch.cuda.device(device_index):
            _jit_drafter_sm80_bf16_out_down32_module()
            self._workspace = torch.empty(
                DOWN_SPLITK4_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda"
            )
        self._run = drafter_sm80_bf16_down32


class _Qwen3CublasLtPortfolioDispatch:
    """Fresh-process cached exact-shape cuBLASLt tactics for one worker.

    Subclasses pin the worker's portfolio shapes, its frozen tactic selector,
    and the SM-count target its tactics were discovered under.
    """

    _SM_COUNT_TARGET: int
    _FAMILY: str

    @staticmethod
    def _portfolio():
        raise NotImplementedError

    def __init__(
        self, device_index: int, representative_linears: Tuple[nn.Module, ...]
    ) -> None:
        # Keep this import after the worker fork. The helper owns a process-local
        # native handle and rejects inherited opaque algorithms.
        from sglang.jit_kernel.cublaslt_drafter_gemm import (
            MAX_ALGORITHMS,
            allocate_workspace,
            discover_algorithms,
            matmul,
        )

        portfolio_mkns, select_algorithm = self._portfolio()
        weights = {
            (int(linear.weight.shape[0]), int(linear.weight.shape[1])): linear.weight
            for linear in representative_linears
        }
        self._workspace = allocate_workspace(device_index)
        if self._workspace.data_ptr() % 256:
            raise RuntimeError(
                f"{self._FAMILY} cuBLASLt workspace must be 256-byte aligned"
            )
        self._algorithms = {}
        self._matmul = matmul
        for shape_mkn in portfolio_mkns:
            m, k, n = shape_mkn
            weight = weights.get((n, k))
            if weight is None:
                raise RuntimeError(
                    f"missing representative weight for cuBLASLt shape {shape_mkn}"
                )
            activation = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
            if activation.data_ptr() % 256:
                raise RuntimeError(
                    f"{self._FAMILY} cuBLASLt discovery activation must be "
                    "256-byte aligned"
                )
            candidates = discover_algorithms(
                activation,
                weight,
                sm_count_target=self._SM_COUNT_TARGET,
                top_n=MAX_ALGORITHMS,
                workspace=self._workspace,
            )
            algorithm = select_algorithm(shape_mkn, candidates)
            # Exercise the run path once before any CUDA-graph warmup/capture.
            matmul(
                activation,
                weight,
                algorithm=algorithm,
                sm_count_target=self._SM_COUNT_TARGET,
                workspace=self._workspace,
            )
            self._algorithms[shape_mkn] = algorithm
        torch.cuda.current_stream(device_index).synchronize()

    @staticmethod
    def supports_linear(linear: nn.Module) -> bool:
        weight = getattr(linear, "weight", None)
        return bool(
            getattr(linear, "tp_size", None) == 1
            and isinstance(
                getattr(linear, "quant_method", None), UnquantizedLinearMethod
            )
            and getattr(linear, "bias", None) is None
            and not getattr(linear, "gather_output", False)
            and getattr(linear, "input_is_parallel", True)
            and not getattr(linear, "use_dp_attention_reduce", False)
            and isinstance(weight, torch.Tensor)
            and weight.is_cuda
            and weight.dtype == torch.bfloat16
            and weight.ndim == 2
            and weight.is_contiguous()
            and weight.data_ptr() % 256 == 0
        )

    def __call__(
        self, linear: nn.Module, activation: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self.supports_linear(linear):
            return None
        weight = linear.weight
        if (
            activation.ndim != 2
            or not activation.is_cuda
            or activation.device != weight.device
            or activation.dtype != torch.bfloat16
            or not activation.is_contiguous()
            or activation.data_ptr() % 256
        ):
            return None
        shape_mkn = (
            int(activation.shape[0]),
            int(activation.shape[1]),
            int(weight.shape[0]),
        )
        algorithm = self._algorithms.get(shape_mkn)
        if algorithm is None:
            return None
        return self._matmul(
            activation,
            weight,
            algorithm=algorithm,
            sm_count_target=self._SM_COUNT_TARGET,
            workspace=self._workspace,
        )


class _Qwen3DrafterCublasLtDispatch(_Qwen3CublasLtPortfolioDispatch):
    """Selected target-52 tactics for the Qwen3-0.6B draft worker."""

    _SM_COUNT_TARGET = 52
    _FAMILY = "drafter"

    @staticmethod
    def _portfolio():
        from sglang.jit_kernel.cublaslt_drafter_gemm import (
            DRAFTER_CUBLASLT_PORTFOLIO_MKNS,
            select_drafter_portfolio_algorithm,
        )

        return DRAFTER_CUBLASLT_PORTFOLIO_MKNS, select_drafter_portfolio_algorithm


class _Qwen3VerifierCublasLtDispatch(_Qwen3CublasLtPortfolioDispatch):
    """Selected target-0 tactics for the Qwen3-8B target/verifier worker."""

    _SM_COUNT_TARGET = 0
    _FAMILY = "verifier"

    @staticmethod
    def _portfolio():
        from sglang.jit_kernel.cublaslt_drafter_gemm import (
            VERIFIER_CUBLASLT_PORTFOLIO_MKNS,
            select_verifier_portfolio_algorithm,
        )

        return VERIFIER_CUBLASLT_PORTFOLIO_MKNS, select_verifier_portfolio_algorithm


class _Qwen3PortableCublasLtDispatch:
    """Per-GPU correctness-gated tactics selected before graph capture."""

    supports_linear = staticmethod(_Qwen3CublasLtPortfolioDispatch.supports_linear)

    def __init__(
        self,
        device_index: int,
        representative_linears: Tuple[nn.Module, ...],
        *,
        shape_mkns: Tuple[Tuple[int, int, int], ...],
        worker: str,
        realized_sm_target: int,
        planning_context: str,
    ) -> None:
        from sglang.jit_kernel.cublaslt_autotune import (
            autotune_projection_portfolio,
        )
        from sglang.jit_kernel.cublaslt_drafter_gemm import (
            allocate_workspace,
            matmul,
        )

        self._workspace = allocate_workspace(device_index)
        if self._workspace.data_ptr() % 256:
            raise RuntimeError("portable cuBLASLt workspace must be 256-byte aligned")
        tuned = autotune_projection_portfolio(
            device_index=device_index,
            representative_linears=representative_linears,
            shape_mkns=shape_mkns,
            worker=worker,
            realized_sm_target=realized_sm_target,
            planning_context=planning_context,
            workspace=self._workspace,
            logger=logger,
        )
        self._algorithms = dict(tuned.algorithms)
        self._decisions = dict(tuned.decisions)
        self._matmul = matmul

    @property
    def selected_shape_count(self) -> int:
        return len(self._algorithms)

    def __call__(
        self, linear: nn.Module, activation: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self.supports_linear(linear):
            return None
        weight = linear.weight
        if (
            activation.ndim != 2
            or not activation.is_cuda
            or activation.device != weight.device
            or activation.dtype != torch.bfloat16
            or not activation.is_contiguous()
            or activation.data_ptr() % 256
        ):
            return None
        shape_mkn = (
            int(activation.shape[0]),
            int(activation.shape[1]),
            int(weight.shape[0]),
        )
        algorithm = self._algorithms.get(shape_mkn)
        if algorithm is None:
            return None
        return self._matmul(
            activation,
            weight,
            algorithm=algorithm,
            sm_count_target=algorithm.sm_count_target,
            workspace=self._workspace,
        )


_DrafterProjectionDispatch = Callable[[nn.Module, torch.Tensor], Optional[torch.Tensor]]


class _Qwen3ChainedProjectionDispatch:
    """Try projection dispatches in priority order; first non-None wins.

    Composition per exact call is TMA specialization -> retained cuBLASLt
    tactic -> production linear (the fallthrough when every dispatch
    returns None).
    """

    def __init__(self, *dispatches: _DrafterProjectionDispatch) -> None:
        flat: list = []
        for dispatch in dispatches:
            if dispatch is None:
                continue
            if isinstance(dispatch, _Qwen3ChainedProjectionDispatch):
                flat.extend(dispatch._dispatches)
            else:
                flat.append(dispatch)
        self._dispatches = tuple(flat)

    def __call__(
        self, linear: nn.Module, activation: torch.Tensor
    ) -> Optional[torch.Tensor]:
        for dispatch in self._dispatches:
            output = dispatch(linear, activation)
            if output is not None:
                return output
        return None


def _qwen3_drafter_projection_or_linear(
    dispatch: Optional[_DrafterProjectionDispatch],
    linear: nn.Module,
    activation: torch.Tensor,
    **linear_kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Dispatch predicates read data_ptr alignment and are not Dynamo-traceable;
    # torch.compile regions (the target's tc_piecewise prefill graphs) always
    # take the production linear, which keeps the portfolio surface confined to
    # the plainly captured decode/verify graphs it was selected on.
    if torch.compiler.is_compiling():
        return linear(activation, **linear_kwargs)

    # The diagnostic boundary encloses the dispatch and its production
    # fallback.  It therefore times whichever exact tactic the captured graph
    # actually selected, including any split-K reduction companion.
    with draft_extend_projection_scope(linear, activation):
        if dispatch is not None:
            output = dispatch(linear, activation)
            if output is not None:
                return output, None
        return linear(activation, **linear_kwargs)


# Compatibility alias for the earlier private TMA-only seam.
_qwen3_drafter_tma_or_linear = _qwen3_drafter_projection_or_linear


class Qwen3MLP(Qwen2MLP):
    """Qwen3-local MLP with an optional per-instance projection-dispatch seam."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            quant_config=quant_config,
            prefix=prefix,
        )
        self._drafter_projection_dispatch: Optional[_DrafterProjectionDispatch] = None

    def set_drafter_projection_dispatch(
        self, dispatch: _DrafterProjectionDispatch
    ) -> None:
        self._drafter_projection_dispatch = dispatch

    def set_drafter_tma_dispatch(self, dispatch: _Qwen3DrafterTmaDispatch) -> None:
        self.set_drafter_projection_dispatch(dispatch)

    def forward(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch = None,
    ) -> torch.Tensor:
        if get_global_server_args().rl_on_policy_target is not None:
            x = x.bfloat16()

        gate_up, _ = _qwen3_drafter_projection_or_linear(
            self._drafter_projection_dispatch,
            self.gate_up_proj,
            x,
        )
        x = self.act_fn(gate_up)
        x, _ = _qwen3_drafter_projection_or_linear(
            self._drafter_projection_dispatch,
            self.down_proj,
            x,
            forward_batch=forward_batch,
        )
        return x


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        start_layer: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = None,
        attention_bias: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.start_layer = start_layer
        self.tp_size = get_parallel().tp_size
        self.total_num_heads = num_heads
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_parallel().tp_rank

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )
        self.alt_stream = alt_stream
        self._drafter_projection_dispatch: Optional[_DrafterProjectionDispatch] = None

        self.use_fused_qk_norm_mrope = (
            _has_fused_qk_norm_mrope
            and isinstance(self.rotary_emb, MRotaryEmbedding)
            and getattr(self.rotary_emb, "mrope_section", None) is not None
        )
        if self.use_fused_qk_norm_mrope:
            # Scale tensors MUST stay on CPU: the C++ kernel uses .item<float>()
            # which triggers hipMemcpy D2H + sync on CUDA tensors, breaking graph capture.
            # Explicit device='cpu' is required because SGLang constructs models inside
            # a `with torch.device('cuda'):` context that changes the default device.
            self._fused_k_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")
            self._fused_v_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")

    def set_drafter_projection_dispatch(
        self, dispatch: _DrafterProjectionDispatch
    ) -> None:
        self._drafter_projection_dispatch = dispatch

    def set_drafter_tma_dispatch(self, dispatch: _Qwen3DrafterTmaDispatch) -> None:
        self.set_drafter_projection_dispatch(dispatch)

    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = _qwen3_drafter_projection_or_linear(
            self._drafter_projection_dispatch,
            self.qkv_proj,
            hidden_states,
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn.layer_id == self.start_layer:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            eps=self.q_norm.variance_epsilon,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_bias=getattr(self.q_norm, "bias", None),
            k_bias=getattr(self.k_norm, "bias", None),
        )
        return q, k, v

    def forward_prepare_aiter_fused_mrope(
        self, positions, hidden_states, forward_batch
    ):
        """Fused QK-norm + 3D mRoPE + KV cache write for decode (ROCm/aiter).

        The fused HIP kernel replaces split → QK norm → mRoPE → cache write,
        so KV is already in the paged cache when this returns.
        Returns (q, None, None); caller must pass save_kv_cache=False to attn.
        """
        qkv, _ = self.qkv_proj(hidden_states)
        num_tokens = qkv.shape[0]

        qkv_3d = qkv.view(num_tokens, -1, self.head_dim)

        token_to_kv_pool = get_token_to_kv_pool()
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(self.attn.layer_id)
        slot_mapping = forward_batch.out_cache_loc

        cos_sin = self.rotary_emb.cos_sin_cache
        if cos_sin.dtype != qkv.dtype:
            cos_sin = cos_sin.to(dtype=qkv.dtype)

        q_out = torch.empty(
            num_tokens,
            self.num_heads,
            self.head_dim,
            dtype=qkv.dtype,
            device=qkv.device,
        )

        fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
            qkv_3d,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin,
            positions,
            num_tokens,
            self.num_heads,
            self.num_kv_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rotary_emb.is_neox_style,
            self.rotary_emb.mrope_section,
            self.rotary_emb.mrope_interleaved,
            self.q_norm.variance_epsilon,
            q_out,
            k_cache,
            v_cache,
            slot_mapping,
            self._fused_k_scale,
            self._fused_v_scale,
            None,
            None,
            False,
            False,
            0,
            0,
        )

        q = q_out.reshape(num_tokens, -1)
        return q, None, None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if get_global_server_args().rl_on_policy_target is not None:
            hidden_states = hidden_states.bfloat16()

        save_kv_cache = True
        use_aiter_fused = (
            self.use_fused_qk_norm_mrope
            and forward_batch.forward_mode.is_decode()
            and get_global_server_args().rl_on_policy_target is None
        )

        if use_aiter_fused:
            q, k, v = self.forward_prepare_aiter_fused_mrope(
                positions, hidden_states, forward_batch
            )
            save_kv_cache = False
        elif not _is_npu:
            q, k, v = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
            )
        else:
            q, k, v = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        if get_global_server_args().rl_on_policy_target is not None:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)

        attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=save_kv_cache)
        output, _ = _qwen3_drafter_projection_or_linear(
            self._drafter_projection_dispatch,
            self.o_proj,
            attn_output,
        )
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_id: int = 0,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        if (
            hasattr(config, "rope_parameters")
            and config.rope_parameters
            and "rope_theta" in config.rope_parameters
        ):
            rope_theta = config.rope_parameters["rope_theta"]
            rope_scaling = config.rope_parameters
        else:
            rope_theta = getattr(config, "rope_theta", 1000000)
            rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
                override_orig_dtype=torch.float32,
                fp32_residual=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
            is_next_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            post_residual_addition=post_residual_addition,
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
            cache=(
                [self.mlp.gate_up_proj.weight, self.mlp.down_proj.weight]
                if _is_npu
                and check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                and (
                    hasattr(self.mlp.gate_up_proj, "weight")
                    and hasattr(self.mlp.down_proj, "weight")
                )
                else None
            ),
        )
        hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
        if _is_npu and get_cmo_stream():
            wait_cmo_stream()
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual


class Qwen3Model(Qwen2Model):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        alt_stream = torch.cuda.Stream() if _is_cuda else None
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen3DecoderLayer,
            alt_stream=alt_stream,
        )

    def _eligible_qwen3_drafter_layers(
        self,
        device_index: int,
        supports_linear: Callable[[nn.Module], bool],
    ) -> Optional[List[Qwen3DecoderLayer]]:
        return self._eligible_qwen3_layers(
            device_index,
            supports_linear,
            expected_config=(1024, 3072, 28),
            expected_weight_shapes=(
                (4096, 1024),
                (1024, 2048),
                (6144, 1024),
                (1024, 3072),
            ),
        )

    def _eligible_qwen3_verifier_layers(
        self,
        device_index: int,
        supports_linear: Callable[[nn.Module], bool],
    ) -> Optional[List[Qwen3DecoderLayer]]:
        return self._eligible_qwen3_layers(
            device_index,
            supports_linear,
            expected_config=(4096, 12288, 36),
            expected_weight_shapes=(
                (6144, 4096),
                (4096, 4096),
                (24576, 4096),
                (4096, 12288),
            ),
        )

    def _eligible_qwen3_layers(
        self,
        device_index: int,
        supports_linear: Callable[[nn.Module], bool],
        expected_config: Tuple[int, int, int],
        expected_weight_shapes: Tuple[Tuple[int, int], ...],
    ) -> Optional[List[Qwen3DecoderLayer]]:
        hidden_size, intermediate_size, num_layers = expected_config
        if (
            self.config.hidden_size != hidden_size
            or self.config.intermediate_size != intermediate_size
            or self.config.num_hidden_layers != num_layers
            or self.start_layer != 0
            or self.end_layer != num_layers
        ):
            return None

        eligible_layers = []
        for layer in self.layers:
            if not isinstance(layer, Qwen3DecoderLayer):
                return None
            projections = (
                layer.self_attn.qkv_proj,
                layer.self_attn.o_proj,
                layer.mlp.gate_up_proj,
                layer.mlp.down_proj,
            )
            if (
                tuple(tuple(item.weight.shape) for item in projections)
                != expected_weight_shapes
            ):
                return None
            if not all(supports_linear(item) for item in projections):
                return None
            if any(item.weight.device.index != device_index for item in projections):
                return None
            eligible_layers.append(layer)
        return eligible_layers if len(eligible_layers) == num_layers else None

    @staticmethod
    def _install_drafter_projection_dispatch(
        layers: List[Qwen3DecoderLayer],
        dispatch: _DrafterProjectionDispatch,
        prepend: bool = False,
    ) -> None:
        for layer in layers:
            for module in (layer.self_attn, layer.mlp):
                existing = getattr(module, "_drafter_projection_dispatch", None)
                combined = (
                    dispatch
                    if existing is None
                    else _Qwen3ChainedProjectionDispatch(
                        *((dispatch, existing) if prepend else (existing, dispatch))
                    )
                )
                module.set_drafter_projection_dispatch(combined)

    def enable_qwen3_drafter_tma(self, device_index: int) -> bool:
        """Install TMA dispatch only for the exact Qwen3-0.6B TP1 model."""

        layers = self._eligible_qwen3_drafter_layers(
            device_index, _Qwen3DrafterTmaDispatch.supports_linear
        )
        if layers is None:
            return False
        dispatch = _Qwen3DrafterTmaDispatch(device_index)
        # TMA specializations take priority over any installed cuBLASLt
        # portfolio; unsupported shapes fall through to it, then production.
        self._install_drafter_projection_dispatch(layers, dispatch, prepend=True)
        return True

    def enable_qwen3_drafter_sm80_gate_up32(self, device_index: int) -> bool:
        """Install the selected exact-M32 A100 gate-up specialization."""

        layers = self._eligible_qwen3_drafter_layers(
            device_index, _Qwen3DrafterSm80GateUp32Dispatch.supports_linear
        )
        if layers is None:
            return False
        dispatch = _Qwen3DrafterSm80GateUp32Dispatch(device_index)
        # The exact A100 gate-up specialization precedes any installed
        # cuBLASLt portfolio; all unsupported calls fall through unchanged.
        self._install_drafter_projection_dispatch(layers, dispatch, prepend=True)
        return True

    def enable_qwen3_drafter_sm80_qkv32(self, device_index: int) -> bool:
        """Install the selected exact-M32 A100 QKV specialization."""

        layers = self._eligible_qwen3_drafter_layers(
            device_index, _Qwen3DrafterSm80Qkv32Dispatch.supports_linear
        )
        if layers is None:
            return False
        dispatch = _Qwen3DrafterSm80Qkv32Dispatch(device_index)
        # The exact A100 QKV specialization composes with gate_up32 and
        # precedes any installed cuBLASLt portfolio. Other shapes fall through.
        self._install_drafter_projection_dispatch(layers, dispatch, prepend=True)
        return True

    def enable_qwen3_drafter_sm80_down32(self, device_index: int) -> bool:
        """Install the selected exact-M32 A100 down specialization."""

        layers = self._eligible_qwen3_drafter_layers(
            device_index, _Qwen3DrafterSm80Down32Dispatch.supports_linear
        )
        if layers is None:
            return False
        dispatch = _Qwen3DrafterSm80Down32Dispatch(device_index)
        # The exact A100 down specialization composes with gate_up32 and qkv32 and
        # precedes any installed cuBLASLt portfolio. Other shapes fall through.
        self._install_drafter_projection_dispatch(layers, dispatch, prepend=True)
        return True

    def enable_qwen3_drafter_cublaslt_portfolio(
        self,
        device_index: int,
        realized_sm_target: int,
        planning_context: str,
    ) -> bool:
        """Install portable per-GPU tactics for the exact drafter model."""

        layers = self._eligible_qwen3_drafter_layers(
            device_index, _Qwen3PortableCublasLtDispatch.supports_linear
        )
        if layers is None:
            return False
        first = layers[0]
        representative_linears = (
            first.self_attn.qkv_proj,
            first.self_attn.o_proj,
            first.mlp.gate_up_proj,
            first.mlp.down_proj,
        )
        from sglang.jit_kernel.cublaslt_drafter_gemm import DRAFTER_CUBLASLT_MKNS

        dispatch = _Qwen3PortableCublasLtDispatch(
            device_index,
            representative_linears,
            shape_mkns=DRAFTER_CUBLASLT_MKNS,
            worker="drafter",
            realized_sm_target=realized_sm_target,
            planning_context=planning_context,
        )
        self._install_drafter_projection_dispatch(layers, dispatch)
        return True

    def enable_qwen3_verifier_cublaslt_portfolio(
        self,
        device_index: int,
        realized_sm_target: int,
        planning_context: str,
    ) -> bool:
        """Install portable per-GPU tactics for the exact Qwen3-8B verifier."""

        layers = self._eligible_qwen3_verifier_layers(
            device_index, _Qwen3PortableCublasLtDispatch.supports_linear
        )
        if layers is None:
            return False
        first = layers[0]
        representative_linears = (
            first.self_attn.qkv_proj,
            first.self_attn.o_proj,
            first.mlp.gate_up_proj,
            first.mlp.down_proj,
        )
        from sglang.jit_kernel.cublaslt_drafter_gemm import VERIFIER_CUBLASLT_MKNS

        dispatch = _Qwen3PortableCublasLtDispatch(
            device_index,
            representative_linears,
            shape_mkns=VERIFIER_CUBLASLT_MKNS,
            worker="verifier",
            realized_sm_target=realized_sm_target,
            planning_context=planning_context,
        )
        self._install_drafter_projection_dispatch(layers, dispatch)
        return True


class Qwen3ForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def enable_qwen3_drafter_tma(self, device_index: int) -> bool:
        return self.model.enable_qwen3_drafter_tma(device_index)

    def enable_qwen3_drafter_sm80_gate_up32(self, device_index: int) -> bool:
        return self.model.enable_qwen3_drafter_sm80_gate_up32(device_index)

    def enable_qwen3_drafter_sm80_qkv32(self, device_index: int) -> bool:
        return self.model.enable_qwen3_drafter_sm80_qkv32(device_index)

    def enable_qwen3_drafter_sm80_down32(self, device_index: int) -> bool:
        return self.model.enable_qwen3_drafter_sm80_down32(device_index)

    def enable_qwen3_drafter_cublaslt_portfolio(
        self,
        device_index: int,
        realized_sm_target: int,
        planning_context: str,
    ) -> bool:
        return self.model.enable_qwen3_drafter_cublaslt_portfolio(
            device_index, realized_sm_target, planning_context
        )

    def enable_qwen3_verifier_cublaslt_portfolio(
        self,
        device_index: int,
        realized_sm_target: int,
        planning_context: str,
    ) -> bool:
        return self.model.enable_qwen3_verifier_cublaslt_portfolio(
            device_index, realized_sm_target, planning_context
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if not name.startswith("model.") and (
                name.startswith("layers.")
                or name.startswith("embed_tokens.")
                or name.startswith("norm.")
            ):
                name = add_prefix(name, "model")

            if name == "model.embed_tokens.weight":
                if self.pp_group.is_last_rank and self.config.tie_word_embeddings:
                    if "lm_head.weight" in params_dict:
                        param = params_dict["lm_head.weight"]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        if hasattr(self.model.embed_tokens, "weight"):
            del self.model.embed_tokens.weight
        if hasattr(self.lm_head, "weight"):
            del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )

        self.capture_aux_hidden_states = True
        # SGLang captures "before layer i". To capture the hidden state after target
        # layer `k` (HF-style), we capture before layer `k + 1`.
        self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = Qwen3ForCausalLM
