"""Small CPU references/edge cases; never loads the real checkpoint or CUDA."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.debug_utils import b2_layer0_reference as ref


def counts(values, centers, radii=None):
    y = torch.tensor(values, dtype=torch.bfloat16, device="cpu")
    r = torch.tensor(centers, dtype=torch.float64, device="cpu")
    e = torch.zeros_like(r) if radii is None else torch.tensor(radii, dtype=torch.float64, device="cpu")
    return ref.rounding_counts(y, r, e)


def test_midpoint_even_and_odd_are_not_cast_guesses():
    result = counts([1.0, 1 + 2**-7], [1 + 2**-8] * 2)
    assert result["guaranteed_cell"] == result["excluded_cell"] == 1
    assert result["central_midpoint"] == 2
    assert result["steps_0"] == result["steps_1"] == 1
    assert result["ambiguous_cell"] == 0


def test_uncertainty_straddling_midpoint_is_explicitly_ambiguous():
    result = counts([1.0, 1 + 2**-7], [1 + 2**-8] * 2, [1e-12] * 2)
    assert result["ambiguous_cell"] == 2
    assert result["guaranteed_cell"] == result["excluded_cell"] == 0


@pytest.mark.parametrize("sign", [1, -1])
def test_asymmetric_power_of_two_cells_and_sign(sign):
    result = counts([sign, sign], [sign * (1 - 2**-9), sign * (1 + 2**-8)])
    assert result["guaranteed_cell"] == 2 and result["central_midpoint"] == 2
    assert result["steps_0"] == 2


def test_subnormal_midpoint_and_signed_zero():
    result = counts([0, 2**-133], [2**-134] * 2)
    assert result["guaranteed_cell"] == result["excluded_cell"] == 1
    zeros = counts([-0.0, 0.0], [0.0, 0.0])
    assert zeros["guaranteed_cell"] == zeros["steps_0"] == 2
    assert zeros["negative_zero"] == 1


def test_finite_extremes_and_outside_domain_reject():
    largest = torch.finfo(torch.bfloat16).max
    result = counts([-largest, largest], [-largest, largest])
    assert result["guaranteed_cell"] == result["steps_0"] == 2
    with pytest.raises(ValueError, match="outside finite"):
        counts([largest], [largest * 2])


@pytest.mark.parametrize("bad", ["dtype", "nan", "negative_radius", "shape"])
def test_malformed_rounding_inputs_reject(bad):
    y, r, e = torch.ones(2, dtype=torch.bfloat16), torch.ones(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    if bad == "dtype": y = y.float()
    if bad == "nan": r[0] = float("nan")
    if bad == "negative_radius": e[0] = -1
    if bad == "shape": e = e[:1]
    with pytest.raises(ValueError): ref.rounding_counts(y, r, e)


def test_error_reduction_sumabs_uncertainty_and_zero_norm():
    r = torch.tensor([1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([1.0, 2.0 + 2**-6], dtype=torch.bfloat16)
    total = {}
    for i in range(2):
        ref.merge_statistics(total, ref.block_statistics(y[i:i+1], r[i:i+1], r[i:i+1].abs()))
    result = ref.finish_statistics(total)
    assert result["squared_error"] == (2**-6)**2
    assert result["relative_l2_error"] == ((2**-6)**2 / 5)**0.5
    assert result["max_abs"] == 2**-6 and result["max_steps"] == 1
    assert result["max_fp64_uncertainty"] > 0
    zero = ref.finish_statistics(ref.block_statistics(torch.zeros(2, dtype=torch.bfloat16), torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)))
    assert zero["relative_l2_error"] is None and zero["reference_l2_zero"]
    assert zero["max_fp64_uncertainty"] == 0
    with pytest.raises(ValueError, match="sumabs"):
        ref.block_statistics(y, r, torch.zeros_like(r))


def test_native_tp1_loader_orientation_with_synthetic_weights():
    from sglang.srt.layers.linear import QKVParallelLinear
    owner = SimpleNamespace(num_heads=2, head_size=2, num_kv_heads=1, v_head_size=2,
                            tp_rank=0, num_kv_head_replicas=1, use_presharded_weights=False)
    weight = torch.nn.Parameter(torch.empty(8, 3, dtype=torch.bfloat16), requires_grad=False)
    weight.output_dim = 0
    pieces = [torch.arange(n * 3, dtype=torch.bfloat16).reshape(n, 3) + 10*i for i, n in enumerate((4, 2, 2))]
    for shard, piece in zip("qkv", pieces):
        QKVParallelLinear.weight_loader(owner, weight, piece, shard)
    assert torch.equal(weight, torch.cat(pieces))
    x = torch.tensor([[1, -2, 3]], dtype=torch.float64)
    expected = torch.tensor([[sum(float(x[0, j]) * float(w[j]) for j in range(3)) for w in torch.cat(pieces)]], dtype=torch.float64)
    assert torch.equal(x @ weight.double().T, expected)


def snapshots():
    return [{name: torch.zeros(ref.ROWS, width, dtype=torch.bfloat16) for name, width in
             (("embedding", ref.K), ("input_norm", ref.K), ("raw_qkv", ref.N), ("post_rope_k", 1024), ("v", 1024))} for _ in range(8)]


@pytest.mark.parametrize("bad", [None, "missing", "different_norm", "shape", "dtype", "nonfinite"])
def test_fixed_first_calls_source_and_payload(bad):
    saved = snapshots()
    if bad == "missing": saved.pop()
    if bad == "different_norm": saved[1]["input_norm"][0, 0] = 1
    if bad == "shape": saved[0]["raw_qkv"] = saved[0]["raw_qkv"][:1]
    if bad == "dtype": saved[0]["input_norm"] = saved[0]["input_norm"].float()
    if bad == "nonfinite": saved[0]["input_norm"][0, 0] = float("inf")
    if bad is not None:
        with pytest.raises(ValueError): ref.selected_inputs(saved)
    else:
        x, outputs, bound = ref.selected_inputs(saved)
        assert x is saved[0]["input_norm"] and outputs[0] is saved[0]["raw_qkv"] and outputs[1] is saved[1]["raw_qkv"]
        assert bound <= 128 << 20
    with pytest.raises(ValueError, match="payload"):
        ref.payload_bound(128 << 20)


def test_no_native_forward_cuda_alloc_or_all_weight_fp64_loading():
    tree = ast.parse(Path(ref.__file__).read_text())
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    for forbidden in ("recorded.build_native", "recorded.native_chain", "torch.cuda.init", "weights.get_tensor"):
        assert forbidden not in calls
    assert "weights.get_slice" in calls
    assert "torch.autocast" in calls
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("key,bad", [("call_indices", [1, 0]), ("call_indices", [False, True]),
    ("output_block", 128), ("output_block", 256.0), ("tensor_payload_limit_bytes", 256 << 20),
    ("performance_eligible", True)])
def test_changed_reference_contract_rejects_before_any_runtime_or_weights(tmp_path, key, bad):
    plan = dict(call_indices=[0, 1], output_block=256, tensor_payload_limit_bytes=128 << 20, performance_eligible=False)
    plan[key] = bad
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="contract"):
        ref.validate_plan(path)
