"""CPU structural gates for Qwen3.5 MoE EAGLE3 feature capture.

Compile the actual capture methods without importing GPU model constructors.
The fixture matches the 40-layer Qwen3.5-35B-A3B target and the SpecForge
checkpoint's first/middle/penultimate full-attention output-layer convention.
These tests do not establish checkpoint inference or graph correctness.
"""

import __future__
import ast
import contextlib
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_SRT = Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt"


def _method(path, class_name, method_name):
    tree = ast.parse((_SRT / path).read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    recorder = SimpleNamespace(with_current_layer=lambda _: contextlib.nullcontext())
    namespace = {"get_global_expert_distribution_recorder": lambda: recorder}
    exec(
        compile(
            module, str(_SRT / path), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    return namespace[method_name]


_EAGLE = _method(
    "models/qwen3_5.py",
    "Qwen3_5MoeForConditionalGeneration",
    "set_eagle3_layers_to_capture",
)
_DFLASH = _method(
    "models/qwen3_vl.py",
    "Qwen3VLForConditionalGeneration",
    "set_dflash_layers_to_capture",
)
_MARK = _method(
    "models/qwen3_5.py", "Qwen3_5ForCausalLM", "set_dflash_layers_to_capture"
)
_FORWARD = _method("models/qwen3_5.py", "Qwen3_5ForCausalLM", "forward")
_CAPTURE = _method(
    "layers/communicator.py",
    "LayerCommunicator",
    "prepare_attn_and_capture_last_layer_outputs",
)


def _wrapper(num_layers=40, last_rank=True):
    body = SimpleNamespace(
        layers=[SimpleNamespace() for _ in range(num_layers)], layers_to_capture=[]
    )
    body.set_dflash_layers_to_capture = MethodType(_MARK, body)
    wrapper = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=num_layers,
            layers_block_type=[
                "attention" if (i + 1) % 4 == 0 else "linear_attention"
                for i in range(num_layers)
            ],
        ),
        model=body,
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=last_rank),
        capture_aux_hidden_states=False,
    )
    wrapper.set_dflash_layers_to_capture = MethodType(_DFLASH, wrapper)
    return wrapper


def _marked(wrapper):
    return [
        i
        for i, layer in enumerate(wrapper.model.layers)
        if getattr(layer, "_is_layer_to_capture", False)
    ]


def _communicator():
    return SimpleNamespace(
        prepare_attn=lambda hidden, residual, *_args, **_kwargs: (
            hidden,
            hidden if residual is None else hidden + residual,
        ),
        _communicate_simple_fn=lambda *, hidden_states, **_kwargs: hidden_states,
        _post_attn_residual_is_read_only=lambda _: False,
        _context=None,
    )


class TestQwen35Eagle3Capture(unittest.TestCase):
    def test_pinned_b4_default_propagates_to_body_markers(self):
        wrapper = _wrapper()
        _EAGLE(wrapper)
        self.assertTrue(wrapper.capture_aux_hidden_states)
        self.assertEqual(wrapper.model.layers_to_capture, [4, 24, 36])
        self.assertEqual(_marked(wrapper), [4, 24, 36])

    def test_default_is_derived_from_full_attention_layout(self):
        wrapper = _wrapper(num_layers=64)
        _EAGLE(wrapper)
        self.assertEqual(_marked(wrapper), [4, 36, 60])

    def test_explicit_output_layers_override_native_default(self):
        wrapper = _wrapper()
        _EAGLE(wrapper, [0, 9, 20])
        self.assertEqual(wrapper.model.layers_to_capture, [1, 10, 21])
        self.assertEqual(_marked(wrapper), [1, 10, 21])

    def test_existing_dflash_propagation_is_unchanged(self):
        wrapper = _wrapper()
        wrapper.set_dflash_layers_to_capture([1, 8, 22])
        self.assertTrue(wrapper.capture_aux_hidden_states)
        self.assertEqual(_marked(wrapper), [2, 9, 23])

    def test_non_last_pipeline_rank_does_not_enable_capture(self):
        wrapper = _wrapper(last_rank=False)
        _EAGLE(wrapper)
        self.assertFalse(wrapper.capture_aux_hidden_states)
        self.assertEqual(_marked(wrapper), [])

    def test_ambiguous_short_default_fails_before_mutation(self):
        wrapper = _wrapper(num_layers=16)
        with self.assertRaisesRegex(ValueError, "explicit layer_ids"):
            _EAGLE(wrapper)
        self.assertFalse(wrapper.capture_aux_hidden_states)
        self.assertEqual(_marked(wrapper), [])
        _EAGLE(wrapper, [0, 3, 5])
        self.assertEqual(_marked(wrapper), [1, 4, 6])

    def test_actual_forward_routes_previous_layer_outputs(self):
        class Layer:
            def __init__(self, index):
                self.index = index

            def __call__(
                self, *, hidden_states, residual, captured_last_layer_outputs, **kwargs
            ):
                _CAPTURE(
                    _communicator(),
                    hidden_states,
                    residual,
                    None,
                    captured_last_layer_outputs,
                )
                return torch.tensor([float(self.index + 1)]), None

        wrapper = _wrapper()
        body = wrapper.model
        body.layers = [Layer(i) for i in range(40)]
        body.pp_group = wrapper.pp_group
        body.start_layer, body.end_layer = 0, 40
        body.norm = lambda hidden: hidden
        _EAGLE(wrapper)
        output, captured = _FORWARD(
            body, None, None, None, input_embeds=torch.tensor([0.0])
        )
        self.assertEqual(output.item(), 40.0)
        # Output of layer 3 is read on entry to layer 4, not after layer 4.
        self.assertEqual([value.item() for value in captured], [4.0, 24.0, 36.0])

    def test_shared_capture_keeps_residual_sum_before_later_mutation(self):
        captured = []
        _, residual = _CAPTURE(
            _communicator(), torch.tensor([3.0]), torch.tensor([5.0]), None, captured
        )
        residual.add_(100)
        self.assertEqual(captured[0].item(), 8.0)

    def test_b1_qwen3_default_remains_distinct(self):
        setter = _method(
            "models/qwen3.py", "Qwen3ForCausalLM", "set_eagle3_layers_to_capture"
        )
        wrapper = _wrapper()
        setter(wrapper)
        self.assertEqual(wrapper.model.layers_to_capture, [2, 20, 37])
        self.assertEqual(_marked(wrapper), [])


if __name__ == "__main__":
    unittest.main()
