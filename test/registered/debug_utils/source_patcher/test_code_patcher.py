import functools
import inspect
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import yaml

from sglang.srt.debug_utils.source_patcher.code_patcher import (
    CodePatcher,
    _resolve_target,
    patch_function,
)
from sglang.srt.debug_utils.source_patcher.types import EditSpec, PatchConfig, PatchSpec
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu", nightly=True)
register_cpu_ci(est_time=7, suite="base-c-test-cpu")

SAMPLE_MODULE_NAME = "_source_patcher_test_fixtures.sample_module"

_decorations = []
_decorated_global = 11


def _tagged(fn):
    _decorations.append(fn.__name__)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return ("wrapped", fn(*args, **kwargs))

    return wrapper


@_tagged
@_tagged
def _decorated_plain(value):
    return value + _decorated_global


@torch.no_grad()
def _decorated_no_grad(value):
    return value * 2


class TestPatchFunction:
    def test_decorated_wraps_keeps_outer_identity_behavior_and_restoration(self):
        outer = _decorated_plain
        middle = outer.__wrapped__
        inner = inspect.unwrap(outer)
        original = inner.__code__
        wrapper_codes = (outer.__code__, middle.__code__)
        applications = list(_decorations)
        state = patch_function(
            target=outer,
            edits=[
                EditSpec(
                    match="return value + _decorated_global",
                    replacement="return value * _decorated_global",
                )
            ],
        )
        try:
            assert _decorated_plain is outer and outer.__wrapped__ is middle
            assert inspect.unwrap(outer) is inner
            assert (outer.__code__, middle.__code__) == wrapper_codes
            assert _decorated_plain(3) == ("wrapped", ("wrapped", 33))
            assert _decorations == applications  # Never re-run decorators.
        finally:
            state.restore()
        assert inner.__code__ is original
        assert _decorated_plain(3) == ("wrapped", ("wrapped", 14))

    def test_decorated_torch_no_grad_preserves_context_and_wrapper_globals(self):
        outer, inner = _decorated_no_grad, inspect.unwrap(_decorated_no_grad)
        original, wrapper_code = inner.__code__, outer.__code__
        keys = set(outer.__globals__)
        value = torch.tensor(2.0, requires_grad=True)
        with torch.enable_grad():
            assert not outer(value).requires_grad
            state = patch_function(
                target=outer,
                edits=[
                    EditSpec(
                        match="return value * 2",
                        replacement="return value * 3",
                    )
                ],
            )
            try:
                assert outer is _decorated_no_grad and outer.__code__ is wrapper_code
                result = outer(value)
                assert result.item() == 6 and not result.requires_grad
                assert set(outer.__globals__) == keys
            finally:
                state.restore()
            assert inner.__code__ is original
            assert outer(value).item() == 4 and not outer(value).requires_grad

    def test_decorated_actual_dspark_sampler_installs_real_fixture_and_restores(
        self, monkeypatch
    ):
        # CPU installation/dispatch regression for the exact decorated target;
        # not a checkpoint, head-arithmetic or CUDA component proof.
        from sglang.srt.debug_utils import b6_same_input_observer as observer
        from sglang.srt.speculative.dspark_worker_v2 import _DSparkDraftSampler

        fixture = (
            Path(__file__).resolve().parents[3] / "manual/b6_same_input_observer.yaml"
        )
        specs = PatchConfig(**yaml.safe_load(fixture.read_text())).patches
        spec = next(
            p for p in specs if p.target.endswith("_DSparkDraftSampler.__call__")
        )
        watch = observer.GraphWatch("cpu")
        ticket = observer.Ledger().begin(
            0,
            [SimpleNamespace(req_pool_idx=i, retraction_count=0) for i in range(1, 33)],
        )
        watch.activate(ticket)
        monkeypatch.setattr(observer, "_session", SimpleNamespace(draft=watch))

        def proposal(hidden, seed, confidence_threshold):
            assert torch.is_inference_mode_enabled()
            return SimpleNamespace(
                tokens=seed[:, None].expand(-1, 7),
                confidence_logits=torch.zeros(32, 7),
                lengths=torch.full((32,), 7, dtype=torch.int64),
            )

        model = SimpleNamespace(
            block_size=7,
            propose=proposal,
            get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty(1)),
        )
        sampler = _DSparkDraftSampler(model, 64)
        ids = torch.arange(32 * 7).view(32, 7)
        rows = torch.arange(1, 33)
        rows[3] = 999  # The real observer hook must execute and detect this row.
        batch = SimpleNamespace(
            batch_size=32, input_ids=ids.flatten(), req_pool_indices=rows
        )
        hidden = torch.zeros(32 * 7, 4)
        outer = _DSparkDraftSampler.__call__
        inner, wrapper_code = inspect.unwrap(outer), outer.__code__
        original = inner.__code__
        with CodePatcher(patches=[spec]):
            assert (
                _DSparkDraftSampler.__call__ is outer and outer.__code__ is wrapper_code
            )
            sampler(hidden, batch)
            assert torch.equal(sampler.tokens[:32], ids[:, :1].expand(-1, 7))
            assert watch.flags[0, 3, 4] == 1
        assert inner.__code__ is original
        watch.flags.zero_()
        sampler(hidden, batch)
        assert not watch.flags.any()

    def test_basic_patch_changes_behavior(self, sample_module: ModuleType) -> None:
        cls = sample_module.SampleClass
        obj = cls()
        assert obj.greet("world") == "hello world"

        state = patch_function(
            target=cls.greet,
            edits=[
                EditSpec(
                    match='greeting = f"hello {name}"',
                    replacement='greeting = f"patched {name}"',
                )
            ],
        )
        try:
            assert obj.greet("world") == "patched world"
        finally:
            state.restore()

        assert obj.greet("world") == "hello world"

    def test_globals_preserved_after_patch(self, sample_module: ModuleType) -> None:
        cls = sample_module.SampleClass
        obj = cls()
        assert obj.uses_global() == "value=global_value"

        state = patch_function(
            target=cls.uses_global,
            edits=[
                EditSpec(
                    match='return f"value={GLOBAL_VAR}"',
                    replacement='return f"patched_value={GLOBAL_VAR}"',
                )
            ],
        )
        try:
            assert obj.uses_global() == "patched_value=global_value"
        finally:
            state.restore()

    def test_function_identity_preserved(self, sample_module: ModuleType) -> None:
        cls = sample_module.SampleClass
        fn_id_before = id(cls.greet)

        state = patch_function(
            target=cls.greet,
            edits=[
                EditSpec(
                    match='greeting = f"hello {name}"',
                    replacement='greeting = f"patched {name}"',
                )
            ],
        )
        try:
            assert id(cls.greet) == fn_id_before
        finally:
            state.restore()

    def test_patch_standalone_function(self, sample_module: ModuleType) -> None:
        fn = sample_module.standalone_function
        assert fn(2, 3) == 5

        state = patch_function(
            target=fn,
            edits=[
                EditSpec(
                    match="return a + b",
                    replacement="return a * b",
                )
            ],
        )
        try:
            assert fn(2, 3) == 6
        finally:
            state.restore()

        assert fn(2, 3) == 5

    def test_patched_code_can_reference_global_variable(
        self, sample_module: ModuleType
    ) -> None:
        """Replacement code that references a module-level global should work."""
        cls = sample_module.SampleClass
        obj = cls()

        state = patch_function(
            target=cls.greet,
            edits=[
                EditSpec(
                    match='greeting = f"hello {name}"',
                    replacement='greeting = f"{GLOBAL_VAR} {name}"',
                )
            ],
        )
        try:
            assert obj.greet("world") == "global_value world"
        finally:
            state.restore()

    def test_patched_code_can_call_another_class_method(
        self, sample_module: ModuleType
    ) -> None:
        """Replacement code that calls HelperClass.format_value should work."""
        cls = sample_module.SampleClass
        obj = cls()

        state = patch_function(
            target=cls.greet,
            edits=[
                EditSpec(
                    match='greeting = f"hello {name}"',
                    replacement="greeting = HelperClass.format_value(name)",
                )
            ],
        )
        try:
            assert obj.greet("world") == "[world]"
        finally:
            state.restore()

    def test_patched_code_uses_helper_via_existing_method(
        self, sample_module: ModuleType
    ) -> None:
        """The uses_helper method already calls HelperClass; verify it survives patching."""
        cls = sample_module.SampleClass
        obj = cls()
        assert obj.uses_helper("test") == "[test]"

        state = patch_function(
            target=cls.uses_helper,
            edits=[
                EditSpec(
                    match="return HelperClass.format_value(value)",
                    replacement='return HelperClass.format_value("patched_" + value)',
                )
            ],
        )
        try:
            assert obj.uses_helper("test") == "[patched_test]"
        finally:
            state.restore()

        assert obj.uses_helper("test") == "[test]"


class TestResolveTarget:
    def test_resolve_class_method(self, sample_module: ModuleType) -> None:
        target = _resolve_target(f"{SAMPLE_MODULE_NAME}.SampleClass.greet")
        assert target is sample_module.SampleClass.greet

    def test_resolve_standalone_function(self, sample_module: ModuleType) -> None:
        target = _resolve_target(f"{SAMPLE_MODULE_NAME}.standalone_function")
        assert target is sample_module.standalone_function

    def test_resolve_nonexistent_raises(self, sample_module: ModuleType) -> None:
        with pytest.raises((ImportError, AttributeError)):
            _resolve_target(f"{SAMPLE_MODULE_NAME}.NonexistentClass.method")


class TestCodePatcher:
    @pytest.mark.parametrize(
        "fixture_name", ["gdn_compact_shadow_b5.yaml", "b6_same_input_observer.yaml"]
    )
    def test_complete_native_observer_fixture_installation_and_restoration(
        self, fixture_name
    ):
        # Installation only: do not invoke model/dumper/scheduler methods or load
        # weights. The B5 fixture can be read from its separately frozen lineage.
        directory = Path(__file__).resolve().parents[3] / "manual"
        path = directory / fixture_name
        if fixture_name == "gdn_compact_shadow_b5.yaml":
            path = Path(
                os.environ.get("SGLANG_TEST_B5_SOURCE_PATCHER_FIXTURE", str(path))
            )
            if not path.is_file():
                pytest.skip(
                    "B5 fixture is not in this lineage; pass its explicit fixture path"
                )
        specs = PatchConfig(**yaml.safe_load(path.read_text())).patches
        saved = []
        for spec in specs:
            outer = _resolve_target(spec.target)
            inner = inspect.unwrap(outer)
            saved.append((spec.target, outer, inner, outer.__code__, inner.__code__))
        try:
            with CodePatcher(patches=specs):
                for name, outer, inner, wrapper_code, body_code in saved:
                    assert _resolve_target(name) is outer
                    assert (
                        inspect.unwrap(outer) is inner
                        and inner.__code__ is not body_code
                    )
                    if outer is not inner:
                        assert outer.__code__ is wrapper_code
            for name, outer, inner, wrapper_code, body_code in saved:
                assert _resolve_target(name) is outer
                assert inner.__code__ is body_code and outer.__code__ is wrapper_code
        finally:
            # Keep this regression isolated even if a partial install fails;
            # the assertions above, not this cleanup, test normal restoration.
            for _, _, inner, _, body_code in saved:
                inner.__code__ = body_code

    def test_context_manager_patches_and_restores(
        self, sample_module: ModuleType
    ) -> None:
        cls = sample_module.SampleClass
        obj = cls()
        assert obj.greet("world") == "hello world"

        patches = [
            PatchSpec(
                target=f"{SAMPLE_MODULE_NAME}.SampleClass.greet",
                edits=[
                    EditSpec(
                        match='greeting = f"hello {name}"',
                        replacement='greeting = f"ctx_patched {name}"',
                    )
                ],
            )
        ]

        with CodePatcher(patches=patches):
            assert obj.greet("world") == "ctx_patched world"

        assert obj.greet("world") == "hello world"

    def test_context_manager_multiple_patches(self, sample_module: ModuleType) -> None:
        cls = sample_module.SampleClass
        obj = cls()

        patches = [
            PatchSpec(
                target=f"{SAMPLE_MODULE_NAME}.SampleClass.greet",
                edits=[
                    EditSpec(
                        match='greeting = f"hello {name}"',
                        replacement='greeting = f"p1 {name}"',
                    )
                ],
            ),
            PatchSpec(
                target=f"{SAMPLE_MODULE_NAME}.SampleClass.compute",
                edits=[
                    EditSpec(
                        match="result = x * 2 + 1",
                        replacement="result = x * 100",
                    )
                ],
            ),
        ]

        with CodePatcher(patches=patches):
            assert obj.greet("world") == "p1 world"
            assert obj.compute(5) == 500

        assert obj.greet("world") == "hello world"
        assert obj.compute(5) == 11

    def test_restores_on_exception(self, sample_module: ModuleType) -> None:
        cls = sample_module.SampleClass
        obj = cls()

        patches = [
            PatchSpec(
                target=f"{SAMPLE_MODULE_NAME}.SampleClass.greet",
                edits=[
                    EditSpec(
                        match='greeting = f"hello {name}"',
                        replacement='greeting = f"err_patched {name}"',
                    )
                ],
            )
        ]

        with pytest.raises(RuntimeError):
            with CodePatcher(patches=patches):
                assert obj.greet("world") == "err_patched world"
                raise RuntimeError("test error")

        assert obj.greet("world") == "hello world"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
