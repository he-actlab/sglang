from types import ModuleType
import functools
import inspect

import pytest
import torch

from sglang.srt.debug_utils.source_patcher.code_patcher import (
    CodePatcher,
    _resolve_target,
    patch_function,
)
from sglang.srt.debug_utils.source_patcher.types import EditSpec, PatchSpec
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu", nightly=True)
register_cpu_ci(est_time=7, suite="base-c-test-cpu")

SAMPLE_MODULE_NAME = "_source_patcher_test_fixtures.sample_module"
DECORATIONS = []
BODY_GLOBAL = 3


def counted_decorator(fn):
    DECORATIONS.append(fn)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapped


@counted_decorator
@torch.no_grad()
def decorated_forward(value):
    return value * BODY_GLOBAL


class TestPatchFunction:
    def test_decorated_native_globals_wrapper_semantics_and_restore(self):
        outer = decorated_forward
        middle = outer.__wrapped__
        inner = inspect.unwrap(outer)
        identities = (outer.__code__, outer.__closure__, middle.__code__, middle.__closure__, inner.__code__)
        value = torch.tensor(2.0, requires_grad=True)
        before = len(DECORATIONS)
        state = patch_function(target=outer, edits=[EditSpec(
            match="return value * BODY_GLOBAL", replacement="return value * (BODY_GLOBAL + 1)")])
        try:
            assert state.target_fn is inner
            assert decorated_forward is outer and outer.__wrapped__ is middle
            assert (outer.__code__, outer.__closure__, middle.__code__, middle.__closure__) == identities[:4]
            result = outer(value)
            assert result.item() == 8 and not result.requires_grad
            assert len(DECORATIONS) == before  # Never re-evaluate decorators.
        finally:
            state.restore()
        assert inner.__code__ is identities[4]
        assert outer(value).item() == 6 and not outer(value).requires_grad

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
    def test_later_install_failure_restores_earlier_native_code(self, sample_module):
        fn = sample_module.SampleClass.greet
        original = fn.__code__
        patches = [PatchSpec(target=f"{SAMPLE_MODULE_NAME}.SampleClass.greet", edits=[EditSpec(
            match='greeting = f"hello {name}"', replacement='greeting = f"changed {name}"')]),
            PatchSpec(target=f"{SAMPLE_MODULE_NAME}.does_not_exist", edits=[])]
        with pytest.raises(AttributeError):
            with CodePatcher(patches=patches):
                pytest.fail("Installation must fail")
        assert fn.__code__ is original
        assert sample_module.SampleClass().greet("world") == "hello world"

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
