"""Project dependency and public-export boundaries, with one rule per package."""

import ast
from importlib import import_module
import inspect
from pathlib import Path

import pytest

from resagent2_runtime.models import RuntimeModel


ROOT = Path(__file__).parents[2]
PACKAGE_PATHS = {
    **{name: f"packages/{name}/src" for name in
       ("contracts", "runtime", "components", "capabilities", "orchestrator")},
    **{name: f"packages/agents/{name}/src" for name in ("coding", "experiment", "scientific", "executor")},
}
# Preserve the existing per-package policy: sharing the scanner does not grant
# one package another's dependencies. Components keeps its forbidden-root rule.
ALLOWED_IMPORT_ROOTS = {
    "capabilities": {
        "__future__", "collections", "dataclasses", "datetime", "defusedxml", "email", "hashlib",
        "json", "logging", "mimetypes", "os", "pathlib", "platform", "pydantic",
        "resagent2_components", "resagent2_contracts", "resagent2_runtime", "shlex", "shutil",
        "signal", "subprocess", "tempfile", "threading", "time", "typing", "urllib", "uuid",
    },
    "coding": {
        "__future__", "datetime", "hashlib", "json", "pathlib", "pydantic",
        "resagent2_capabilities", "resagent2_components", "resagent2_contracts",
        "resagent2_runtime", "resagent2_executor", "shlex", "time", "typing", "uuid",
    },
    "contracts": {
        "__future__", "collections", "datetime", "enum", "hashlib", "json", "os", "pathlib", "pydantic",
        "shlex", "typing", "uuid",
    },
    "experiment": {
        "__future__", "hashlib", "json", "pathlib", "pydantic", "resagent2_capabilities",
        "resagent2_components", "resagent2_contracts", "resagent2_runtime", "resagent2_executor", "typing",
    },
    "executor": {
        "__future__", "collections", "dataclasses", "datetime", "enum", "hashlib", "json",
        "pathlib", "pydantic", "resagent2_capabilities", "resagent2_components",
        "resagent2_contracts", "resagent2_runtime", "shlex", "time", "typing", "uuid",
    },
    "orchestrator": {
        "__future__", "collections", "contextlib", "contextvars", "dataclasses", "datetime", "enum",
        "fcntl", "graphlib", "hashlib", "importlib", "json", "math", "os", "pathlib", "pydantic",
        "resagent2_contracts", "shutil", "sys", "tempfile", "threading", "typing", "urllib", "uuid",
    },
    "runtime": {
        "__future__", "asyncio", "collections", "contextlib", "contextvars", "dataclasses",
        "datetime", "fcntl", "hashlib", "httpx", "ipaddress", "json", "logging", "math", "os",
        "pathlib", "pydantic", "re", "resagent2_contracts", "socket", "tempfile", "threading",
        "time", "typing", "urllib", "uuid",
    },
    "scientific": {
        "__future__", "collections", "json", "pathlib", "pydantic", "resagent2_capabilities",
        "resagent2_components", "resagent2_contracts", "resagent2_runtime", "typing",
    },
}
COMPONENTS_FORBIDDEN_ROOTS = {
    "resagent2_capabilities", "resagent2_cli", "resagent2_coding", "resagent2_experiment",
    "resagent2_orchestrator", "resagent2_scientific", "resagent2_executor",
}
ORCHESTRATOR_ALLOWED_MODULES = {
    "resagent2_components.artifacts", "resagent2_components.literature",
    "resagent2_components.materials", "resagent2_components.workspace", "resagent2_runtime.budget",
    "resagent2_runtime.cancellation", "resagent2_runtime.lifecycle",
}


@pytest.mark.parametrize("package", PACKAGE_PATHS)
def test_package_dependency_boundary(package):
    for path in (ROOT / PACKAGE_PATHS[package]).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if package == "orchestrator" and node.module in ORCHESTRATOR_ALLOWED_MODULES:
                    continue
                roots = {node.module.split(".")[0]}
            else:
                continue
            if package == "components":
                forbidden = roots & COMPONENTS_FORBIDDEN_ROOTS
            else:
                forbidden = roots - ALLOWED_IMPORT_ROOTS[package]
            assert not forbidden, (path, node.lineno, forbidden)


@pytest.mark.parametrize("package", [
    "contracts", "runtime", "components", "capabilities", "orchestrator", "coding", "experiment", "executor",
])
def test_public_classes_have_docstrings(package):
    module = import_module(f"resagent2_{package}")
    missing = [name for name in module.__all__
               if inspect.isclass(getattr(module, name))
               and not inspect.getdoc(getattr(module, name))]
    assert not missing, (package, missing)


def test_public_components_are_not_tool_entry_points():
    import resagent2_components

    for name in resagent2_components.__all__:
        value = getattr(resagent2_components, name)
        if inspect.isclass(value):
            assert not hasattr(value, "input_model"), name


def test_capabilities_export_only_tools_and_their_inputs() -> None:
    """Operations must be imported from components, not forwarded by tools."""
    import resagent2_capabilities

    public = [getattr(resagent2_capabilities, name) for name in resagent2_capabilities.__all__]
    tools = [value for value in public if hasattr(value, "input_model")]
    inputs = {tool.input_model for tool in tools}
    assert set(public) == set(tools) | inputs
    assert all(issubclass(value, RuntimeModel) for value in inputs)
    assert all(callable(tool.execute) for tool in tools)


def test_capabilities_do_not_define_operation_classes() -> None:
    """Small tool helpers are fine; service classes belong in components."""
    for source_file in (ROOT / PACKAGE_PATHS["capabilities"]).rglob("*.py"):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            is_input = any(isinstance(base, ast.Name) and base.id == "RuntimeModel" for base in node.bases)
            is_tool = any(isinstance(item, ast.FunctionDef) and item.name == "execute" for item in node.body)
            is_permission_adapter = (
                source_file.name == "permissions.py"
                and node.name == "OperationPermissionPolicy"
            )
            assert is_input or is_tool or is_permission_adapter, (source_file, node.name)
