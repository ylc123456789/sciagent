"""Static evidence uses actual authorized bytes without importing user code."""

import hashlib
from pathlib import Path

import pytest

from sciagent_components.python_source_evidence import (
    MAX_FILE_BYTES, MAX_FILES, MAX_IMPORT_DEPTH, MAX_IMPORTS, MAX_TOTAL_BYTES,
    collect_python_source_evidence,
)
from sciagent_components.workspace import WorkspaceBoundary
from sciagent_contracts import WorkspaceAccess, WorkspaceGrant


def boundary(root, *, read=(".",), denied=()):
    return WorkspaceBoundary(WorkspaceGrant(
        root=str(root), source="local", access=WorkspaceAccess(
            read_paths=list(read), write_paths=[], denied_paths=list(denied),
        ),
    ))


def collect(root, command, **kwargs):
    return collect_python_source_evidence(command, boundary(root, **kwargs))


def files(evidence):
    return {Path(item["path"]).name: item for item in evidence.review_payload()["files"]}


def codes(evidence):
    return {item["code"] for item in evidence.review_payload()["gaps"]}


@pytest.mark.parametrize("command,entrypoint", [
    ("python3 --version && python3 -m unittest -v test_metric", "test_metric.py"),
    ("python3.12 -u measure.py", "measure.py"),
])
def test_test_and_measurement_include_actual_source_and_local_implementation(tmp_path, command, entrypoint):
    raw = b"def mse(a, b):\r\n    return sum((x-y)**2 for x,y in zip(a,b))/len(a)\r\n"
    (tmp_path / "metric.py").write_bytes(raw)
    source = "from metric import mse\nassert mse([0], [1]) == 1\n"
    (tmp_path / entrypoint).write_text(source)
    evidence = collect(tmp_path, command)
    actual = files(evidence)
    assert set(actual) == {entrypoint, "metric.py"}
    assert actual[entrypoint]["content"] == source
    assert actual["metric.py"]["content"] == raw.decode()
    assert actual["metric.py"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert actual["metric.py"]["path"] == (tmp_path / "metric.py").as_posix()
    assert not evidence.review_payload()["gaps"]


def test_collection_never_imports_user_code_or_runs_shell(tmp_path):
    marker = tmp_path / "executed"
    (tmp_path / "entry.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\nimport local\n"
    )
    (tmp_path / "local.py").write_text("raise RuntimeError('must not import')\n")
    evidence = collect(tmp_path, "python3 entry.py")
    assert set(files(evidence)) == {"entry.py", "local.py"}
    assert not marker.exists()
    assert "external_or_unresolved_import" in codes(evidence)


def test_package_initializers_relative_imports_and_namespace_are_read(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("from . import helper\n")
    (tmp_path / "pkg" / "helper.py").write_text("from .nested import leaf\n")
    (tmp_path / "pkg" / "nested").mkdir()
    (tmp_path / "pkg" / "nested" / "leaf.py").write_text("value=1\n")
    evidence = collect(tmp_path, "python -m unittest -v pkg.helper.TestCase.test_one")
    paths = {Path(item["path"]).relative_to(tmp_path).as_posix()
             for item in evidence.review_payload()["files"]}
    assert paths == {"pkg/__init__.py", "pkg/helper.py", "pkg/nested/leaf.py"}
    assert any(item.get("namespace_packages") for item in evidence.review_payload()["imports"])


def test_script_directory_is_local_import_root_and_absolute_entrypoint_is_supported(tmp_path):
    (tmp_path / "scripts").mkdir()
    entry = tmp_path / "scripts" / "main file.py"
    entry.write_text("import helper\n")
    (entry.parent / "helper.py").write_text("value=1\n")
    (tmp_path / "helper.py").write_text("raise RuntimeError('wrong root')\n")
    evidence = collect(tmp_path, f"python3 '{entry}' '&&' \"a\"b")
    assert files(evidence)["helper.py"]["path"] == (entry.parent / "helper.py").as_posix()
    assert evidence.review_payload()["coverage"][0]["argv"][-2:] == ["&&", "ab"]


@pytest.mark.parametrize("command", [
    "python3 -c 'print(1)'", "python3 -m unittest discover", "python3 -m unittest",
    "python3 -m pytest", "env python3 entry.py", "bash -c 'python3 entry.py'",
    "python3 entry.py | cat", "python3 entry.py > out", "python3 entry.py; true",
    "python3 $(echo entry.py)", "python3 entry.py\ntrue", "python3 entry.py &",
    "python3 entry.py &&", "", "python3 'unterminated",
])
def test_unsupported_commands_are_material_gaps_not_allow_or_deny_decisions(tmp_path, command):
    (tmp_path / "entry.py").write_text("pass\n")
    evidence = collect(tmp_path, command)
    assert evidence.review_payload()["gaps"]
    assert "decision" not in evidence.review_payload()


def test_missing_reserved_denied_and_symlink_escaped_sources_are_not_read(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-secret.py"
    outside.write_text("SECRET_DO_NOT_INCLUDE\n")
    (tmp_path / "escape.py").symlink_to(outside)
    (tmp_path / ".sciagent").mkdir()
    (tmp_path / ".sciagent" / "private.py").write_text("RESERVED\n")
    (tmp_path / "private.py").write_text("DENIED\n")
    for command in ("python3 escape.py", "python3 .sciagent/private.py", "python3 private.py",
                    "python3 missing.py", f"python3 {outside}", "python3 ../secret.py"):
        evidence = collect(tmp_path, command, denied=("private.py",))
        assert not files(evidence)
        assert evidence.review_payload()["gaps"]


def test_symlink_inside_boundary_uses_effective_scope_and_canonical_path(tmp_path):
    (tmp_path / "real.py").write_text("pass\n")
    (tmp_path / "alias.py").symlink_to(tmp_path / "real.py")
    assert files(collect(tmp_path, "python3 alias.py"))["real.py"]["path"] == (tmp_path / "real.py").as_posix()
    denied = collect(tmp_path, "python3 alias.py", denied=("real.py",))
    assert not files(denied)
    assert "source_unavailable" in codes(denied)


@pytest.mark.parametrize("name", [" entry.py", "entry.py ", "folder\\entry.py"])
def test_boundary_normalization_never_substitutes_a_different_entrypoint(tmp_path, name):
    (tmp_path / "entry.py").write_text("WRONG_SOURCE=1\n")
    (tmp_path / name).write_text("ACTUAL_SOURCE=2\n")
    evidence = collect(tmp_path, f"python3 '{name}'")
    assert not files(evidence)
    assert "entrypoint_path_unresolved" in codes(evidence)


@pytest.mark.parametrize("name", ["{safe,bad}.py", "entry*.py", "entry?.py", "entry[12].py", "~entry.py"])
def test_unquoted_shell_expansions_never_use_a_same_named_literal_file(tmp_path, name):
    (tmp_path / name).write_text("LITERAL_SOURCE=1\n")
    evidence = collect(tmp_path, f"python3 {name}")
    assert not files(evidence)
    assert "unsupported_shell_syntax" in codes(evidence)
    literal = collect(tmp_path, f"python3 '{name}'")
    assert files(literal)[name]["content"] == "LITERAL_SOURCE=1\n"


def test_escaped_glob_is_a_literal_path_without_shell_expansion(tmp_path):
    (tmp_path / "entry*.py").write_text("LITERAL_SOURCE=1\n")
    evidence = collect(tmp_path, r"python3 entry\*.py")
    assert files(evidence)["entry*.py"]["content"] == "LITERAL_SOURCE=1\n"


def test_symlink_entrypoint_import_root_is_canonical_directory(tmp_path):
    (tmp_path / "alias").mkdir()
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "main.py").write_text("import helper\n")
    (tmp_path / "real" / "helper.py").write_text("ACTUAL_SOURCE=1\n")
    (tmp_path / "alias" / "helper.py").write_text("WRONG_SOURCE=2\n")
    (tmp_path / "alias" / "main.py").symlink_to(tmp_path / "real" / "main.py")
    evidence = collect(tmp_path, "python3 alias/main.py")
    assert files(evidence)["helper.py"]["path"] == (tmp_path / "real" / "helper.py").as_posix()
    assert files(evidence)["helper.py"]["content"] == "ACTUAL_SOURCE=1\n"


def test_dependency_permission_failure_stays_visible_without_reading_body(tmp_path):
    (tmp_path / "entry.py").write_text("import private\n")
    (tmp_path / "private.py").write_text("PRIVATE_CONTENT\n")
    evidence = collect(tmp_path, "python entry.py", denied=("private.py",))
    assert set(files(evidence)) == {"entry.py"}
    assert "import_source_unavailable" in codes(evidence)
    assert "PRIVATE_CONTENT" not in str(evidence.review_payload())


def test_exact_file_read_grants_still_collect_readable_local_module(tmp_path):
    (tmp_path / "test_metric.py").write_text("from metric import mse\n")
    (tmp_path / "metric.py").write_text("mse=1\n")
    evidence = collect(tmp_path, "python -m unittest -v test_metric",
                       read=("test_metric.py", "metric.py"))
    assert set(files(evidence)) == {"test_metric.py", "metric.py"}
    # An ungranted package candidate cannot be inspected to prove it absent.
    assert "import_candidate_unavailable" in codes(evidence)


def test_invalid_encoding_and_syntax_are_explicit_gaps(tmp_path):
    (tmp_path / "syntax.py").write_text("def broken(\n")
    assert "source_parse_failed" in codes(collect(tmp_path, "python syntax.py"))
    (tmp_path / "encoded.py").write_bytes(b"# coding: invalid-codec\npass\n")
    assert "source_encoding_unavailable" in codes(collect(tmp_path, "python encoded.py"))


def test_non_utf8_source_hashes_original_bytes(tmp_path):
    raw = "# coding: latin-1\nlabel='caf\xe9'\n".encode("latin-1")
    (tmp_path / "entry.py").write_bytes(raw)
    source = files(collect(tmp_path, "python entry.py"))["entry.py"]
    assert source["content"] == raw.decode("latin-1")
    assert source["sha256"] == hashlib.sha256(raw).hexdigest()


def test_fingerprint_is_stable_but_tracks_file_changes_and_new_local_shadow(tmp_path):
    (tmp_path / "entry.py").write_text("import os\nimport helper\n")
    (tmp_path / "helper.py").write_text("value=1\n")
    one = collect(tmp_path, "python3 entry.py")
    assert one.fingerprint == collect(tmp_path, "python3 entry.py").fingerprint
    (tmp_path / "helper.py").write_text("value=2\n")
    two = collect(tmp_path, "python3 entry.py")
    assert two.fingerprint != one.fingerprint
    (tmp_path / "os.py").write_text("value=3\n")
    three = collect(tmp_path, "python3 entry.py")
    assert three.fingerprint != two.fingerprint
    assert "os.py" in files(three)
    (tmp_path / "os.py").unlink()
    assert collect(tmp_path, "python3 entry.py").fingerprint == two.fingerprint
    # Display consumers cannot mutate the collector's own evidence identity.
    payload = two.review_payload()
    payload["files"][0]["content"] = "changed display"
    assert two.review_payload()["files"][0]["content"] != "changed display"


def test_dynamic_import_and_search_path_changes_are_gaps(tmp_path):
    (tmp_path / "entry.py").write_text(
        "import sys\nimport importlib\nsys.path.append('/elsewhere')\n"
        "importlib.import_module('unknown')\nexec('pass')\n__import__('unknown')\n"
    )
    evidence = collect(tmp_path, "python3 entry.py")
    assert {"dynamic_code_or_import", "runtime_import_path"} <= codes(evidence)


def test_file_size_and_total_source_budget_are_bounded(tmp_path):
    (tmp_path / "huge.py").write_bytes(b"#" * (MAX_FILE_BYTES + 1))
    evidence = collect(tmp_path, "python huge.py")
    assert not files(evidence)
    assert "source_file_bytes_limit" in codes(evidence)
    (tmp_path / "entry.py").write_text("\n".join(f"import m{i}" for i in range(7)))
    for number in range(7):
        (tmp_path / f"m{number}.py").write_bytes(b"#" * (MAX_FILE_BYTES - 100))
    evidence = collect(tmp_path, "python entry.py")
    assert sum(item["size_bytes"] for item in evidence.review_payload()["files"]) <= MAX_TOTAL_BYTES
    assert "source_total_bytes_limit" in codes(evidence)


def test_file_import_and_recursion_budgets_are_explicit(tmp_path):
    (tmp_path / "entry.py").write_text("\n".join(f"import m{i}" for i in range(MAX_FILES + 3)))
    for number in range(MAX_FILES + 3):
        (tmp_path / f"m{number}.py").write_text("pass\n")
    evidence = collect(tmp_path, "python entry.py")
    assert len(evidence.review_payload()["files"]) == MAX_FILES
    assert "file_count_limit" in codes(evidence)
    for number in range(MAX_IMPORT_DEPTH + 3):
        (tmp_path / f"d{number}.py").write_text(f"import d{number + 1}\n")
    evidence = collect(tmp_path, "python d0.py")
    assert "import_depth_limit" in codes(evidence)
    (tmp_path / "entry.py").write_text("\n".join(f"import absent{i}" for i in range(MAX_IMPORTS + 10)))
    evidence = collect(tmp_path, "python entry.py")
    assert len(evidence.review_payload()["imports"]) <= MAX_IMPORTS
    assert "import_count_limit" in codes(evidence)


def test_version_commands_need_no_source_and_do_not_claim_a_safety_verdict(tmp_path):
    evidence = collect(tmp_path, "python --version && python3.12 -V")
    payload = evidence.review_payload()
    assert not payload["files"]
    assert not payload["gaps"]
    assert [item["status"] for item in payload["coverage"]] == ["interpreter_version_only"] * 2
    assert "not a complete dependency or behavior proof" in payload["scope"]


def test_new_local_unittest_runner_is_collected_and_changes_identity(tmp_path):
    (tmp_path / "test_metric.py").write_text("pass\n")
    command = "python3 -m unittest -v test_metric"
    before = collect(tmp_path, command)
    (tmp_path / "unittest").mkdir()
    (tmp_path / "unittest" / "__init__.py").write_text("value=1\n")
    (tmp_path / "unittest" / "__main__.py").write_text("raise RuntimeError('local runner')\n")
    after = collect(tmp_path, command)
    assert after.fingerprint != before.fingerprint
    assert {Path(item["path"]).relative_to(tmp_path).as_posix()
            for item in after.review_payload()["files"]} == {
                "test_metric.py", "unittest/__init__.py", "unittest/__main__.py",
            }
    assert "local_unittest_runner" in codes(after)


@pytest.mark.parametrize("descriptor", [None, {}, "file", [{"kind": "file", "path": "a", "extra": True}],
                                        [{"kind": "write", "path": "a"}], [{"kind": "file", "path": "a"}] * 4])
def test_invalid_inspection_descriptors_cannot_become_prepared_evidence(tmp_path, descriptor):
    if descriptor is None:
        assert collect_python_source_evidence("true", boundary(tmp_path), inspections=None)
    else:
        with pytest.raises(ValueError):
            collect_python_source_evidence("true", boundary(tmp_path), inspections=descriptor)


def test_requested_python_source_and_imports_are_bound_without_simulating_wrapped_command(tmp_path):
    (tmp_path / "entry.py").write_text("import helper\n")
    helper = tmp_path / "helper.py"
    helper.write_text("value = 1\n")
    descriptors = [{"kind": "file", "path": "entry.py"}]
    command = 'WS=$PWD; cd "$WS"; python entry.py | cat'
    original = collect_python_source_evidence(command, boundary(tmp_path), inspections=descriptors)
    assert set(files(original)) == {"entry.py", "helper.py"}
    assert original.review_payload()["gaps"]  # inspection does not claim to simulate Bash
    assert original.fingerprint == collect_python_source_evidence(command, boundary(tmp_path), inspections=descriptors).fingerprint
    helper.write_text("value = 2\n")
    assert original.fingerprint != collect_python_source_evidence(command, boundary(tmp_path), inspections=descriptors).fingerprint


def test_inspection_binds_supporting_file_and_directory_existence_without_executing(tmp_path):
    protocol = tmp_path / "protocol.txt"
    protocol.write_text("frozen protocol\n")
    descriptors = [{"kind": "directory", "path": "."}, {"kind": "file", "path": "protocol.txt"}]
    original = collect_python_source_evidence("true", boundary(tmp_path), inspections=descriptors)
    assert all(item["status"] == "available" for item in original.review_payload()["review_inspections"])
    assert files(original)["protocol.txt"]["content"] == "frozen protocol\n"
    protocol.unlink()
    changed = collect_python_source_evidence("true", boundary(tmp_path), inspections=descriptors)
    assert original.fingerprint != changed.fingerprint
    assert changed.review_payload()["review_inspections"][-1]["status"] == "unavailable"


@pytest.mark.parametrize("size", [36_079, 60_776, MAX_FILE_BYTES])
def test_large_source_is_complete_through_the_shared_per_file_boundary(tmp_path, size):
    body = b"#" * size
    (tmp_path / "entry.py").write_bytes(body)
    evidence = collect(tmp_path, "python entry.py")
    source = files(evidence)["entry.py"]
    assert source["content"].encode() == body
    assert source["size_bytes"] == size
    assert source["sha256"] == hashlib.sha256(body).hexdigest()
    assert not evidence.review_payload()["gaps"]


@pytest.mark.parametrize("extra", [0, 1])
def test_total_byte_boundary_and_failed_rereads_charge_unique_observed_files(tmp_path, extra):
    # Four files exactly fill the quota. Include aliases/repeated imports, plus
    # explicit rereads, to exercise dedup across acquisition paths.
    imports = "import a, b, c, alias\nimport a\n"
    (tmp_path / "entry.py").write_bytes(imports.encode() + b"#" * (MAX_FILE_BYTES - len(imports)))
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.py").write_bytes(b"#" * MAX_FILE_BYTES)
    (tmp_path / "alias.py").symlink_to(tmp_path / "a.py")
    (tmp_path / "supplement.txt").write_bytes(b"x" * extra)
    evidence = collect_python_source_evidence("python entry.py", boundary(tmp_path), inspections=[
        {"kind": "file", "path": "alias.py"},
        {"kind": "file", "path": "supplement.txt"},
        {"kind": "file", "path": "supplement.txt"},
    ])
    payload = evidence.review_payload()
    assert sum(item["size_bytes"] for item in payload["files"]) == MAX_TOTAL_BYTES
    assert len(payload["files"]) == 4 + (extra == 0)
    assert payload["review_inspections"][0]["source_path"] == (tmp_path / "a.py").as_posix()
    for observation in payload["review_inspections"][1:]:
        if extra:
            assert observation["status"] == "unavailable"
            assert observation["gap"] == {
                "code": "source_total_bytes_limit", "path": (tmp_path / "supplement.txt").as_posix(),
                "limit": MAX_TOTAL_BYTES, "remaining_bytes": 0, "size_bytes_at_least": 1,
            }
        else:
            assert observation["status"] == "available"


@pytest.mark.parametrize("kind", ["python", "text"])
def test_single_file_limit_diagnostic_has_path_limit_remaining_and_no_partial_body(tmp_path, kind):
    source = tmp_path / ("oversized.py" if kind == "python" else "oversized.txt")
    source.write_bytes(b"#" * (MAX_FILE_BYTES + 1))
    evidence = collect_python_source_evidence("true", boundary(tmp_path), inspections=[
        {"kind": "file", "path": source.name},
    ])
    payload = evidence.review_payload()
    assert not payload["files"]
    observation = payload["review_inspections"][0]
    assert observation["status"] == "unavailable"
    assert observation["reason"] == "source_file_bytes_limit"
    assert observation["gap"] == {
        "code": "source_file_bytes_limit", "path": source.as_posix(),
        "limit": MAX_FILE_BYTES, "remaining_bytes": MAX_TOTAL_BYTES,
        "size_bytes_at_least": MAX_FILE_BYTES + 1,
    }


def test_file_count_diagnostic_and_reread_at_exact_count_limit(tmp_path):
    (tmp_path / "entry.py").write_text("import " + ", ".join(f"p{i}" for i in range(MAX_FILES - 1)))
    for i in range(MAX_FILES - 1):
        (tmp_path / f"p{i}.py").write_text("pass\n")
    (tmp_path / "extra.txt").write_text("UNREAD_BODY")
    evidence = collect_python_source_evidence("python entry.py", boundary(tmp_path), inspections=[
        {"kind": "file", "path": "p0.py"}, {"kind": "file", "path": "extra.txt"},
    ])
    payload = evidence.review_payload()
    assert len(payload["files"]) == MAX_FILES
    assert payload["review_inspections"][0]["status"] == "available"
    failure = payload["review_inspections"][1]
    assert failure["reason"] == "file_count_limit"
    assert failure["gap"] == {
        "code": "file_count_limit", "path": (tmp_path / "extra.txt").as_posix(),
        "limit": MAX_FILES, "files_collected": MAX_FILES,
        "remaining_bytes": MAX_TOTAL_BYTES - sum(item["size_bytes"] for item in payload["files"]),
    }


def test_invalid_utf8_is_unavailable_and_does_not_double_charge_rereads(tmp_path, monkeypatch):
    from sciagent_components import python_source_evidence

    monkeypatch.setattr(python_source_evidence, "MAX_TOTAL_BYTES", 5)
    (tmp_path / "invalid.txt").write_bytes(b"\xff\xff")
    (tmp_path / "valid.txt").write_bytes(b"abc")
    evidence = collect_python_source_evidence("true", boundary(tmp_path), inspections=[
        {"kind": "file", "path": "invalid.txt"},
        {"kind": "file", "path": "invalid.txt"},
        {"kind": "file", "path": "valid.txt"},
    ])
    payload = evidence.review_payload()
    assert [item["status"] for item in payload["review_inspections"]] == [
        "unavailable", "unavailable", "available",
    ]
    assert all(item["reason"] == "source_encoding_unavailable" for item in payload["review_inspections"][:2])
    assert [item["content"] for item in payload["files"]] == ["abc"]


def test_shared_quota_changes_invalidate_the_prepared_fingerprint(tmp_path, monkeypatch):
    from sciagent_components import python_source_evidence

    (tmp_path / "entry.py").write_text("pass\n")
    original = collect(tmp_path, "python entry.py")
    for constant in ("MAX_FILES", "MAX_FILE_BYTES", "MAX_TOTAL_BYTES"):
        with monkeypatch.context() as patch:
            patch.setattr(python_source_evidence, constant, getattr(python_source_evidence, constant) + 1)
            assert collect(tmp_path, "python entry.py").fingerprint != original.fingerprint


@pytest.mark.parametrize("used", [0, 1, 2, 3])
def test_displayed_inspection_budget_tracks_all_attempts_without_mutating_evidence(tmp_path, used):
    # Failed inspections consume the same finite budget as available reads.
    inspections = [{"kind": "directory", "path": "missing"}] * used
    evidence = collect_python_source_evidence("true", boundary(tmp_path), inspections=inspections)
    expected = {"limit": 3, "used": used, "remaining": 3 - used}
    payload = evidence.review_payload()
    assert payload["inspection_budget"] == expected
    payload["inspection_budget"]["remaining"] = 99
    assert evidence.review_payload()["inspection_budget"] == expected
    if used:
        previous = collect_python_source_evidence("true", boundary(tmp_path), inspections=inspections[:-1])
        receipt = evidence.inspection_payload(previous)
        assert receipt["inspection_budget"] == expected
        assert receipt["inspection"]["status"] == "unavailable"
    assert evidence.fingerprint == collect_python_source_evidence(
        "true", boundary(tmp_path), inspections=inspections,
    ).fingerprint
