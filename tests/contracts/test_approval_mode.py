"""Operation review is a persisted caller selection, not an execution grant."""

import pytest
from pydantic import ValidationError

from resagent2_contracts import AgentPermissions, RunPermissions


@pytest.mark.parametrize("permission_type", [RunPermissions, AgentPermissions])
def test_review_defaults_to_manual_without_granting_execution(permission_type):
    permissions = permission_type()
    assert permissions.approval_mode == "manual"
    assert not permissions.execute_commands


@pytest.mark.parametrize("mode", ["manual", "auto_review"])
def test_review_mode_survives_run_to_agent_permission_and_json_roundtrip(mode):
    permissions = RunPermissions(approval_mode=mode)
    restored = RunPermissions.model_validate_json(permissions.model_dump_json())
    child = AgentPermissions(**restored.model_dump(exclude={"schema_version"}))
    assert child.approval_mode == mode
    assert not child.execute_commands


@pytest.mark.parametrize("permission_type", [RunPermissions, AgentPermissions])
@pytest.mark.parametrize("invalid", ["allow_all", "auto", None])
def test_review_rejects_unrecognized_modes(permission_type, invalid):
    with pytest.raises(ValidationError, match="approval_mode"):
        permission_type(approval_mode=invalid)


@pytest.mark.parametrize("permission_type", [RunPermissions, AgentPermissions])
def test_permissions_reject_legacy_confirmation_toggle(permission_type):
    with pytest.raises(ValidationError, match="confirm_commands"):
        permission_type(confirm_commands=True)
