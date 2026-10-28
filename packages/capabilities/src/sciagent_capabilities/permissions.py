"""Run grants and deterministic operation rules share one tool policy."""

from sciagent_runtime import AllowListPermissionPolicy, PermissionDecision


class OperationPermissionPolicy(AllowListPermissionPolicy):
    """Prepare a bounded operation without performing or approving its effects."""

    def __init__(self, tools, *, boundary, request):
        super().__init__({tool.name for tool in tools})
        self.tools = {tool.name: tool for tool in tools}
        self.boundary = boundary
        self.answers = request.resume_answers

    def check(self, action, state, request, *, expected_context=None):
        decision = super().check(action, state, request)
        if decision.outcome == "deny":
            return decision
        if action.tool == "run_shell" and not request.permissions.execute_commands:
            return PermissionDecision(outcome="deny", reason="Process execution is not authorized")
        if action.tool == "run_shell" and not self.boundary.grant.access.unrestricted:
            return PermissionDecision(outcome="deny", reason="Script execution requires a fully readable/writable trusted workspace; scoped execution needs an isolation backend")
        tool = self.tools[action.tool]
        arguments = tool.input_model.model_validate(action.arguments)
        pending = state.pending_action
        pending_matches = (
            pending is not None and pending.run_id == request.run_id
            and pending.session_id == state.session_id
            and pending.task_id == request.task_id and pending.attempt_number == request.attempt_number
            and pending.tool == action.tool and pending.arguments == arguments.model_dump(mode="json")
        )
        context = {"cwd": str(self.boundary.root)}
        prepared = None
        if action.tool == "delete_path":
            prepared = tool.prepare(arguments)
            context["deletion"] = prepared
            if prepared["requires_confirmation"]:
                decision = PermissionDecision(outcome="ask", reason=f"Confirm deletion of {prepared['path']} ({len(prepared['entries'])} entries)")
        elif action.tool == "run_shell":
            review_context = expected_context
            if review_context is None and pending_matches:
                review_context = pending.context
            inspections = review_context.get("review_inspections") if review_context is not None else None
            context.update(tool.prepare(arguments, inspections=inspections) if inspections is not None else tool.prepare(arguments))
            prepared = dict(context)
            decision = PermissionDecision(outcome="ask", reason="Confirm this Bash script before execution")
        decision = decision.model_copy(update={"context": context, "prepared": prepared})
        if decision.outcome != "ask":
            return decision
        if not pending_matches or pending.context != context:
            return decision
        for answer in self.answers:
            if (answer.action != pending or answer.run_id != request.run_id
                    or answer.task_id != request.task_id or answer.attempt_number != request.attempt_number
                    or answer.session_id != state.session_id):
                continue
            approved = answer.values.get("approve", "").strip().lower() in {"yes", "true", "approve", "approved", "确认", "同意", "是"}
            return decision.model_copy(update={"outcome": "allow" if approved else "deny",
                                               "reason": "Operation approved" if approved else "User declined this operation",
                                               "approval_id": pending.action_id,
                                               "user_comment": answer.comment})
        return decision
