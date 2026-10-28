"""Native protocol fixture builders for deterministic tests and mock E2E."""

import json
from uuid import uuid4

from pydantic import BaseModel

from sciagent_runtime import ToolCallTurn


def tool_turn(tool, arguments=None, *, call_id=None):
    return ToolCallTurn(tool_calls=[{
        "id": call_id or f"call_{uuid4().hex}", "name": tool,
        "arguments": json.dumps(arguments or {}),
    }])


def tool_turns(actions):
    actions = [action.model_dump(mode="json") if isinstance(action, BaseModel) else action
               for action in actions]
    return [tool_turn(action["tool"], action.get("arguments")) for action in actions]


from uuid import uuid4
from sciagent_runtime import ScriptedLLMClient


class ParentQuestionClient(ScriptedLLMClient):
    """Scripted Scientific escalates child questions, then replies to the exact child.

    Root question and child question remain distinct identities. The model-visible
    notification drives the reply, so fixture code never writes Run control state.
    """

    def __init__(self, turns):
        super().__init__(turns)
        self._asked_child_ids = set()

    def next_tool_call(self, context, schemas, turns, *, max_input_tokens):
        import json
        from e2e.native_fixtures import tool_turn
        questions = json.loads(context.pending_notifications or "[]")
        if questions:
            question = questions[0]
            if question["id"] not in self._asked_child_ids:
                self._asked_child_ids.add(question["id"])
                result = tool_turn("ask_user", {
                    "assessment": {"statement": "The child needs an upstream decision"},
                    "text": question["text"], "requested_fields": question["requested_fields"],
                    "related_question_id": question["id"],
                    "options": question.get("options"),
                }, call_id="call_parent_ask_" + uuid4().hex)
            else:
                task = json.loads(context.task_context.split("\n", 1)[1])
                answer = next(a for a in reversed([event["record"] for event in task["business_dialogue"] if event["type"] == "answer"]) if a["question_text"] == question["text"])
                authority = json.loads(context.runtime_context.split("\n", 1)[1])
                result = tool_turn("update_work", {
                    "work_request_id": authority["child_work"]["active_work_request_id"],
                    "answers": [{"task_id": question["task_id"], "question_id": question["id"], "values": answer["values"]}],
                }, call_id="call_parent_reply_" + uuid4().hex)
            self.contexts.append(context)
            self.schemas.append(schemas)
            self.histories.append([turn.model_copy(deep=True) for turn in turns])
            return result
        return super().next_tool_call(context, schemas, turns, max_input_tokens=max_input_tokens)
