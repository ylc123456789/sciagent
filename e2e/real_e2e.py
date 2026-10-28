"""Real closed-loop tests with native Coding/Experiment/Scientific agents.

Uses the production CLI application builder for every stage. Full-system
scenarios drive its ResearchController; direct stages invoke its bound Agent.
Execution agents inspect, select and manage environments through run_shell
(ADR-0030); pass version or environment requirements in task instructions.
SCIAGENT_DATASET_ROOT points at the deployment dataset catalog and files.

Stages: ``python -m e2e.real_e2e direct|code-experiment|repair|ask-start|ask-resume|literature``
(``full`` aliases ``code-experiment``; ``code`` and ``experiment`` invoke one agent).
"""

from __future__ import annotations

import os
import json
import math
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from sciagent_cli.composition import CliApplication, build_application
from sciagent_contracts import (
    AgentPermissions,
    RunPermissions,
    ExecutionLimits,
    AgentOwner,
    AttemptStatus,
    WorkflowAgentKind,
    AgentResult,
    ModuleStatus,
    AgentRequest,
    ResearchRequest,
    RunBudget,
    RunStatus,
    ScientificVerdict,
    TaskBudget,
    TaskStatus,
    UserAnswer,
    VerificationResult,
    latest_command_results,
    WorkspaceGrant,
    WorkspaceAccess,
    WorkspaceSourceKind,
    WorkspaceSpec,
)
from sciagent_components import (
    DatasetCatalog,
    ResourceLayout,
    RegisteredArtifactReader,
    read_artifact_json,
)

UTIL_PY = 'def add(a, b):\n    return a + b\n'

# A small, real CIFAR-10 training script. The candidate method is a
# Squeeze-and-Excitation (SE) block whose forward pass is deliberately left as
# an identity placeholder; the Coding Agent must implement it. Running without
# --use-se always yields a valid baseline, so the Experiment Agent can always
# produce metrics for both arms.
TRAIN_PY = """\
import argparse
import json
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def _data_root():
    # Datasets are exposed generically by the Experiment Agent. Pick the cifar10
    # entry from the id -> path map, falling back to the shared dataset root.
    mapping = os.environ.get("SCIAGENT_DATASETS_JSON")
    if mapping:
        try:
            datasets = json.loads(mapping)
            if "cifar10" in datasets:
                return datasets["cifar10"]
        except (ValueError, TypeError):
            pass
    return os.environ.get("SCIAGENT_DATASET_ROOT", "./data")


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.channel = channel
        self.reduction = reduction
        # TODO(candidate): implement squeeze-and-excitation here.
        # Squeeze: global average pool over spatial dims -> (b, c, 1, 1).
        # Excite: fc(channel -> channel/reduction) -> relu ->
        #         fc(channel/reduction -> channel) -> sigmoid -> (b, c, 1, 1).
        # Then rescale x channel-wise by the excitation.
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # TODO(candidate): implement the SE forward pass here.
        # Squeeze: b, c, h, w = x.size(); y = avg_pool2d(x, (h, w)).view(b, c)
        # Excite:  y = self.fc(y).view(b, c, 1, 1)
        # Return:  x * y
        raise NotImplementedError("SELayer.forward must be implemented")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, use_se=False):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.use_se = use_se
        if use_se:
            self.se = SELayer(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, 10)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.use_se:
            # SE operates on the 4D feature map (b, c, h, w) before pooling.
            out = self.se(out)
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(use_se=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], use_se=use_se)


def _loader(train):
    if train:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    return torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root=_data_root(), train=train, download=False, transform=transform),
        batch_size=128,
        shuffle=train,
        num_workers=2,
    )


def _accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return round(correct / total, 4)


def _train_arm(epochs, seed, use_se, device, loaders):
    torch.manual_seed(seed)
    model = resnet18(use_se=use_se).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    train_loader, test_loader = loaders
    for _ in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
    return _accuracy(model, test_loader, device)


def train(epochs, seed):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = _loader(train=True)
    test_loader = _loader(train=False)
    baseline = _train_arm(epochs, seed, use_se=False, device=device, loaders=(train_loader, test_loader))
    candidate = _train_arm(epochs, seed, use_se=True, device=device, loaders=(train_loader, test_loader))
    metrics = {
        "baseline_accuracy": baseline,
        "candidate_accuracy": candidate,
        "epochs": epochs,
        "seed": seed,
        "device": device,
    }
    json.dump(metrics, open("metrics.json", "w"))
    print(f"baseline={baseline} candidate={candidate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args().epochs, parser.parse_args().seed)
"""

# The candidate method must be reflected in the training script. Any fitting
# executor configuration may implement and measure, together or in separate tasks.
# Acceptance checks actual patches, successful training and frozen metrics.
_REQUIREMENTS_TXT = """\
torch>=2.0
torchvision>=0.15
"""

_EXPECTED_TASK_CAPABILITIES = {
    WorkflowAgentKind.SCIENTIFIC,
    WorkflowAgentKind.EXECUTOR,
}

def _repo(workdir: Path) -> Path:
    repo = workdir / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "train.py").write_text(TRAIN_PY, encoding="utf-8")
    (repo / "requirements.txt").write_text(_REQUIREMENTS_TXT, encoding="utf-8")
    if not (repo / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "e2e@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "e2e"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def _grant(repo: Path) -> WorkspaceGrant:
    return WorkspaceGrant(
        root=str(repo),
        access=WorkspaceAccess(read_paths=["."], write_paths=["."]),
        source=WorkspaceSourceKind.LOCAL,
    )


def _application(workdir: Path, repo: Path | None = None) -> CliApplication:
    """Prepare scenario workspaces and reuse the production application builder."""
    workspaces = {}
    if repo is not None:
        workspaces["ws_main"] = WorkspaceSpec(
            workspace_id="ws_main",
            source_kind=WorkspaceSourceKind.LOCAL,
            location=str(repo),
            access=WorkspaceAccess(read_paths=["."], write_paths=["."]),
        )
    return build_application(data_root=workdir / "data", workspaces=workspaces)


_BUGGY_TRAIN_PY = """\
import json


def main():
    correct = 8
    total = 10
    result = correct / totla  # BUG: typo — should be `total`
    json.dump({"accuracy": result}, open("metrics.json", "w"))


if __name__ == "__main__":
    main()
"""


def _repair_repo(workdir: Path) -> Path:
    repo = workdir / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "train.py").write_text(_BUGGY_TRAIN_PY, encoding="utf-8")
    (repo / "requirements.txt").write_text("", encoding="utf-8")
    if not (repo / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "e2e@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "e2e"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def _real_e2e_succeeded(run) -> bool:
    """Accept any valid task decomposition, but require frozen GPU training evidence."""
    if run.status != RunStatus.COMPLETED or run.final_opinion is None:
        return False
    if run.final_report_artifact_id not in run.artifacts:
        return False
    if run.workflow is None:
        return False
    tasks = run.workflow.tasks
    if {task.workflow_agent_kind for task in tasks} != _EXPECTED_TASK_CAPABILITIES:
        return False
    if any(
        task.status != TaskStatus.COMPLETED or not task.attempts
        for task in tasks
    ):
        return False
    if not _has_training_patch(run):
        return False
    return any(
        all(_accuracy(metrics.get(key)) for key in ("baseline_accuracy", "candidate_accuracy"))
        and type(metrics.get("epochs")) is int and metrics["epochs"] == 1
        and type(metrics.get("seed")) is int and metrics["seed"] == 0
        and metrics.get("device") == "cuda"
        for metrics in _training_metrics(run)
    )


def _accuracy(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def _attempt_artifacts(run, task, attempt):
    return [
        run.artifacts[key] for key in attempt.artifact_ids
        if key in run.artifacts
        and run.artifacts[key].run_id == run.run_id
        and run.artifacts[key].task_id == task.id
        and run.artifacts[key].attempt_number == attempt.number
        and run.artifacts[key].producer.value == task.workflow_agent_kind.value
    ]


def _has_training_patch(run) -> bool:
    reader = RegisteredArtifactReader(list(run.artifacts.values()), run_id=run.run_id)
    for task in run.workflow.tasks:
        if (task.workflow_agent_kind != WorkflowAgentKind.EXECUTOR
                or task.status != TaskStatus.COMPLETED or not task.attempts
                or task.attempts[-1].status != AttemptStatus.COMPLETED):
            continue
        for ref in _attempt_artifacts(run, task, task.attempts[-1]):
            if ref.kind != "code_patch":
                continue
            try:
                patch = reader.read_text(ref.id, max_chars=16_000_000)["content"]
            except (ValueError, OSError):
                continue
            if "+++ b/train.py" in patch and "\n@@ " in patch:
                return True
    return False


def _training_commands(run, refs):
    reader = RegisteredArtifactReader(refs, run_id=run.run_id)
    for ref in refs:
        if ref.kind != "execution_record":
            continue
        try:
            data = read_artifact_json(reader, ref.id)
            rows = [VerificationResult.model_validate(row) for row in data["results"]]
            for row in rows:
                words = shlex.split(row.command)
                if not words or not re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", Path(words[0]).name):
                    continue
                arguments = words[1:]
                while arguments and arguments[0] in {"-u", "-B", "-s", "-E", "-I", "-O", "-OO"}:
                    arguments = arguments[1:]
                if arguments and Path(arguments[0]).name == "train.py":
                    yield row
        except (ValueError, OSError, TypeError, KeyError):
            continue


def _training_metrics(run):
    """Require cited workspace metrics and a successful train command from the same attempt."""
    reader = RegisteredArtifactReader(list(run.artifacts.values()), run_id=run.run_id)
    for task in run.workflow.tasks:
        if (task.workflow_agent_kind != WorkflowAgentKind.EXECUTOR
                or task.status != TaskStatus.COMPLETED or not task.attempts
                or task.attempts[-1].status != AttemptStatus.COMPLETED):
            continue
        refs = _attempt_artifacts(run, task, task.attempts[-1])
        commands = list(_training_commands(run, refs))
        if not commands or any(row.exit_code != 0 or row.timed_out
                               for row in latest_command_results(commands)):
            continue
        for ref in refs:
            if (ref.media_type != "application/json"
                    or ref.metadata.get("source_path") != "metrics.json"
                    or ref.metadata.get("source_root") != "workspace"
                    or ref.id not in run.final_opinion.evidence_artifact_ids):
                continue
            try:
                metrics = read_artifact_json(reader, ref.id)
            except (ValueError, OSError):
                continue
            if isinstance(metrics, dict):
                yield metrics


def _dataset_context(layout: ResourceLayout):
    """Pass initial dataset facts through the common direct Agent request."""
    refs = DatasetCatalog(layout.dataset_root).references()
    return {"dataset_refs": [ref.model_dump(mode="json") for ref in refs]}


def run_code(workdir: Path) -> AgentResult:
    repo = _repo(workdir)
    app = _application(workdir, repo)
    agent = app.controller.scheduler.bindings[WorkflowAgentKind.EXECUTOR].port
    request = AgentRequest(work_request_id="work_direct",
        run_id="run_real",
        task_id="task_code",
        attempt_number=1,
        agent=AgentOwner.EXECUTOR,
        execution_definition=app.controller.registry.execution_definitions["coding@1"],
        task_context=_dataset_context(agent.resource_layout),
        instruction=(
            "Goal:\nImplement the Squeeze-and-Excitation forward pass in train.py\n\n"
            "Task:\nImplement SELayer.forward in train.py (it raises NotImplementedError)"
        ),
        budget=TaskBudget(max_llm_calls=40, timeout_seconds=900),
        workspace=_grant(repo),
        workspace_id="ws_main",
        output_dir=str(workdir / "out"),
        permissions=AgentPermissions(execute_commands=True),
    )
    return agent.invoke(request)


def run_experiment(workdir: Path) -> AgentResult:
    repo = _repo(workdir)
    app = _application(workdir, repo)
    agent = app.controller.scheduler.bindings[WorkflowAgentKind.EXECUTOR].port
    request = AgentRequest(work_request_id="work_direct",
        run_id="run_real",
        task_id="task_experiment",
        attempt_number=1,
        agent=AgentOwner.EXECUTOR,
        execution_definition=app.controller.registry.execution_definitions["experiment@1"],
        task_context=_dataset_context(agent.resource_layout),
        instruction=(
            "Goal:\nRun train.py and record baseline and candidate accuracy\n\n"
            "Task:\nRun train.py (it trains both the baseline and the SE candidate "
            "and writes metrics.json with baseline_accuracy and candidate_accuracy). "
            "Record those accuracies."
        ),
        budget=TaskBudget(max_llm_calls=60, timeout_seconds=1800),
        workspace=_grant(repo),
        workspace_id="ws_main",
        output_dir=str(workdir / "out"),
        permissions=AgentPermissions(execute_commands=True),
    )
    return agent.invoke(request)


def run_full(workdir: Path) -> bool:
    repo = _repo(workdir)
    request = ResearchRequest(
        goal=(
            "On CIFAR-10, compare the test accuracy of two ResNet18 variants. "
            "The baseline is the standard ResNet18 already implemented in train.py. "
            "The candidate adds a Squeeze-and-Excitation (SE) channel-attention "
            "block after the final conv layer (on the 4D feature map, before "
            "global pooling): global average pool over spatial dims, then "
            "fc(channel -> channel/16) -> relu -> fc(channel/16 -> channel) -> "
            "sigmoid, rescaling channels element-wise. The SELayer.forward in "
            "train.py currently raises NotImplementedError and must be "
            "implemented (the SE arm crashes until it is). train.py already "
            "runs both arms (baseline and SE) and "
            "writes metrics.json with baseline_accuracy, candidate_accuracy, and device; "
            "run both arms on CUDA and preserve the original metrics.json as evidence. "
            "The training protocol (SGD, epochs=1, seed=0) is fixed and must not "
            "be changed. Conclude whether the SE block improves accuracy over "
            "the baseline."
        ),
        budget=RunBudget(
            max_llm_calls=200, timeout_seconds=3600
        ),
        permissions=RunPermissions(execute_commands=True),
        execution_limits=ExecutionLimits(max_tasks=4, max_attempts_per_task=2),
    )
    controller = _application(workdir, repo).controller
    run = controller.create_run("run_full_real", request)
    tasks = run.workflow.tasks if run.workflow is not None else []
    for task in tasks:
        attempts = ", ".join(f"{a.number}:{a.status.value}" for a in task.attempts)
        print(f"{task.id} [{task.workflow_agent_kind.value}] {task.status.value} attempts={attempts}")
    print(f"run status={run.status.value} artifacts={len(run.artifacts)}")
    return _real_e2e_succeeded(run)


def _direct_succeeded(run) -> bool:
    """Scenario 1 acceptance: completed, inconclusive, root only, final report."""
    return (
        run.status == RunStatus.COMPLETED
        and run.final_opinion is not None
        and run.final_opinion.verdict == ScientificVerdict.INCONCLUSIVE
        and run.final_report_artifact_id is not None
        and run.workflow is not None
        and len(run.workflow.tasks) == 1
        and run.workflow.tasks[0].id == "task_root"
        and run.workflow.tasks[0].workflow_agent_kind == WorkflowAgentKind.SCIENTIFIC
        and len(run.work_requests) == 1
    )


def _repair_succeeded(run) -> bool:
    """Require a failed training attempt, a code patch and measured recovery."""
    tasks = run.workflow.tasks if run.workflow is not None else []
    had_failure = any(
        row.exit_code != 0 or row.timed_out
        for task in tasks if task.workflow_agent_kind == WorkflowAgentKind.EXECUTOR
        for attempt in task.attempts if attempt.status == AttemptStatus.FAILED
        for row in _training_commands(run, _attempt_artifacts(run, task, attempt))
    )
    return (
        run.status == RunStatus.COMPLETED
        and run.final_opinion is not None
        and run.final_report_artifact_id in run.artifacts
        and had_failure
        and _has_training_patch(run)
        and any(_accuracy(metrics.get("accuracy")) and metrics["accuracy"] == 0.8
                for metrics in _training_metrics(run))
    )


def _ask_start_succeeded(run) -> bool:
    """Scenario 4a acceptance: paused with a persisted pending question."""
    return run.status == RunStatus.PAUSED and run.pending_question is not None


def _ask_resume_succeeded(run) -> bool:
    """Scenario 4b acceptance: resumed and completed with a final opinion."""
    return run.status == RunStatus.COMPLETED and run.final_opinion is not None


def _literature_succeeded(run) -> bool:
    """Scenario 5 acceptance: cite an intact paper or its extracted full text."""
    cited = set(run.final_opinion.evidence_artifact_ids) if run.final_opinion else set()
    reader = RegisteredArtifactReader(list(run.artifacts.values()), run_id=run.run_id)
    literature_ids = []
    for ref in run.artifacts.values():
        if ref.kind not in {"literature_paper", "literature_fulltext"} or ref.id not in cited:
            continue
        try:
            reader.verify(ref.id)
        except (ValueError, OSError):
            continue
        literature_ids.append(ref.id)
    return (
        run.status == RunStatus.COMPLETED
        and run.final_opinion is not None
        and run.final_report_artifact_id in run.artifacts
        and bool(literature_ids)
        and any(artifact_id in cited for artifact_id in literature_ids)
    )


def run_direct(workdir: Path) -> bool:
    """Scenario 1: the Scientific Agent concludes inconclusive without any work."""
    controller = _application(workdir).controller
    request = ResearchRequest(
        goal=(
            "Determine whether the observed CIFAR-10 accuracy improvement is a "
            "causal effect of the SE block or merely correlation."
        ),
        constraints=[
            "Do not request experiments or any additional work; conclude from "
            "the available evidence only."
        ],
        budget=RunBudget(
            max_llm_calls=30, timeout_seconds=600
        ),
        permissions=RunPermissions(execute_commands=True),
        execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=1),
    )
    run = controller.create_run("run_direct", request)
    print(f"run status={run.status.value}")
    if run.final_opinion is not None:
        print(f"verdict={run.final_opinion.verdict.value}")
    return _direct_succeeded(run)


def run_repair(workdir: Path) -> bool:
    """Scenario 3: a failed experiment is diagnosed, fixed and rerun."""
    repo = _repair_repo(workdir)
    request = ResearchRequest(
        goal=(
            "Run the existing train.py. If it fails, diagnose the error, fix it, "
            "and rerun to obtain the accuracy."
        ),
        budget=RunBudget(
            max_llm_calls=200, timeout_seconds=3600
        ),
        permissions=RunPermissions(execute_commands=True),
        execution_limits=ExecutionLimits(max_tasks=4, max_attempts_per_task=3),
    )
    controller = _application(workdir, repo).controller
    run = controller.create_run("run_repair", request)
    tasks = run.workflow.tasks if run.workflow is not None else []
    for task in tasks:
        attempts = ", ".join(f"{a.number}:{a.status.value}" for a in task.attempts)
        print(f"{task.id} [{task.workflow_agent_kind.value}] {task.status.value} attempts={attempts}")
    print(f"run status={run.status.value} work_requests={len(run.work_requests)}")
    return _repair_succeeded(run)


def run_ask_start(workdir: Path) -> bool:
    """Scenario 4a: pause on a Scientific question and persist the session."""
    request = ResearchRequest(
        goal=(
            "Ask the user which primary evaluation metric should be recorded "
            "for a future comparison. This run is only for obtaining and "
            "recording the user's preference. After receiving the answer, "
            "finish and state the selected metric."
        ),
        constraints=[
            "This run only obtains and records the user's preferred evaluation "
            "metric.",
            "Ask the user which metric to record before doing anything else.",
            "Do not request or execute code or experiment work: after the user "
            "answers, finish and state the selected metric.",
        ],
        budget=RunBudget(
            max_llm_calls=20, timeout_seconds=600
        ),
        permissions=RunPermissions(execute_commands=True),
        execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=1),
    )
    controller = _application(workdir).controller
    run = controller.create_run("run_ask", request)
    print(f"run status={run.status.value}")
    if run.pending_question is not None:
        print(f"pending_question={run.pending_question.text}")
        print(f"requested_fields={run.pending_question.requested_fields}")
    return _ask_start_succeeded(run)


def run_ask_resume(workdir: Path, answer_text: str) -> bool:
    """Scenario 4b: resume the paused run in a fresh process and complete it."""
    app = _application(workdir)
    controller = app.controller
    run = app.run_store.load("run_ask")
    question = run.pending_question
    if question is None:
        print("no pending question to answer")
        return False
    answer = UserAnswer(
        question_id=question.id,
        values={field: answer_text for field in question.requested_fields},
        answered_at=datetime.now(UTC),
    )
    run = controller.answer_question("run_ask", answer)
    print(f"run status={run.status.value}")
    return _ask_resume_succeeded(run)


def run_literature(workdir: Path) -> bool:
    """Scenario 5: gather literature evidence and cite it in the opinion."""
    controller = _application(workdir).controller
    request = ResearchRequest(
        goal=(
            "According to the literature, does the Squeeze-and-Excitation (SE) "
            "block in SENet improve image classification accuracy on ImageNet? "
            "Retrieve relevant papers with literature_search, read them, and "
            "cite them in your final opinion."
        ),
        constraints=[
            "This is a literature-only review: use literature_search directly and "
            "do not request code or experiment work."
        ],
        required_evidence_kinds=["literature_paper"],
        budget=RunBudget(
            max_llm_calls=60, timeout_seconds=900
        ),
        permissions=RunPermissions(execute_commands=True),
        execution_limits=ExecutionLimits(max_tasks=1, max_attempts_per_task=1),
    )
    run = controller.create_run("run_literature", request)
    print(f"run status={run.status.value} artifacts={len(run.artifacts)}")
    return _literature_succeeded(run)


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "full"
    workdir = Path(os.environ.get("REAL_E2E_WORKDIR", tempfile.mkdtemp(prefix="sciagent-real-")))
    workdir.mkdir(parents=True, exist_ok=True)
    if stage == "code":
        result = run_code(workdir)
        print(f"status={result.status.value}")
        print(f"report={result.report}")
        print(f"artifacts={[item.kind for item in result.artifacts]}")
        sys.exit(0 if result.status == ModuleStatus.COMPLETED else 1)
    elif stage == "experiment":
        result = run_experiment(workdir)
        print(f"status={result.status.value}")
        print(f"report={result.report}")
        print(f"artifacts={[item.kind for item in result.artifacts]}")
        sys.exit(0 if result.status == ModuleStatus.COMPLETED else 1)
    elif stage in {"full", "code-experiment"}:
        sys.exit(0 if run_full(workdir) else 1)
    elif stage == "direct":
        sys.exit(0 if run_direct(workdir) else 1)
    elif stage == "repair":
        sys.exit(0 if run_repair(workdir) else 1)
    elif stage == "ask-start":
        sys.exit(0 if run_ask_start(workdir) else 1)
    elif stage == "ask-resume":
        answer_text = sys.argv[2] if len(sys.argv) > 2 else ""
        sys.exit(0 if run_ask_resume(workdir, answer_text) else 1)
    elif stage == "literature":
        sys.exit(0 if run_literature(workdir) else 1)
    else:
        raise SystemExit(f"unknown stage: {stage}")


if __name__ == "__main__":
    main()
