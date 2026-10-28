"""Prepare a fresh neutral research workspace without models, training or downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
BASELINE = PACKAGE_ROOT / 'baseline'
DATA_MANIFEST = BASELINE / 'resources' / 'hash_manifest.json'
BASELINE_FILES = (
    'main.py', 'models/resnet.py', 'models/__init__.py', 'utils.py', 'LICENSE',
    'resources/split.json', 'resources/hash_manifest.json',
)
PROTOCOL_ID = 'l3-calibration-public-v1'
BUDGETS = {
    'max_tasks': 12,
    'max_attempts': 2,
    'max_llm_calls': 200,
    'timeout_seconds': 14400,
}


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    return {'sha256': digest.hexdigest(), 'bytes': size}


def verify_dataset(dataset_root: Path) -> dict:
    """Verify the official Python batches by bytes, without deserializing pickle."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    expected = json.loads(DATA_MANIFEST.read_text(encoding='utf-8'))
    actual = {}
    for name, identity in expected.items():
        path = dataset_root / 'cifar-10-batches-py' / name
        if not path.is_file():
            raise ValueError(f'Missing dataset file: {path}')
        actual[name] = file_identity(path)
        if actual[name] != identity:
            raise ValueError(f'Dataset size or SHA256 mismatch: {path}')
    return actual


def validate_catalog(dataset_root: Path) -> None:
    """Require existing deployment registration; never modify shared datasets."""
    dataset_root = Path(dataset_root).resolve()
    path = dataset_root.parent / 'catalog.json'
    try:
        catalog = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Prepare {path} with a cifar10 relative-path entry first') from error
    relative = catalog.get('cifar10') if isinstance(catalog, dict) else None
    if not isinstance(relative, str):
        raise ValueError(f'{path} must register cifar10')
    candidate = Path(relative)
    if candidate.is_absolute() or '..' in candidate.parts:
        raise ValueError('cifar10 catalog entry must stay within the dataset parent directory')
    if (dataset_root.parent / candidate).resolve() != dataset_root:
        raise ValueError('cifar10 catalog entry does not resolve to --dataset-root')


def _git(workspace: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ['git', '-C', str(workspace), *arguments], text=True, stderr=subprocess.STDOUT,
    ).strip()


def prepare_workspace(root: Path, dataset_root: Path) -> dict:
    """Create one new run's inputs. An existing root is always an error."""
    root = Path(root).expanduser()
    dataset_root = Path(dataset_root).expanduser()
    if not root.is_absolute() or not dataset_root.is_absolute():
        raise ValueError('--root and --dataset-root must be absolute paths')
    if root.exists() or root.is_symlink():
        raise ValueError(f'Refusing to reuse an existing run root: {root}')
    root = root.resolve()
    dataset_root = dataset_root.resolve()
    if root.exists() or root.is_symlink():
        raise ValueError(f'Refusing to reuse an existing run root: {root}')
    if root.is_relative_to(REPO_ROOT) or REPO_ROOT.is_relative_to(root):
        raise ValueError('The run root must be outside the sciagent source checkout')
    if root.is_relative_to(dataset_root) or dataset_root.is_relative_to(root):
        raise ValueError('The run root and dataset directory must not overlap')
    if not dataset_root.is_dir():
        raise ValueError('The dataset root must already exist')
    if shutil.which('git') is None:
        raise ValueError('Git is required to prepare the baseline workspace')
    dataset_files = verify_dataset(dataset_root)
    validate_catalog(dataset_root)

    root.mkdir(parents=True, mode=0o700)
    workspace = root / 'workspace'
    workspace.mkdir()
    for relative in BASELINE_FILES:
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BASELINE / relative, destination)
    for name in ('logs', 'traces', 'analysis'):
        (root / name).mkdir(mode=0o700)
    shutil.copyfile(PACKAGE_ROOT / 'goal.txt', root / 'goal.txt')
    (workspace / '.gitignore').write_text(
        '__pycache__/\n*.py[cod]\ncheckpoint/\noutputs/\n.venv/\n', encoding='utf-8',
    )
    (workspace / 'README.md').write_text(
        '# Neutral CIFAR-10 / ResNet18 input\n\n'
        'This workspace contains a baseline training program, not a calibration solution.\n'
        'Use `python main.py --epochs N --seed S --output-dir DIR`. '
        'Epochs and training choices are for the research task to determine; '
        'the program default is not a required research budget.\n\n'
        f'Dataset ID: cifar10. Torchvision data root: {dataset_root}\n'
        'Read the cifar10 entry in SCIAGENT_DATASETS_JSON; '
        'SCIAGENT_DATASET_ROOT is the parent deployment directory.\n\n'
        'Concatenate data_batch_1 through data_batch_5 in official order. '
        'Indices 0..44999 are train; 45000..49999 are calibration. '
        'The official test_batch contains the 10000 final-test examples. '
        'See resources/split.json and resources/hash_manifest.json.\n\n'
        'Training uses only train, does not read the official test set, '
        'and writes ckpt.pth. Use the same frozen checkpoint before/after calibration. '
        'Do not change the architecture/split, use external pretrained weights, '
        'or try multiple candidate methods. Freeze the method, checkpoint, '
        'fitted parameters and metric definitions before final test evaluation. '
        'Do not select methods or parameters from test results.\n\n'
        'Dependencies: compatible PyTorch, torchvision and NumPy; a CUDA GPU is '
        'recommended for the formal task. No packages or data are downloaded by '
        'this preparer. The operator must establish and record a viable runtime '
        'and a separate baseline smoke test before starting the formal run.\n',
        encoding='utf-8',
    )
    input_files = {
        str(path.relative_to(workspace)): file_identity(path)
        for path in sorted(workspace.rglob('*')) if path.is_file()
    }
    _git(workspace, '-c', 'init.templateDir=', 'init', '-q', '-b', 'main')
    _git(workspace, 'config', '--local', 'core.autocrlf', 'false')
    _git(workspace, 'add', '--all')
    _git(
        workspace, '-c', 'user.name=L3 fixture', '-c', 'user.email=l3@example.invalid',
        '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null',
        'commit', '-qm', 'Neutral public L3 baseline',
    )
    settings = {
        'SCIAGENT_MODEL': os.environ.get('SCIAGENT_MODEL', 'deepseek-v4-flash'),
        'SCIAGENT_DATASET_ROOT': str(dataset_root.parent),
        'SCIAGENT_DATASETS_JSON': json.dumps({'cifar10': str(dataset_root)}, ensure_ascii=False),
        'SCIAGENT_LLM_TRACE_LEVEL': 'full',
        'SCIAGENT_LLM_TRACE_DIR': str(root / 'traces'),
    }
    (root / 'run-config.sh').write_text(
        '# Non-secret deployment settings. Provide API keys through the environment.\n'
        + ''.join(f'{key}={shlex.quote(value)}\n' for key, value in settings.items()),
        encoding='utf-8',
    )
    (root / 'run-config.sh').chmod(0o600)
    # Preserve a venv's interpreter path; resolving its symlink can bypass the venv.
    python_executable = os.path.abspath(sys.executable)
    (root / 'python-path.txt').write_text(python_executable + '\n', encoding='utf-8')
    manifest = {
        'protocol_id': PROTOCOL_ID,
        'workspace': str(workspace),
        'workspace_commit': _git(workspace, 'rev-parse', 'HEAD'),
        'dataset_root': str(dataset_root),
        'dataset_files': dataset_files,
        'input_files': input_files,
        'goal_sha256': file_identity(root / 'goal.txt')['sha256'],
        'budgets': dict(BUDGETS),
        'prepared_with_python': python_executable,
        'preparation_performed_training_or_model_calls': False,
        'historical_run_inputs_claimed_identical': False,
    }
    (root / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
    )
    return manifest


def claim_start(root: Path, run_id: str) -> None:
    """Allow exactly one formal start from an unchanged neutral workspace."""
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    workspace = root / 'workspace'
    if manifest.get('protocol_id') != PROTOCOL_ID or manifest.get('budgets') != BUDGETS:
        raise ValueError('Unexpected protocol or budget manifest')
    if (root / 'data').exists() or (root / 'data').is_symlink():
        raise ValueError('Run data already exists; use answer/resume or prepare a new root')
    if (root / '.started').exists() or (root / '.started').is_symlink():
        raise ValueError('This root has already been started; use answer/resume')
    actual = {}
    for path in sorted(workspace.rglob('*')):
        if path.relative_to(workspace).parts[0] == '.git':
            continue
        if path.is_symlink():
            raise ValueError('Unexpected symlink in the initial workspace')
        if path.is_file():
            actual[str(path.relative_to(workspace))] = file_identity(path)
    if actual != manifest['input_files'] or file_identity(root / 'goal.txt')['sha256'] != manifest['goal_sha256']:
        raise ValueError('Initial workspace or goal changed; prepare a fresh neutral root')
    if _git(workspace, 'rev-parse', 'HEAD') != manifest['workspace_commit']:
        raise ValueError('Initial workspace Git revision changed')
    try:
        (root / '.started').mkdir(mode=0o700)
    except FileExistsError as error:
        raise ValueError('This root has already been started') from error
    (root / '.started' / 'run-id.txt').write_text(run_id + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='New absolute run directory outside this checkout')
    parser.add_argument('--dataset-root', type=Path, help='Existing torchvision CIFAR-10 directory')
    parser.add_argument('--claim-run', metavar='RUN_ID', help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    try:
        if arguments.claim_run is not None:
            if arguments.dataset_root is not None:
                parser.error('--claim-run cannot be combined with --dataset-root')
            claim_start(arguments.root, arguments.claim_run)
            return
        if arguments.dataset_root is None:
            parser.error('--dataset-root is required for preparation')
        manifest = prepare_workspace(arguments.root, arguments.dataset_root)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f'Preparation failed: {error}\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
