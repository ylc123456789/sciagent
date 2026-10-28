"""Attempt-owned environment locations, separate from project source files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class ExecutionEnvironment:
    """Locate environments and caches; do not create or activate an interpreter."""

    root: Path
    workspace_root: Path

    @classmethod
    def allocate(cls, *, workspace_root: Path, output_dir: str | None) -> ExecutionEnvironment:
        workspace = workspace_root.resolve()
        parent = Path(output_dir).expanduser().resolve() if output_dir else Path(tempfile.gettempdir()).resolve()
        if parent.is_relative_to(workspace):
            parent = Path(tempfile.gettempdir()).resolve()
        if parent.is_relative_to(workspace):
            raise ValueError('Execution environment requires an output or temporary directory outside the workspace')
        parent.mkdir(parents=True, exist_ok=True)
        value = cls(Path(tempfile.mkdtemp(prefix='sciagent-execution-', dir=parent)), workspace)
        value.validate()
        for directory in (value.cache_dir, value.cache_dir / 'pip', value.cache_dir / 'uv'):
            directory.mkdir()
        return value

    @classmethod
    def from_memory(cls, value: object, *, workspace_root: Path) -> ExecutionEnvironment:
        if (not isinstance(value, dict) or set(value) != {'version', 'root'}
                or type(value['version']) is not int or value['version'] != 1
                or not isinstance(value['root'], str) or not value['root']):
            raise ValueError('Resumed execution has no valid persisted environment directory')
        environment = cls(Path(value['root']), workspace_root.resolve())
        environment.validate()
        return environment

    @property
    def environment_dir(self) -> Path:
        return self.root / 'environment'

    @property
    def cache_dir(self) -> Path:
        return self.root / 'cache'

    def validate(self) -> None:
        # Pin the physical address across recovery and exact-operation approval.
        # Interpreter symlinks inside an ordinary venv are unaffected.
        if (not self.root.is_absolute() or self.root.resolve() != self.root
                or not self.root.is_dir()
                or self.root.is_relative_to(self.workspace_root)
                or self.workspace_root.is_relative_to(self.root)):
            raise ValueError('Execution environment directory is missing, changed, or overlaps the workspace')
        for directory in (self.environment_dir, self.cache_dir,
                          self.cache_dir / 'pip', self.cache_dir / 'uv'):
            if directory.resolve() != directory or (directory.exists() and not directory.is_dir()):
                raise ValueError(f'Execution environment directory changed: {directory}')

    def to_memory(self) -> dict:
        return {'version': 1, 'root': str(self.root)}

    def env_overrides(self) -> dict[str, str]:
        self.validate()
        return {
            'SCIAGENT_ENVIRONMENT_DIR': str(self.environment_dir),
            'UV_PROJECT_ENVIRONMENT': str(self.environment_dir),
            'XDG_CACHE_HOME': str(self.cache_dir),
            'PIP_CACHE_DIR': str(self.cache_dir / 'pip'),
            'UV_CACHE_DIR': str(self.cache_dir / 'uv'),
        }

    def context(self) -> dict:
        return {
            'source': 'runtime', 'root': str(self.root),
            'environment_dir': str(self.environment_dir), 'cache_dir': str(self.cache_dir),
            'environment_variables': self.env_overrides(),
            'scope': 'This Attempt only: create or reuse its environment and caches. '
                     'No authority over parent or sibling directories. Ordinary operation approval still applies.',
            'guidance': 'Inspect and reuse a suitable existing interpreter, or create a needed environment at '
                        'SCIAGENT_ENVIRONMENT_DIR (for example, python -m venv "$SCIAGENT_ENVIRONMENT_DIR"). '
                        'Select its interpreter explicitly; no environment is created or activated automatically. '
                        'Keep installed dependencies and caches outside the source workspace. '
                        'Retain dependency declarations and lock files in the project when needed for reproduction.',
        }
