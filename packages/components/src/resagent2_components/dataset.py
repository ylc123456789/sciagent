"""Shared dataset catalog, context and execution bindings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from resagent2_contracts import DatasetRef


# Generic dataset hand-off surface. These are the only dataset-related env vars
# the Experiment Agent promises to its scripts: nothing framework-specific and
# no "first dataset" special-casing. Model/Hub caches (TORCH_HOME/HF_HOME/...)
# are deliberately kept out of the dataset root.
RESAGENT2_DATASET_ROOT = "RESAGENT2_DATASET_ROOT"
RESAGENT2_DATASETS_JSON = "RESAGENT2_DATASETS_JSON"
DATASET_CATALOG_FILENAME = "catalog.json"


class DatasetResolutionError(ValueError):
    """Raised when a task-level dataset reference cannot be safely resolved."""


@dataclass
class DatasetAvailability:
    """One checked view shared by Agent context and script environment.

    Directory existence is not validation of dataset contents. Missing entries
    are recoverable; malformed references and unsafe paths remain errors.
    """

    available: list[dict] = field(default_factory=list)
    unavailable_ids: list[str] = field(default_factory=list)


class DatasetCatalog:
    """Read the deployment-owned ``dataset_id -> relative path`` catalog.

    The catalog lives under the shared dataset root and is the only place where
    physical dataset directories are registered.  An absent catalog means that
    no datasets are registered. Malformed/unsafe entries are configuration
    errors; a missing directory is registered but not yet available.
    """

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()

    @property
    def path(self) -> Path:
        return self.dataset_root / DATASET_CATALOG_FILENAME

    def references(self) -> list[DatasetRef]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetResolutionError(
                f"cannot read dataset catalog {self.path}: {error}"
            ) from error
        if not isinstance(raw, dict) or not all(
            isinstance(dataset_id, str) and isinstance(relative_path, str)
            for dataset_id, relative_path in raw.items()
        ):
            raise DatasetResolutionError(
                "dataset catalog must be a JSON object mapping ids to relative paths"
            )
        try:
            refs = [
                DatasetRef(dataset_id=dataset_id, relative_path=relative_path)
                for dataset_id, relative_path in sorted(raw.items())
            ]
        except ValueError as error:
            raise DatasetResolutionError(f"invalid dataset catalog: {error}") from error
        resolve_dataset_refs(self.dataset_root, refs)
        return refs


def dataset_context(availability: DatasetAvailability) -> dict:
    """Render one compact policy/context payload shared by all Agents."""

    return {
        "available_dataset_ids": sorted(
            entry["dataset_id"] for entry in availability.available
        ),
        "unavailable_dataset_ids": sorted(availability.unavailable_ids),
        "availability_basis": (
            "available_dataset_ids: registered directories that exist; contents not validated. "
            "unavailable_dataset_ids: registered IDs whose directories do not exist. "
            "An ID in neither list is unregistered in this view, not confirmed available."
        ),
        "access": "read_only",
        "environment": {
            "root": RESAGENT2_DATASET_ROOT,
            "id_to_path_map": RESAGENT2_DATASETS_JSON,
        },
        "missing_dataset_action": "ask_user",
        "missing_dataset_guidance": (
            "If a dataset needed by the current task is not in available_dataset_ids, "
            "call ask_user before work that needs it, including when it is absent "
            "from both lists. Do not block on unrelated missing datasets. Ask the user to "
            "place its data under the dataset root, register its id and relative "
            "path in catalog.json under that root, then answer. "
            f"{RESAGENT2_DATASETS_JSON} contains the ID-to-path JSON for scripts, "
            "not a catalog file path. This checked view, not a user's "
            "confirmation alone, determines availability after resume. If the "
            "required dataset is still not in available_dataset_ids after a reply, "
            "ask_user again; having asked once is not permission to run without it. "
            "Earlier command results describe the earlier resource state, not "
            "the refreshed view. Missing "
            "files or invalid contents also require user help; do not download, "
            "invent a path, or substitute data."
        ),
        "download_allowed": False,
        "substitution_allowed": False,
    }


def resolve_dataset_refs(
    dataset_root: str | Path, refs: list[DatasetRef]
) -> DatasetAvailability:
    """Check registered references without blocking on unrelated missing data.

    Each reference's ``relative_path`` is joined under ``dataset_root``, then
    checked for directory escape (``..`` / absolute) and existence. Available
    entries contain ``{dataset_id, path, access}``; missing IDs are kept apart.
    The root is the shared directory, never one specific dataset. A duplicate
    ``dataset_id`` is rejected so one id can never resolve to two paths.
    """
    root = Path(dataset_root).expanduser().resolve()
    availability = DatasetAvailability()
    seen: set[str] = set()
    for ref in refs:
        if ref.dataset_id in seen:
            raise DatasetResolutionError(f"duplicate dataset_id: {ref.dataset_id!r}")
        seen.add(ref.dataset_id)
        candidate = (root / ref.relative_path).resolve()
        if not candidate.is_relative_to(root):
            raise DatasetResolutionError(
                f"dataset relative_path escapes the root: {ref.relative_path!r}"
            )
        if not candidate.is_dir():
            availability.unavailable_ids.append(ref.dataset_id)
            continue
        availability.available.append(
            {
                "dataset_id": ref.dataset_id,
                "path": str(candidate),
                "access": "read_only",
            }
        )
    return availability


def dataset_env_overrides(
    dataset_root: str | Path, availability: DatasetAvailability
) -> dict[str, str]:
    """Expose resolved datasets to scripts as a generic ``id -> path`` map.

    No framework is named and no single dataset is preferred: the Experiment
    Agent passes this mapping through so a script can look up the dataset it
    actually needs by id. Model/Hub cache variables are deliberately not set.
    """
    return {
        RESAGENT2_DATASET_ROOT: str(Path(dataset_root).expanduser().resolve()),
        RESAGENT2_DATASETS_JSON: json.dumps(
            {entry["dataset_id"]: entry["path"] for entry in availability.available},
            ensure_ascii=False,
        ),
    }
