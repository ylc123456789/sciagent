"""Reusable operations and presentation; no Tool entry points or Agent loop."""

from .materials import (
    read_artifact_json, request_dataset_refs, request_task_context, runtime_authority_context,
)
from .material_tree import material_index_context
from .artifacts import (
    ArtifactRegistrationPort,
    ArtifactReadError,
    RegisteredArtifactReader,
    media_type_for,
)
from .dataset import (
    DatasetAvailability,
    DatasetCatalog,
    DatasetResolutionError,
    dataset_context,
    dataset_env_overrides,
    resolve_dataset_refs,
)
from .git import (
    GitBaseline,
    GitWorkspace,
    GitWorkspaceError,
)
from .execution_records import execution_record, preserve_execution_record
from .process import ProcessRunner
from .repo import (
    MaterializedRepo,
    RepoMaterializer,
    RepoMaterializerError,
)
from .resources import ResourceLayout
from .workspace import (
    ReadOnlyWorkspaceBoundary,
    WorkspaceBoundary,
    WorkspacePermissionError,
)

from .literature import (
    ArxivLiteratureBackend,
    OpenAlexLiteratureBackend,
    MultiSourceLiteratureBackend,
    LiteraturePaper,
    LiteratureSearchBackend,
    LiteratureSearchResult,
    render_paper,
    PreparedLiteratureImport,
    load_literature_manifest,
    LiteratureSearchError,
    LiteratureUnavailableError,
)

from .web_deepseek import DeepSeekWebSearchBackend

from .web import (
    WebFetchError,
    WebPage,
    WebPageFetcher,
    WebSearchBackend,
    WebSearchError,
    WebSearchItem,
    WebSearchResult,
)

__all__ = [
    'execution_record',
    'preserve_execution_record',
    'read_artifact_json',
    'request_dataset_refs',
    'request_task_context',
    'runtime_authority_context',
    'material_index_context',
    'ArxivLiteratureBackend',
    'OpenAlexLiteratureBackend',
    'MultiSourceLiteratureBackend',
    'LiteraturePaper',
    'LiteratureSearchBackend',
    'LiteratureSearchResult',
    'render_paper',
    'PreparedLiteratureImport',
    'load_literature_manifest',
    'LiteratureSearchError',
    'LiteratureUnavailableError',
    'DeepSeekWebSearchBackend',
    'WebFetchError',
    'WebPage',
    'WebPageFetcher',
    'WebSearchBackend',
    'WebSearchError',
    'WebSearchItem',
    'WebSearchResult',
    'ArtifactRegistrationPort',
    'ArtifactReadError',
    'DatasetAvailability',
    'DatasetCatalog',
    'DatasetResolutionError',
    'GitBaseline',
    'GitWorkspace',
    'GitWorkspaceError',
    'MaterializedRepo',
    'ProcessRunner',
    'RegisteredArtifactReader',
    'RepoMaterializer',
    'RepoMaterializerError',
    'ResourceLayout',
    'ReadOnlyWorkspaceBoundary',
    'WorkspaceBoundary',
    'WorkspacePermissionError',
    'dataset_context',
    'dataset_env_overrides',
    'media_type_for',
    'resolve_dataset_refs',
]
