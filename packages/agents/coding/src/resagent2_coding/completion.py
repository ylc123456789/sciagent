"""Git-delivery completion policy used by the coding preset."""
from resagent2_executor.completion import ExecutionCompletionCheck


class CodingCompletionCheck(ExecutionCompletionCheck):
    def __init__(self, repository, boundary, *, baseline, output_dir=None):
        super().__init__(boundary, repository=repository, baseline=baseline, output_dir=output_dir)
