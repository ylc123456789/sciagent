"""Owner-injected model-boundary yielding; no second scheduler or memory."""
from contextlib import contextmanager
from contextvars import ContextVar

_boundary = ContextVar("resagent2_model_boundary", default=None)

def boundary_requested():
    callback = _boundary.get()
    return bool(callback and callback())

@contextmanager
def model_boundary(callback):
    marker = _boundary.set(callback)
    try:
        yield
    finally:
        _boundary.reset(marker)
