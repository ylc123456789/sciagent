"""Integration layer that wires the ResAgent2 packages into runnable flows.

This package is intentionally NOT one of the three pip packages: it sits above
``contracts``/``runtime``/``orchestrator`` and may import all three, which the
individual packages may not do (their import roots are boundary-tested).
"""
