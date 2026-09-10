"""Parallel, manifest-driven chest X-ray classification pipeline.

The legacy TensorFlow binary entry points remain unchanged. This package uses
the locked Phase 2 manifests and never reorganizes the immutable source tree.
"""

from .config import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "load_config"]
