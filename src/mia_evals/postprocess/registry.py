from __future__ import annotations

from mia_evals.registry import Registry

from .base import BasePostprocess

PostprocessRegistry: Registry[BasePostprocess] = Registry(BasePostprocess)
