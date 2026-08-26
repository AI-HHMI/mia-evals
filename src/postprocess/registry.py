from __future__ import annotations

from registry import Registry

from .base import BasePostprocess

PostprocessRegistry: Registry[BasePostprocess] = Registry(BasePostprocess)
