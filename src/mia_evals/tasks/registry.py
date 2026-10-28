from __future__ import annotations

from mia_evals.registry import Registry

from .base import BaseTask

TaskRegistry: Registry[BaseTask] = Registry(BaseTask)
