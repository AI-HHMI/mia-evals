from __future__ import annotations

from registry import Registry

from .base import BaseTask

TaskRegistry: Registry[BaseTask] = Registry(BaseTask)
