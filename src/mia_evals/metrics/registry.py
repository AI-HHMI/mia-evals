from __future__ import annotations

from mia_evals.registry import Registry

from .base import BaseMetric

MetricRegistry: Registry[BaseMetric] = Registry(BaseMetric)
