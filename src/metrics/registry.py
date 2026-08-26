from __future__ import annotations

from registry import Registry

from .base import BaseMetric

MetricRegistry: Registry[BaseMetric] = Registry(BaseMetric)
