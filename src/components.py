"""Imports every concrete implementation so the registries are populated.

A registry decorator only runs when its module is imported, so a task config naming
"cc_threshold" resolves only once this module has been. `evaluate.py` imports it before building
anything. Adding a component means adding one line here -- the engine and the registries never
change.

Mirrors `mia-train/src/components.py` deliberately: the two repositories are read by the same
people, and a convention that holds across both is one less thing to learn.
"""

from __future__ import annotations

import metrics.semantic  # noqa: F401  (imported for its registration side effect)
import metrics.skeleton  # noqa: F401  (imported for its registration side effect)
import metrics.voxel_instance  # noqa: F401  (imported for its registration side effect)
import postprocess.cc_threshold  # noqa: F401  (imported for its registration side effect)
import postprocess.labellings  # noqa: F401  (imported for its registration side effect)
import postprocess.mws  # noqa: F401  (imported for its registration side effect)
import tasks.segmentation  # noqa: F401  (imported for its registration side effect)
