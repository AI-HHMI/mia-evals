"""The CellMap organelle semantic tasks: their splits, data configs and scoring configs.

    python -m truth.cellmap --configs configs [--seed 0] [--dry-run]

Writes, for `cellmap_organelle_semantic` and `cellmap_organelle_semantic_fewshot`,

    configs/<task>/data/{test,fit}.yaml       miao data configs, one volume per CellMap crop
    configs/<task>/{identity,argmax}.toml     the scoring configs, with the 48-class table

and prints the split table docs/cellmap.md carries. Nothing is copied: every volume is a crop of an
lmd-v0.0.1 store, `labels/manual_gt-all_organelles-crop<N>`, which holds CellMap's combined `all`
array (one leaf-class id per voxel, decoded by the store's `classes.csv`).

**The rules** (decided with the user 2026-10-07/08; evidence in docs/cellmap.md):

- A crop is *complete* when it annotates all 48 classes the CellMap Segmentation Challenge scores
  (`TESTED`). Only complete crops are reported on: elsewhere a 0 means "not this class" for the
  few classes painted and nothing for the rest. Which classes a crop annotates is read from its
  per-class arrays in CellMap's own tree (`CELLMAP`); lmd keeps only `all`.
- Crops whose boxes overlap form one *group*, and a group is never split between fit and test:
  several crops are re-annotations of one region, or small crops inside a large one.
- `REANNOTATED` regions -- painted two to four times by different annotators, 7-16% of voxels
  disagreeing -- are set aside from both tasks; which annotation is "the truth" there is not ours
  to pick. `NEAR_COPIES` are regions painted twice where the second pass changed 0.4% of voxels:
  one crop is kept, the other dropped. Both lists are checked against the boxes, so a change in
  the data stops the build instead of passing silently.
- `cellmap_organelle_semantic`: per dataset, `TEST_FRACTION` of the complete crops (rounded, in
  whole groups) are test; every class must be present in at least `MIN_CROPS_PER_CLASS` test and as
  many fit crops. Of `TRIALS` seeded draws satisfying that, the one whose per-class test share is
  closest to `TEST_FRACTION` wins. Fit is every other crop, partial ones included; a partial crop
  in a test group is dropped.
- `cellmap_organelle_semantic_fewshot`: fit is one complete crop per dataset, taken from the large
  task's fit (so the two nest: fewshot fit is inside the large fit, the large test inside the
  fewshot test), chosen greedily for class coverage, preferring crops whose group holds no other
  complete crop. Test is every other eligible complete crop outside the fit crops' groups.
- Each volume's `bounding_box` is its crop in the raw's level-0 voxels (checked to be whole
  voxels), and its `resolutions` the raw's own voxel size, so a producer reads level 0 unless it
  asks for another resolution.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zarr

LMD = Path("/groups/miaai/miaai/lmd-v0.0.1/data")
CELLMAP = Path("/nrs/cellmap/data")
LABEL_GROUP = "manual_gt-all_organelles-crop"
TASKS = ("cellmap_organelle_semantic", "cellmap_organelle_semantic_fewshot")

#: The classes the CellMap Segmentation Challenge scores, in its order: its
#: `src/cellmap_segmentation_challenge/utils/tested_classes.csv`, unchanged since 2025-01-10 (its
#: site says 47). 31 are leaves of `classes.csv`, 17 composites (unions of leaves).
TESTED = (
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo", "golgi_mem", "golgi_lum", "ves_mem",
    "ves_lum", "endo_mem", "endo_lum", "lyso_mem", "lyso_lum", "ld_mem", "ld_lum", "er_mem",
    "er_lum", "eres_mem", "eres_lum", "ne_mem", "ne_lum", "np_out", "np_in", "hchrom", "echrom",
    "nucpl", "mt_out", "cyto", "mt_in", "nuc", "golgi", "ves", "endo", "lyso", "ld", "eres",
    "perox_mem", "perox_lum", "perox", "mito", "er", "ne", "np", "chrom", "mt", "cell",
    "er_mem_all", "ne_mem_all",
)

#: Regions painted more than once by different annotators, measured 2026-10-08 voxel by voxel:
#: 6.9-15.5% of voxels differ, annotator-vs-annotator mean IoU 0.41-0.66, no registration offset
#: (no shift within 2 voxels helps). Set aside from both tasks.
REANNOTATED = (
    ("jrc_fly-vnc-1", (173, 185)),
    ("jrc_hela-3", (101, 181)),
    ("jrc_jurkat-1", (126, 180, 182)),
    ("jrc_mus-liver", (157, 183)),
    ("jrc_mus-liver", (171, 416)),
    ("jrc_mus-liver", (172, 417)),
)
#: (dataset, kept, dropped): one region painted twice, the second pass changing 0.4% of voxels.
NEAR_COPIES = (("jrc_cos7-1b", 240, 291),)

TEST_FRACTION = 0.2
MIN_CROPS_PER_CLASS = 3
TRIALS = 2000
PATCH = (128, 128, 128)


@dataclass
class Crop:
    """One CellMap ground-truth crop as lmd holds it."""

    dataset: str
    number: int
    store: Path
    label_voxel: tuple[float, ...]
    label_first: tuple[float, ...]
    label_shape: tuple[int, ...]
    raw_voxel: tuple[float, ...]
    raw_first: tuple[float, ...]
    annotated: frozenset[str] = frozenset()
    #: label id -> voxel count; filled for complete crops only.
    counts: dict[int, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.dataset}_crop{self.number}"

    @property
    def label_key(self) -> str:
        return f"labels/{LABEL_GROUP}{self.number}"

    @property
    def complete(self) -> bool:
        return set(TESTED) <= self.annotated

    def extent(self) -> list[tuple[float, float]]:
        """Physical [lo, hi) per axis, from voxel corners."""
        return [
            (t - v / 2, t + (n - 0.5) * v)
            for t, v, n in zip(self.label_first, self.label_voxel, self.label_shape, strict=True)
        ]

    def box(self) -> list[list[int]]:
        """The crop in the raw's level-0 voxels; refused unless it falls on whole raw voxels."""
        out = []
        for (low, high), t, v in zip(self.extent(), self.raw_first, self.raw_voxel, strict=True):
            lo, hi = (low - (t - v / 2)) / v, (high - (t - v / 2)) / v
            if abs(lo - round(lo)) > 1e-6 or abs(hi - round(hi)) > 1e-6:
                raise ValueError(f"{self.name}: crop [{low}, {high}) is not on whole raw voxels "
                                 f"({lo}, {hi})")
            out.append([int(round(lo)), int(round(hi))])
        return out


def read_classes(path: Path) -> dict[str, tuple[int, tuple[int, ...]]]:
    """`classes.csv`: name -> (id, member leaf ids; empty for a leaf)."""
    with open(path, newline="") as handle:
        return {
            name: (int(cid), tuple(int(m) for m in members.split(",")) if members else ())
            for name, cid, members in csv.reader(handle)
        }


def class_table(classes: dict[str, tuple[int, tuple[int, ...]]]) -> dict[str, list[int]]:
    """Each tested class as every id a voxel of it may carry: its own, its leaves', and the ids of
    composites made only of its leaves (`nuc` includes `chrom`'s 54: a crop painting chromatin as a
    whole is still painting nucleus)."""
    table = {}
    for name in TESTED:
        cid, members = classes[name]
        if not members:
            table[name] = [cid]
            continue
        ids = {cid, *members}
        ids |= {other for other, sub in classes.values() if sub and set(sub) <= set(members)}
        table[name] = sorted(ids)
    return table


def _ome_level0(
    store: Path, key: str
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]]:
    from miao.zarr_meta import read_ome_metadata

    meta = read_ome_metadata(store, key, "zarr3", [0])
    names = [n for n in meta.axis_names]
    if names != ["z", "y", "x"]:
        raise ValueError(f"{store}/{key}: axes {names}, expected z, y, x")
    level = meta.scales[0]
    return (tuple(float(v) for v in level.scale_factors),
            tuple(float(v) for v in level.translation_or_zeros()),
            tuple(int(n) for n in level.shape))


def discover(
    lmd: Path, cellmap: Path
) -> tuple[list[Crop], dict[str, tuple[int, tuple[int, ...]]], str]:
    """Every CellMap crop lmd holds, its geometry and annotated classes; the shared classes.csv."""
    crops: list[Crop] = []
    tables: dict[str, str] = {}
    for store in sorted(lmd.glob("*CellMap*/*.zarr")):
        csv_path = store / "classes.csv"
        tables[str(csv_path)] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        raw_voxel, raw_first, _ = _ome_level0(store, "raw")
        for group in sorted((store / "labels").glob(f"{LABEL_GROUP}*")):
            number = int(group.name[len(LABEL_GROUP):])
            dataset = str(zarr.open_group(str(group), mode="r").attrs["dataset"])
            voxel, first, shape = _ome_level0(store, f"labels/{group.name}")
            pattern = f"{dataset}/{dataset}.zarr/recon-*/labels/groundtruth/crop{number}"
            source = sorted(cellmap.glob(pattern))
            if len(source) != 1:
                raise ValueError(f"{dataset} crop{number}: {len(source)} CellMap crop directories")
            annotated = frozenset(
                e for e in os.listdir(source[0])
                if not e.startswith(".") and e not in ("all", "zarr.json")
                and (source[0] / e).is_dir()
            )
            crops.append(Crop(dataset, number, store, voxel, first, shape, raw_voxel, raw_first,
                              annotated))
    digests = set(tables.values())
    if len(digests) != 1:
        raise ValueError(f"the stores' classes.csv differ: {tables}")
    classes = read_classes(Path(next(iter(tables))))
    return crops, classes, digests.pop()


def count_labels(crop: Crop) -> dict[int, int]:
    array = zarr.open_array(str(crop.store / crop.label_key / "s0"), mode="r")
    counts = np.zeros(256, dtype=np.int64)
    for z in range(0, array.shape[0], 64):
        counts += np.bincount(np.asarray(array[z:z + 64]).ravel(), minlength=256)
    return {int(i): int(n) for i, n in enumerate(counts) if n}


def overlap_groups(crops: list[Crop]) -> dict[str, int]:
    """Crop name -> group id: connected components of positively intersecting boxes, per store."""
    parent = {c.name: c.name for c in crops}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_store = defaultdict(list)
    for c in crops:
        by_store[c.store].append(c)
    for members in by_store.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if all(min(ah, bh) > max(al, bl) for (al, ah), (bl, bh)
                       in zip(a.extent(), b.extent(), strict=True)):
                    parent[find(a.name)] = find(b.name)
    roots = sorted({find(c.name) for c in crops})
    index = {root: i for i, root in enumerate(roots)}
    return {c.name: index[find(c.name)] for c in crops}


def check_reannotations(crops: list[Crop]) -> None:
    """The regions with identical boxes must be exactly those `REANNOTATED`/`NEAR_COPIES` name."""
    boxes = defaultdict(set)
    for c in crops:
        if c.complete:
            key = (c.dataset, tuple(round(v, 3) for pair in c.extent() for v in pair))
            boxes[key].add(c.number)
    found = {(d, tuple(sorted(n))) for (d, _), n in boxes.items() if len(n) > 1}
    expected = {(d, tuple(sorted(n))) for d, n in REANNOTATED}
    expected |= {(d, tuple(sorted((kept, dropped)))) for d, kept, dropped in NEAR_COPIES}
    if found != expected:
        raise ValueError(
            "regions painted more than once changed; review REANNOTATED / NEAR_COPIES:\n"
            f"  found only: {sorted(found - expected)}\n  listed only: {sorted(expected - found)}"
        )


@dataclass
class Split:
    large_test: list[Crop]
    large_fit: list[Crop]
    fewshot_fit: list[Crop]
    fewshot_test: list[Crop]
    set_aside: list[Crop]
    dropped: list[Crop]
    covered: set[str]
    seed: int


def split(crops: list[Crop], table: dict[str, list[int]], seed: int) -> Split:
    aside = {(d, n) for d, ns in REANNOTATED for n in ns}
    drop = {(d, dropped) for d, _, dropped in NEAR_COPIES}
    set_aside = [c for c in crops if (c.dataset, c.number) in aside]
    dropped = [c for c in crops if (c.dataset, c.number) in drop]
    kept = [c for c in crops if (c.dataset, c.number) not in aside | drop]
    group = overlap_groups(crops)
    members: dict[int, list[Crop]] = defaultdict(list)
    for c in kept:
        members[group[c.name]].append(c)

    def present(c: Crop) -> set[str]:
        return {name for name, ids in table.items() if any(c.counts.get(i) for i in ids)}

    classes_of = {c.name: present(c) for c in kept if c.complete}
    eligible = {g: [c for c in cs if c.complete] for g, cs in members.items()}
    eligible = {g: cs for g, cs in eligible.items() if cs}
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for g, cs in sorted(eligible.items()):
        by_dataset[cs[0].dataset].append(g)

    def draw(rng: np.random.Generator) -> set[int]:
        chosen: set[int] = set()
        for dataset in sorted(by_dataset):
            groups = by_dataset[dataset]
            target = math.floor(TEST_FRACTION * sum(len(eligible[g]) for g in groups) + 0.5)
            count = 0
            for k in rng.permutation(len(groups)):
                size = len(eligible[groups[k]])
                if count + size <= target:
                    chosen.add(groups[k])
                    count += size
                if count == target:
                    break
        return chosen

    def score(test: set[int]) -> float | None:
        cost = 0.0
        for name in TESTED:
            inside = sum(name in classes_of[c.name] for g in test for c in eligible[g])
            total = sum(name in classes_of[c.name] for cs in eligible.values() for c in cs)
            if inside < MIN_CROPS_PER_CLASS or total - inside < MIN_CROPS_PER_CLASS:
                return None
            cost += (inside / total - TEST_FRACTION) ** 2
        return cost

    rng = np.random.default_rng(seed)
    best: tuple[float, set[int]] | None = None
    for _ in range(TRIALS):
        test = draw(rng)
        cost = score(test)
        if cost is not None and (best is None or cost < best[0]):
            best = (cost, test)
    if best is None:
        raise ValueError(f"no draw of {TRIALS} met {MIN_CROPS_PER_CLASS} crops per class")
    test_groups = best[1]
    large_test = sorted((c for g in test_groups for c in eligible[g]), key=_order)
    large_fit = sorted((c for g, cs in members.items() if g not in test_groups for c in cs),
                       key=_order)

    # Fewshot fit: one complete crop per dataset from the large fit, greedy on new classes covered.
    candidates: dict[str, list[Crop]] = defaultdict(list)
    for c in large_fit:
        if c.complete:
            candidates[c.dataset].append(c)
    covered: set[str] = set()
    fewshot_fit: list[Crop] = []
    while len(fewshot_fit) < len(candidates):
        taken = {c.dataset for c in fewshot_fit}
        pick = max(
            (c for d, cs in candidates.items() if d not in taken for c in cs),
            key=lambda c: (len(classes_of[c.name] - covered), len(eligible[group[c.name]]) == 1,
                           len(classes_of[c.name]), c.dataset, -c.number),
        )
        fewshot_fit.append(pick)
        covered |= classes_of[pick.name]
    fit_groups = {group[c.name] for c in fewshot_fit}
    fewshot_test = sorted((c for g, cs in eligible.items() if g not in fit_groups for c in cs),
                          key=_order)
    fewshot_fit.sort(key=_order)

    names = lambda cs: {c.name for c in cs}  # noqa: E731
    assert names(large_test) <= names(fewshot_test) and names(fewshot_fit) <= names(large_fit)
    assert not names(large_test) & names(large_fit) and not names(fewshot_test) & names(fewshot_fit)
    for reported, fitted in ((large_test, large_fit), (fewshot_test, fewshot_fit)):
        assert not {group[c.name] for c in reported} & {group[c.name] for c in fitted}
    return Split(large_test, large_fit, fewshot_fit, fewshot_test, set_aside, dropped, covered,
                 seed)


def _order(c: Crop) -> tuple[str, int]:
    return c.dataset, c.number


def data_yaml(crops: list[Crop], header: str) -> str:
    lines = [
        header,
        "# miao needs a global resolution; each volume overrides it with its raw's own voxel size.",
        "resolutions: [[8.0, 8.0, 8.0]]",
        "output_axes: lczyx",
        f"patch_size: {list(PATCH)}",
        "volumes:",
    ]
    for c in crops:
        lines += [
            f"- name: {c.name}",
            f"  path: {c.store}",
            "  image_key: raw",
            f"  label_key: {c.label_key}",
            "  zarr_version: zarr3",
            f"  resolutions: [{[float(v) for v in c.raw_voxel]}]",
            f"  bounding_box: {c.box()}",
        ]
    return "\n".join(lines) + "\n"


def scoring_toml(task: str, route: str, table: dict[str, list[int]], header: str) -> str:
    postprocess = {"identity": "identity", "argmax": "argmax"}[route]
    accepts = {"identity": "class_labels: uint8 CellMap ids, background_id 0",
               "argmax": "class_scores: channel k scores CellMap id k"}[route]
    classes = "\n".join(f"{name} = {ids}" for name, ids in table.items())
    notes = (f"CellMap organelle semantic segmentation, {len(table)} classes, scored on each "
             f"crop's own label grid; artifacts are {accepts}. See docs/cellmap.md.")
    return f'''{header}
task_name = "{task}"
route = "{route}"
notes = "{notes}"

[data.test]
config_path = "data/test.yaml"

[data.fit]                    # the crops a model may be fine-tuned on; nothing here is swept
config_path = "data/fit.yaml"

[task]
name = "semantic_seg"

[postprocess]
name = "{postprocess}"

[metric]
names = ["semantic"]
rank_by = "semantic"

[metric.semantic]
num_classes = 256             # the uint8 id space
ignore_truth = [0]            # unannotated (in complete crops: voxels CellMap marked unknown)

[metric.semantic.classes]     # each class as the CellMap ids its voxels may carry (classes.csv)
{classes}
'''


def summary(split_: Split, crops: list[Crop]) -> str:
    rows: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for key, cs in (("crops", crops), ("complete", [c for c in crops if c.complete]),
                    ("set aside", split_.set_aside + split_.dropped),
                    ("test", split_.large_test), ("fit", split_.large_fit),
                    ("fewshot fit", split_.fewshot_fit), ("fewshot test", split_.fewshot_test)):
        for c in cs:
            rows[c.dataset][key] += 1
    columns = ("crops", "complete", "set aside", "test", "fit", "fewshot fit", "fewshot test")
    lines = ["| dataset | raw voxel (nm) | " + " | ".join(columns) + " |",
             "|---|---|" + "---:|" * len(columns)]
    voxel = {c.dataset: c.raw_voxel for c in crops}
    for dataset in sorted(rows):
        size = " x ".join(f"{v:g}" for v in voxel[dataset])
        lines.append(f"| {dataset} | {size} | "
                     + " | ".join(str(rows[dataset][k]) for k in columns) + " |")
    totals = [sum(rows[d][k] for d in rows) for k in columns]
    lines.append("| **total** | | " + " | ".join(f"**{t}**" for t in totals) + " |")
    return "\n".join(lines)


def class_summary(split_: Split, table: dict[str, list[int]]) -> str:
    """Per class: how many complete crops of each split hold it, and its share of test voxels."""
    parts = (("test", split_.large_test),
             ("fit", [c for c in split_.large_fit if c.complete]),
             ("fewshot fit", split_.fewshot_fit), ("fewshot test", split_.fewshot_test))
    lines = ["| class | ids | " + " | ".join(name for name, _ in parts) + " | voxels in test |",
             "|---|---|" + "---:|" * (len(parts) + 1)]
    test_total = sum(n for c in split_.large_test for i, n in c.counts.items() if i)
    for name, ids in table.items():
        cells = [str(sum(any(c.counts.get(i) for i in ids) for c in cs)) for _, cs in parts]
        voxels = sum(c.counts.get(i, 0) for c in split_.large_test for i in ids)
        shown = ", ".join(map(str, ids)) if len(ids) <= 6 else f"{len(ids)} ids"
        lines.append(f"| {name} | {shown} | " + " | ".join(cells)
                     + f" | {voxels / test_total:.2%} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, required=True, help="mia-evals' configs/ directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lmd", type=Path, default=LMD)
    parser.add_argument("--cellmap", type=Path, default=CELLMAP)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("LSB_DJOB_NUMPROC", "8")))
    parser.add_argument("--dry-run", action="store_true", help="print the split, write nothing")
    args = parser.parse_args()

    crops, classes, digest = discover(args.lmd, args.cellmap)
    table = class_table(classes)
    check_reannotations(crops)
    complete = [c for c in crops if c.complete]
    with ThreadPoolExecutor(args.workers) as pool:
        for c, counts in zip(complete, pool.map(count_labels, complete), strict=True):
            c.counts = counts
    result = split(crops, table, args.seed)

    print(f"{len(crops)} crops in {len({c.dataset for c in crops})} datasets; "
          f"{len(complete)} complete")
    for label, cs in (("set aside", result.set_aside), ("dropped", result.dropped),
                      ("test", result.large_test), ("fit", result.large_fit),
                      ("fewshot fit", result.fewshot_fit), ("fewshot test", result.fewshot_test)):
        print(f"  {label:13s} {len(cs):4d} crops, {len({c.dataset for c in cs}):2d} datasets")
    missing = [n for n in TESTED if n not in result.covered]
    print(f"fewshot fit covers {len(result.covered)} of {len(TESTED)} classes; missing {missing}")
    print(summary(result, crops))
    print(class_summary(result, table))
    if args.dry_run:
        return

    from report.record import git_commit

    stamp = datetime.date.today().isoformat()
    commit = git_commit(Path(__file__).resolve().parents[2])
    for task, test, fit in ((TASKS[0], result.large_test, result.large_fit),
                            (TASKS[1], result.fewshot_test, result.fewshot_fit)):
        directory = args.configs / task
        (directory / "data").mkdir(parents=True, exist_ok=True)
        for part, cs in (("test", test), ("fit", fit)):
            header = (
                f"# {task} {part}: {len(cs)} CellMap crops from {len({c.dataset for c in cs})} "
                "datasets.\n"
                f"# Generated by `python -m truth.cellmap --seed {args.seed}` on {stamp} "
                f"(mia-evals {commit});\n# edit the builder, not this file. Inputs: {args.lmd} "
                f"(labels/{LABEL_GROUP}<N>),\n# {args.cellmap} (annotated classes), classes.csv "
                f"sha256 {digest}."
            )
            (directory / "data" / f"{part}.yaml").write_text(data_yaml(cs, header))
        for route in ("identity", "argmax"):
            header = (f"# Generated by `python -m truth.cellmap` on {stamp}; edit the builder, "
                      "not this file.")
            (directory / f"{route}.toml").write_text(scoring_toml(task, route, table, header))
        print(f"wrote {directory}")


if __name__ == "__main__":
    main()
