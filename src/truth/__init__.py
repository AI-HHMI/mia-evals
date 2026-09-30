"""Building a task's benchmark data from a public release: what its data configs point at.

Scoring never runs this. Each module builds one dataset, once, as
`python -m truth.<dataset> --out <dir>`, and everything it writes is either metadata over the
release's own arrays or ground truth derived from them. The release is read-only and is never
copied: an OME-Zarr wrapper carries the metadata that `miao` and the scorer need and symlinks each
level into the release, so a wrapper costs kilobytes where a copy would cost terabytes.

Needs the optional `[truth]` extra (`pymongo` for the BSON dumps, `kimimaro` for skeletonising
voxel ground truth) on top of what scoring needs.
"""
