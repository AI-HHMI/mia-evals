<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with `mia-evals leaderboard --task lmd_ssl_v1_zebrafish_instance`, or verify with `--check`.
     Rows come from ./records/; edit a record, not this table. -->

# lmd_ssl_v1_zebrafish_instance

A score here is only interpretable against a floor. See
[docs/controls.md](../../docs/controls.md) for what these tasks measure with no model at all.

`links` open the scored artifacts and the checkpoint in fileglancer (Janelia login). A link is
only offered where the record names a path on this cluster; `missing` means the files have since
been deleted -- scratch is reclaimed once a number is recorded -- and the table was re-rendered.
Per-volume neuroglancer views of each row (raw image, scored labelling and ground truth placed
together) are a separate HTML page per task, written outside the repository into a fileglancer
data-link directory named in the untracked `leaderboard/fileglancer_shares.json`, because its
links carry keys that serve the files without a login on the Janelia network.


**Region:** zebrafish_fish2_doublecube1 1920x1920x1920 (sub-region)

| # | model | links | voxel_instance.pq (higher is better) | postprocess | voxel_instance.voi_merge | voxel_instance.voi_split | voxel_instance.sq | voxel_instance.rq | voxel_instance.adapted_rand_error |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1c_step50000 | [artifacts](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-scratch/lmd1_zebrafish_artifacts/arm1/test) · [checkpoint](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-runs/lmd1__1c_dinov3_simmim_ft_subpixel_20260827_231846/checkpoints/step_50000) | 0.0876 | size_filter(min_size=50000) | 3.6554 | 2.0653 | 0.7173 | 0.1221 | 0.9370 |
| 2 | 2c_step50000 | [artifacts](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-scratch/lmd1_zebrafish_artifacts/arm2/test) · [checkpoint](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-runs/lmd1__2c_dinov3_lvd_ft_subpixel_20260824_183919/checkpoints/step_50000) | 0.0862 | size_filter(min_size=50000) | 3.1383 | 2.0274 | 0.6978 | 0.1236 | 0.9317 |
| 3 | 3c_step50000 | [artifacts](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-scratch/lmd1_zebrafish_artifacts/arm3/test) · [checkpoint](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-runs/lmd1__3c_muvit_mae_ft_subpixel_20260905_074206/checkpoints/step_50000) | 0.0719 | size_filter(min_size=50000) | 4.6463 | 2.2950 | 0.7400 | 0.0972 | 0.9593 |
