<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with `mia-evals leaderboard --task lmd_ssl_v1_neuron_instance`, or verify with `--check`.
     Rows come from ./records/; edit a record, not this table. -->

# lmd_ssl_v1_neuron_instance

A score here is only interpretable against a floor. See
[docs/controls.md](../../docs/controls.md) for what these tasks measure with no model at all.


**Region:** kasthuri15_ac4 256x640x640 (sub-region); liconn_expid82 768x1152x1152 (sub-region); liconn_mouse_hippocampus 512x512x512 (sub-region); zebrafish_fish2_doublecube1 1920x1920x1920 (sub-region)

| # | model | voxel_instance.pq (higher is better) | postprocess | voxel_instance.voi_merge | voxel_instance.voi_split | voxel_instance.sq | voxel_instance.rq | voxel_instance.adapted_rand_error |
|---|---|---|---|---|---|---|---|---|
| 1 | 2c_step50000_mws | 0.2287 | size_filter(min_size=50000) | 1.8565 | 1.3023 | 0.7072 | 0.3155 | 0.5380 |
| 2 | 2c_step50000_sizefilter | 0.1420 | cc_threshold(logit=+3, thr=0.6457, min_size=50000) | 3.7074 | 1.0675 | 0.6507 | 0.2110 | 0.7435 |
| 3 | 2c_step50000 | 0.0031 | cc_threshold(logit=+0, thr=0.5000) | 6.7441 | 0.4306 | 0.5048 | 0.0044 | 0.8968 |
