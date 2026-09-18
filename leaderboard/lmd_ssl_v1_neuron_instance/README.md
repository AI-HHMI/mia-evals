<!-- GENERATED FILE -- do not edit by hand.
     Regenerate with `mia-evals leaderboard --task lmd_ssl_v1_neuron_instance`, or verify with `--check`.
     Rows come from ./records/; edit a record, not this table. -->

# lmd_ssl_v1_neuron_instance

The artifact/checkpoint/view links below point to locations on the Janelia cluster and will only work on the Janelia network.

**Views:** [neuroglancer views for every row below](https://fileglancer.int.janelia.org/files/dLLjve-fpPGXhnVi/nrs/scicompsoft/orhane/mia-evals/lmd_ssl_v1_neuron_instance/views/lmd_ssl_v1_neuron_instance.html)

**Region:** kasthuri15_ac4 256x640x640 (sub-region); liconn_expid82 768x1152x1152 (sub-region); liconn_mouse_hippocampus 512x512x512 (sub-region); zebrafish_fish2_doublecube1 1920x1920x1920 (sub-region)

| # | model | links | voxel_instance.pq (higher is better) | postprocess | voxel_instance.voi_merge | voxel_instance.voi_split | voxel_instance.sq | voxel_instance.rq | voxel_instance.adapted_rand_error |
|---|---|---|---|---|---|---|---|---|---|
| 1 | lmd1__1c_dinov3_simmim_ft_subpixel_20260827_231846.step50000.mws | [artifacts](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-experiments/lmd_ssl_v1/eval/neuron_artifacts/arm1/test) · [checkpoint](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-experiments/lmd_ssl_v1/runs/lmd1__1c_dinov3_simmim_ft_subpixel_20260827_231846/checkpoints/step_50000) | 0.2369 | size_filter(min_size=50000) | 2.0755 | 1.3125 | 0.7078 | 0.3241 | 0.5426 |
| 2 | lmd1__2c_dinov3_lvd_ft_subpixel_20260824_183919.step50000.mws | [artifacts](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-experiments/lmd_ssl_v1/eval/mws_artifacts/test) · [checkpoint](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-experiments/lmd_ssl_v1/runs/lmd1__2c_dinov3_lvd_ft_subpixel_20260824_183919/checkpoints/step_50000) | 0.2287 | size_filter(min_size=50000) | 1.8565 | 1.3023 | 0.7072 | 0.3155 | 0.5380 |
| 3 | sam1__arm4_8nm_gb16_r0_20260913_221637.step200000.size_filter | [artifacts](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-experiments/sam_lmd_v1/eval/arm4_8nm_gb16_r0/test) · [checkpoint](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-experiments/sam_lmd_v1/runs/sam1__arm4_8nm_gb16_r0_20260913_221637/checkpoints/step_200000) | 0.2094 | size_filter(min_size=5000) | 3.4492 | 0.9709 | 0.6909 | 0.2945 | 0.6829 |
| 4 | sam1__arm5_8nm_gb32_r0_20260913_223406.step200000.size_filter | [artifacts](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-experiments/sam_lmd_v1/eval/arm5_8nm_gb32_r0/test) · [checkpoint](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-experiments/sam_lmd_v1/runs/sam1__arm5_8nm_gb32_r0_20260913_223406/checkpoints/step_200000) | 0.2038 | size_filter(min_size=5000) | 3.9397 | 0.7063 | 0.7221 | 0.2775 | 0.6845 |
| 5 | lmd1__2c_dinov3_lvd_ft_subpixel_20260824_183919.step50000.cc_threshold | [artifacts](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-experiments/lmd_ssl_v1/eval/arm2_eval/test) · [checkpoint](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-experiments/lmd_ssl_v1/runs/lmd1__2c_dinov3_lvd_ft_subpixel_20260824_183919/checkpoints/step_50000) | 0.1420 | cc_threshold(logit=+3, thr=0.6457, min_size=50000) | 3.7074 | 1.0675 | 0.6507 | 0.2110 | 0.7435 |
| 6 | lmd1__2c_dinov3_lvd_ft_subpixel_20260824_183919.step50000.cc_threshold_nosizesweep | [artifacts](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-experiments/lmd_ssl_v1/eval/arm2_eval/test) · [checkpoint](https://fileglancer.int.janelia.org/browse/nrs_scicompsoft/orhane/mia-train-experiments/lmd_ssl_v1/runs/lmd1__2c_dinov3_lvd_ft_subpixel_20260824_183919/checkpoints/step_50000) | 0.0031 | cc_threshold(logit=+0, thr=0.5000) | 6.7441 | 0.4306 | 0.5048 | 0.0044 | 0.8968 |
