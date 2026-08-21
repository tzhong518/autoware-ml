---
icon: lucide/cctv
---

# StreamPETR

<!-- cspell:ignore CPFPN kokseang fcos3d imgbackbone md5sum -->

StreamPETR is a camera-based 3D object detection model integrated under the `detection3d` task namespace. It uses a multiview image backbone, a feature pyramid neck, and a native query-based detection head with a streaming memory queue for temporal modeling.

## Summary

| Property     | Value                                       |
|--------------|---------------------------------------------|
| Task         | 3D object detection                         |
| Modality     | Camera                                      |
| Input        | Synchronized multiview images               |
| Output       | 3D bounding boxes and class scores          |
| Architecture | Multiview VoVNet/CPFPN + query decoder head |
| Datasets     | NuScenes, T4Dataset                         |

## Available Configurations

| Config Name                                            | Dataset   | Purpose                                               |
|--------------------------------------------------------|-----------|-------------------------------------------------------|
| `detection3d/streampetr/vov_320x800_nuscenes_pretrain` | NuScenes  | Three-stage flow, stage 1: nuScenes pretrain          |
| `detection3d/streampetr/vov_480x640_t4dataset_base`    | T4Dataset | Three-stage flow, stage 2: full T4 base DB, 35 epochs |
| `detection3d/streampetr/vov_480x640_t4dataset_j6gen2`  | T4Dataset | Default T4 recipe, and three-stage flow stage 3       |

The T4 default is 2 GPUs x batch 8 (total 16), 35 epochs, AdamW lr 1.0e-4 with
`img_backbone` lr_mult 0.1, 500-iteration warmup into per-epoch cosine decay,
pc_range ±51.2 m, full train-time augmentation (resize/flip, global
rot/scale, per-frame camera shuffling, grid mask), an auxiliary 2D
`FocalHead2D`, `traffic_cone`/`barrier` partial-ignore, seed 0, and
mAP-based checkpoint selection. There is no `auto_scale_lr`: the config pins
`trainer.devices: 2`; for any other total batch size N rescale both LRs by
N/16 via the optimizer overrides, e.g. for 4 GPUs x batch 8 (total 32):

```bash
batch_size=8 trainer.devices=4 \
    model.optimizer.lr=2.0e-4 \
    model.optimizer_group_overrides.img_backbone.lr=2.0e-5
```

## Training Workflow (T4 / j6gen2)

### 1. Initialization

Train the nuScenes pretrain stage first and initialize from its checkpoint —
see [the three-stage flow](#three-stage-training-flow), whose stage 1 produces
exactly this. Initializing straight from the DD3D/FCOS3D image backbone also
works but converges to a weaker model, since the detection head then starts
from scratch on T4 data alone.

### 2. Train

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2 \
    --weights mlruns/detection3d/streampetr/vov_320x800_nuscenes_pretrain/<run_id>/artifacts/checkpoints/best.ckpt \
    datamodule.data_root=<data_root> \
    datamodule.train_ann_file=<infos_train.pkl> \
    datamodule.val_ann_file=<infos_val.pkl> \
    datamodule.test_ann_file=<infos_test.pkl>
```

For a pipeline validation run add `+trainer.fast_dev_run=true`. For a short
real run instead (e.g. `trainer.max_epochs=1`) also set
`trainer.check_val_every_n_epoch=1`: validation defaults to every 5 epochs,
and a run that finishes without ever validating fails at teardown when the
`val/loss` metric is missing. Full-length training is unaffected.

### 3. Evaluate

```bash
autoware-ml test \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2 \
    --weights mlruns/detection3d/streampetr/vov_480x640_t4dataset_j6gen2/<run_id>/artifacts/checkpoints/best.ckpt \
    trainer.devices=1
```

The headline number is the `0-121m` bucket mAP (BEV center distance); see the
evaluation notes below for what that bucket covers.

### 4. Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2 \
    --weights mlruns/.../checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false
```

The current verification scope covers ONNX export. TensorRT engine
generation has not been validated yet.

## Three-Stage Training Flow

The workflow above trains j6gen2 directly. The full flow inserts a T4 base-DB
stage in between, which is the recommended path:

| Stage | Config                                                         | Data                             | Epochs | Init from                              |
|-------|----------------------------------------------------------------|----------------------------------|--------|----------------------------------------|
| 1     | `detection3d/streampetr/vov_320x800_nuscenes_pretrain`         | nuScenes                         | 30     | DD3D/FCOS3D VoVNet-99 backbone         |
| 2     | `detection3d/streampetr/vov_480x640_t4dataset_base`            | `t4dataset_base_infos_*`         | 35     | stage-1 checkpoint                     |
| 3     | `detection3d/streampetr/vov_480x640_t4dataset_j6gen2`          | `t4dataset_j6gen2_base_infos_*`  | 35     | stage-2 checkpoint                     |

Each stage passes its Lightning checkpoint straight to the next — no weight
conversion happens anywhere inside the flow. All three pin `trainer.devices: 2`
so the total batch stays 16 and the hard-coded LRs hold; for any other total
batch size N rescale both LRs by N/16 with the same
`model.optimizer.lr` / `model.optimizer_group_overrides.img_backbone.lr`
overrides shown above (stage 1's base LRs are 4.0e-4 / 4.0e-5, stages 2 and 3
use 1.0e-4 / 1.0e-5). Validation runs every 5 epochs.

### Stage 1 — nuScenes pretrain

`vov_320x800_nuscenes_pretrain` uses the CPFPN neck and the auxiliary
`FocalHead2D`, matching the T4 stages so its checkpoint loads into stage 2
as-is, and trains with lr 4e-4, grad-clip 35, `eta_min = lr * 1e-3`, random
flip and global rot/scale. It needs the nuScenes dataset mounted and the
DD3D/FCOS3D-pretrained VoVNet-99 image backbone as `--weights`; without that
backbone init the recipe underperforms.

That backbone is the only external artifact the flow uses. Download it from the
upstream StreamPETR release and flip its stem from BGR to RGB once (expected
output: `Kept 626 backbone tensors; dropped 81`):

```bash
curl -L -o fcos3d_vovnet_imgbackbone-remapped.pth \
    https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth
md5sum fcos3d_vovnet_imgbackbone-remapped.pth
# ff1ac3040eabf0f0e54c3c594c26021e
python -m autoware_ml.tools.convert_streampetr_checkpoint \
    --input fcos3d_vovnet_imgbackbone-remapped.pth \
    --output fcos3d_vovnet_imgbackbone-remapped_converted.pth \
    --bgr-to-rgb
```

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_320x800_nuscenes_pretrain \
    --weights fcos3d_vovnet_imgbackbone-remapped_converted.pth \
    datamodule.data_root=<nuscenes_root>
```

The training log should report `Loaded matching weight tensors: 626/1526
(+626 shared-tensor aliases)` for this backbone init — the backbone only, with
the neck and both heads starting from their own initialization.

### Stage 2 — T4 base DB

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_base \
    --weights mlruns/detection3d/streampetr/vov_320x800_nuscenes_pretrain/<run_id>/artifacts/checkpoints/best.ckpt \
    datamodule.data_root=<t4_data_root>
```

The config defaults to `info/detection3d/t4dataset_base_infos_{train,val,test}.pkl`
under `data_root`; override `datamodule.{train,val,test}_ann_file` for a
different info directory. When loading the stage-1 checkpoint (and likewise
the stage-2 checkpoint in stage 3) the log should report
`Loaded matching weight tensors: 1526/1526 (+0 shared-tensor aliases)` —
full coverage, nothing dropped.

### Stage 3 — j6gen2 fine-tune

Stage 3 is the T4 default config again; only `--weights` changes, pointing at
the stage-2 run instead of the stage-1 pretrain:

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2 \
    --weights mlruns/detection3d/streampetr/vov_480x640_t4dataset_base/<run_id>/artifacts/checkpoints/best.ckpt \
    datamodule.data_root=<t4_data_root>
```

`--weights` accepts the stage-2 Lightning checkpoint directly — no conversion
is needed between autoware-ml stages. Evaluate and deploy the stage-3
checkpoint exactly as in steps 3 and 4 above.

## Results (legacy conversion route, verified 2026-08-06)

These numbers predate the native three-stage flow: they come from the earlier
route that initialized j6gen2 from an externally trained checkpoint converted
into this module tree. They are kept as the current known baseline for the
recipe — the native flow's own numbers are not in yet.

The run used the 10-epoch schedule the config carried at the time, so
reproducing it needs
`trainer.max_epochs=10 trainer.check_val_every_n_epoch=1` on top of today's
35-epoch default.

Both frameworks trained the same recipe on the same j6gen2 data split
(kokseang_2_8 infos) from the same converted pretrain, and were scored with
the aligned evaluation (full ±51.2 m square GT, min-points filter engaged):

| Framework / checkpoint                      | Training               | val mAP     | test mAP    |
|---------------------------------------------|------------------------|-------------|-------------|
| **autoware-ml** (this recipe, run 92068f7b) | bf16, global loss norm | **0.39127** | **0.36609** |
| AWML (aligned_bf16, epoch 9)                | bf16                   | 0.37521     | 0.35515     |

Cross-checks that back these numbers:

- **Metric stacks are equivalent**: scoring an identical set of predictions
  and GT with AWML T4MetricV2 and with autoware-ml `MeanAP` agrees to
  ≤ 2.5e-8 per class (pure float32 round-trip noise).
- **Same-weights residual**: running one AWML checkpoint through both
  inference stacks leaves −0.80 mAP (test), consistent in sign across all 7
  classes — numerics of the camera pipeline (image decode/resize, attention
  kernels, fp16-vs-fp32 paths), not an evaluator or recipe difference.
- **Training-parity checklist**: identical pretrained init (880/880 shared
  tensors byte-equal), per-epoch LR schedule matches AWML's logged values
  (< 0.5 %), batch/optimizer/grad-clip/hooks equal.

## Evaluation Notes

Two evaluation-side fixes are part of the default config. Both were originally
found while aligning against the external evaluator used for the legacy results
above, and both are correctness fixes in their own right:

1. **`gt_num_points` collation** — without it the evaluation-time min-points
   GT filter silently never engages.
2. **No radial `eval_class_range` cap inside the pc_range square** — the GT is
   already limited to the ±51.2 m *square* by the pipeline
   `ObjectRangeFilter`. An additional radial cap at 51.2/54 m removed the
   square's corners from the GT while corner *predictions* stayed and became
   guaranteed false positives. The config therefore restates the 121 m dataset
   default, which is a no-op inside the square, and reports the
   0-50 / 50-90 / 90-121 / 0-121 m buckets.

## Implementation

| Path                                                        | Description                                          |
|-------------------------------------------------------------|------------------------------------------------------|
| `autoware_ml/models/detection3d/streampetr.py`              | StreamPETR model wrapper                             |
| `autoware_ml/models/detection3d/heads/streampetr.py`        | Query-based detection head                           |
| `autoware_ml/models/detection3d/heads/focal2d.py`           | Auxiliary 2D head (FocalHead2D)                      |
| `autoware_ml/models/detection3d/partial_ignore.py`          | Partial-ignore label handling                        |
| `autoware_ml/losses/detection2d/losses.py`                  | Quality-focal and GIoU losses for the 2D head        |
| `autoware_ml/models/common/backbones/vovnet.py`             | Multiview image backbone                             |
| `autoware_ml/models/common/necks/cp_fpn.py`                 | CPFPN image neck                                     |
| `autoware_ml/models/detection3d/task_modules/`              | Shared assigners, costs, coders, streaming memory    |
| `autoware_ml/datamodule/common/multiview_detection3d.py`    | Shared multiview detection dataset                   |
| `autoware_ml/datamodule/nuscenes/multiview_detection3d.py`  | NuScenes multiview datamodule                        |
| `autoware_ml/datamodule/t4dataset/multiview_detection3d.py` | T4Dataset multiview datamodule                       |
| `autoware_ml/utils/schedulers/iter_warmup_epoch_cosine.py`  | Warmup + epoch-cosine LR schedule                    |
| `autoware_ml/tools/convert_streampetr_checkpoint.py`        | Image-backbone checkpoint preparation                |
| `autoware_ml/configs/tasks/detection3d/streampetr/`         | Task configurations                                  |

## Acknowledgment

<!-- cspell:ignore exiawsh -->
The Autoware-ML StreamPETR implementation was ported from the official streampetr
project by exiawsh.

<!-- cspell:ignore Shihao -->
- Repository: <https://github.com/exiawsh/streampetr>
- License: Apache License 2.0
- Paper: Wang, Shihao, et al. "Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection" ICCV, 2023.
