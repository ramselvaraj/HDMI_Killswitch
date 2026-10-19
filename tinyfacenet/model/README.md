# TinyFaceMatch128 PyTorch

Clean-room TinyFaceMatch-style training code for a 128-D face embedding model.

The inference backbone is:

```text
112x112 RGB
-> ConvBNPReLU
-> DepthwiseSeparable
-> Residual bottleneck stack
-> 7x7 global depthwise conv
-> Linear 128
-> BatchNorm1d
-> L2 normalize
```

The ArcFace classifier is used only during training. The saved checkpoint contains
`backbone_state_dict` for deployment and `classifier_state_dict` for resuming or
inspection.

## Train

Expected data layout:

```text
CASIA_subset_500x25/
  0000168/
    image1.jpg
    image2.jpg
  0000192/
    image1.jpg
```

PYNQ-Z2-friendly first run:

```powershell
python -m tinyfacematch_pytorch.train `
  --data-dir C:\ramse\hls4ml_projects\tinyFaceNet\CASIA_subset_500x25 `
  --output-dir C:\ramse\hls4ml_projects\tinyFaceNet\runs\tinyfacematch128_w05 `
  --width-mult 0.5 `
  --scale 30 `
  --margin 0.35 `
  --margin-warmup-epochs 5 `
  --epochs 30 `
  --batch-size 64 `
  --augment
```

Closer to the published TinyFaceMatch size:

```powershell
python -m tinyfacematch_pytorch.train `
  --data-dir C:\ramse\hls4ml_projects\tinyFaceNet\CASIA_subset_500x25 `
  --output-dir C:\ramse\hls4ml_projects\tinyFaceNet\runs\tinyfacematch128_w25 `
  --width-mult 2.5 `
  --scale 30 `
  --margin 0.35 `
  --margin-warmup-epochs 5 `
  --epochs 30 `
  --batch-size 32 `
  --augment
```

## Export Backbone

```powershell
python -m tinyfacematch_pytorch.export_onnx `
  --checkpoint C:\ramse\hls4ml_projects\tinyFaceNet\runs\tinyfacematch128_w05\best.pt `
  --output C:\ramse\hls4ml_projects\tinyFaceNet\runs\tinyfacematch128_w05\tinyfacematch128.onnx
```

The ONNX model output is always `N x 128`.

## Evaluate Pair Verification

```powershell
python -m tinyfacematch_pytorch.eval_pairs `
  --checkpoint C:\ramse\hls4ml_projects\tinyFaceNet\runs\tinyfacematch128_w05\best.pt `
  --data-dir C:\ramse\hls4ml_projects\tinyFaceNet\CASIA_subset_500x25 `
  --same-pairs 5000 `
  --diff-pairs 5000 `
  --output-json C:\ramse\hls4ml_projects\tinyFaceNet\runs\tinyfacematch128_w05\pair_eval.json
```

The score is cosine similarity between two L2-normalized 128-D embeddings.

## Prepare HLS Weights

Fold BatchNorm into Conv/Linear layers, quantize weights, and generate files for
Vitis HLS integration:

```powershell
python -m tinyfacematch_pytorch.prepare_hls_weights `
  --checkpoint C:\ramse\hls4ml_projects\tinyFaceNet\runs\tinyfacematch128_w05_v4_continue\last.pt `
  --output-dir C:\ramse\hls4ml_projects\tinyFaceNet\runs\tinyfacematch128_w05_v4_continue\hls_weights_q12
```

Generated files:

```text
weights.bin
weights_quantized.npz
weights_folded_fp32.npz
manifest.json
weights_offsets.hpp
```

Evaluate the prepared quantized reference:

```powershell
python -m tinyfacematch_pytorch.eval_hls_prepared `
  --prepared-dir C:\ramse\hls4ml_projects\tinyFaceNet\runs\tinyfacematch128_w05_v4_continue\hls_weights_q12 `
  --checkpoint C:\ramse\hls4ml_projects\tinyFaceNet\runs\tinyfacematch128_w05_v4_continue\last.pt `
  --data-dir C:\ramse\hls4ml_projects\tinyFaceNet\CASIA_subset_500x25 `
  --same-pairs 5000 `
  --diff-pairs 5000 `
  --batch-size 128 `
  --output-json C:\ramse\hls4ml_projects\tinyFaceNet\runs\tinyfacematch128_w05_v4_continue\hls_weights_q12\pair_eval_quantized.json
```
