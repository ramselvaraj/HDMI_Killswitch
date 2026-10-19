# TinyFaceMatch HLS System Report

This document summarizes the complete TinyFaceMatch flow in this repository: the PyTorch training path, the HLS weight preparation path, the final FPGA IP architecture, the Vivado/PYNQ integration, and the measured runtime behavior on hardware.

## 1. System Overview

The project implements a compact face-embedding pipeline for FPGA deployment. The final system takes a 112x112 RGB face image and produces a 128-dimensional L2-normalized embedding. Matching is done by cosine similarity between a live embedding and one or more enrolled/authorized embeddings.

The final working FPGA implementation is:

```text
hls_ddr_intmac/
```

The packaged Vivado IP is:

```text
hls_ddr_intmac/tinyfacematch_hls_ddr_intmac/solution1/impl/ip
```

The current deployed bitstream used for benchmarking was:

```text
hvCNN_INT.bit
```

The final measured dummy embedding latency on PYNQ was:

```text
CNN-only mean:       1287.28 ms
Full embedding mean: 1288.66 ms
```

The non-CNN software/buffer overhead was only about:

```text
1.38 ms
```

So almost all embedding time is the HLS CNN accelerator itself.

## 2. Repository Components

Important directories and files:

```text
tinyfacematch_pytorch/
  model.py                    PyTorch model definition
  data.py                     face dataset loader and preprocessing
  train.py                    ArcFace-style training script
  prepare_hls_weights.py      folds BN, quantizes weights, writes weights.bin
  eval_hls_prepared.py        evaluates quantized/prepared weights in PyTorch

hls/
  original HLS implementation and test vectors

hls_ddr/
  first viable DDR-backed HLS implementation

hls_ddr_fast/
  cached-weight + 1x1 fast-path version

hls_ddr_intmac/
  final integer-MAC HLS implementation

scripts/
  pynq_hls_camera_verify.py   standalone PYNQ/OpenCV verification flow
```

The latest trained checkpoint used in this flow is:

```text
runs/tinyfacematch128_w05_v4_continue/last.pt
```

The prepared HLS weights are:

```text
runs/tinyfacematch128_w05_v4_continue/hls_weights_q12/weights.bin
```

## 3. PyTorch Model

The model is defined in:

```text
tinyfacematch_pytorch/model.py
```

The inference backbone is `TinyFaceMatch128`. It produces a fixed 128-dimensional embedding.

The architecture is a small MobileFaceNet-style CNN:

```text
Input: 3 x 112 x 112

conv0:       Conv-BN-PReLU, 3x3 stride 2
dwsep0:      depthwise 3x3 + pointwise 1x1

b0:          bottleneck, stride 2
b1:          bottleneck, residual
b2:          bottleneck, residual
b3:          bottleneck, stride 2
b4:          bottleneck, residual
b5:          bottleneck, residual
b6:          bottleneck, stride 2
b7:          bottleneck, residual

conv_final:  1x1 Conv-BN-PReLU
global_dw7:  global depthwise 7x7 Conv-BN
dense:       Linear to 128
embedding_bn
L2 normalize
```

With the default `width_mult=0.5`, the channel sizes are:

```text
c32  = 16
c64  = 32
c128 = 64
c256 = 128
```

The inference model returns:

```python
F.normalize(x, p=2, dim=1)
```

So each embedding is unit length and cosine similarity can be computed as a dot product.

## 4. Training Flow

Training is handled by:

```text
tinyfacematch_pytorch/train.py
```

The training model is `TinyFaceMatchTrainingModel`, which combines:

```text
TinyFaceMatch128 backbone
ArcFaceClassifier training head
```

The ArcFace classifier is training-only. It is not exported to HLS. It improves embedding separability by applying angular margin loss during classification training.

The dataset loader is:

```text
tinyfacematch_pytorch/data.py
```

It expects an ImageFolder-style identity layout:

```text
data_root/
  identity_000/
    img1.jpg
    img2.jpg
  identity_001/
    img1.jpg
```

Each image is:

```text
converted to RGB
resized to 112x112
normalized as (pixel - 127.5) / 128.0
returned as CHW float32
```

Training uses:

```text
optimizer: AdamW
loss: CrossEntropyLoss over ArcFace logits
scheduler: CosineAnnealingLR
optional AMP
optional random horizontal flip/crop augmentation
margin warmup before enabling ArcFace margin
```

Checkpoints contain:

```text
backbone_state_dict
classifier_state_dict
optimizer_state_dict
scheduler_state_dict
config
class_to_index
```

The important exportable part is only:

```text
backbone_state_dict
```

## 5. Weight Preparation for HLS

Weight preparation is handled by:

```text
tinyfacematch_pytorch/prepare_hls_weights.py
```

This script converts a trained PyTorch checkpoint into files usable by the HLS IP.

It performs four main jobs:

1. Rebuild the `TinyFaceMatch128` model using the checkpoint config.
2. Load the trained `backbone_state_dict`.
3. Fold BatchNorm into Conv/Linear weights.
4. Quantize and pack all weights into a flat binary.

### 5.1 BatchNorm Folding

For Conv-BN blocks, the script folds batch normalization into convolution weights:

```text
folded_weight = conv_weight * gamma / sqrt(var + eps)
folded_bias   = beta + (conv_bias - mean) * gamma / sqrt(var + eps)
```

For Linear-BN, it does the same with the dense layer and final embedding batchnorm.

This is important because the HLS inference engine does not implement separate BatchNorm layers. Each convolution or dense layer already includes the folded bias.

### 5.2 Quantization Formats

The final HLS quantization constants are generated into `weights_offsets.hpp`:

```text
TFM_WEIGHT_FRAC_BITS = 12
TFM_BIAS_FRAC_BITS   = 20
TFM_PRELU_FRAC_BITS  = 12
TFM_ACT_FRAC_BITS    = 10
TFM_OUT_FRAC_BITS    = 14
```

The runtime data formats are:

```text
image input:    uint8 RGB HWC
activations:    int16 Q6.10
weights:        int16 Q*.12
biases:         int32 Q*.20
PReLU:          int16 Q*.12
output vector:  int16 Q*.14
```

The final prepared files include:

```text
weights.bin              flat int16/int32-packed binary used by HLS
weights_offsets.hpp      C++ offsets and quantization constants
manifest.json            layer metadata and quantization config
weights_quantized.npz    debug/validation arrays
weights_folded_fp32.npz  folded float weights for debugging
```

For the current model:

```text
weights.bin bytes: 258688
weights words:     129344 int16 words
```

Biases are int32, but in `weights.bin` they occupy two int16 words. HLS reconstructs them with `read_i32_words()`.

### 5.3 Layer Packing Order

The packed binary follows the model execution order:

```text
conv0
dwsep0_dw
dwsep0_pw
b0_expand
b0_dw
b0_project
b1_expand
b1_dw
b1_project
b2_expand
b2_dw
b2_project
b3_expand
b3_dw
b3_project
b4_expand
b4_dw
b4_project
b5_expand
b5_dw
b5_project
b6_expand
b6_dw
b6_project
b7_expand
b7_dw
b7_project
conv_final
global_dw7
dense
```

Each layer is stored as:

```text
weights
bias
optional PReLU
```

The HLS code uses generated byte offsets such as:

```cpp
TFM_B4_EXPAND_OFFSET_BYTES / 2
```

The division by 2 converts byte offsets into int16 word offsets.

## 6. Quantized Reference Evaluation

The script:

```text
tinyfacematch_pytorch/eval_hls_prepared.py
```

builds a PyTorch reference model from the prepared HLS weights. It dequantizes the packed arrays and simulates fake quantization after layers.

This lets you compare:

```text
full precision PyTorch checkpoint
prepared/folded/quantized HLS-style model
```

It also evaluates pairwise cosine similarity over same/different face pairs and reports:

```text
best threshold
accuracy
tpr/fpr
precision
eer approximation
prepared-vs-PyTorch embedding error
```

This is the software-side sanity check before building HLS.

## 7. HLS Evolution

Several HLS versions were explored.

### 7.1 Original `hls/`

The first HLS implementation kept large activation tensors as static on-chip arrays. It passed C simulation after fixing fixed-point saturation issues, but it was far too large for the Zynq-7020.

Original C simulation after fixes:

```text
max_abs_error_int16=83
mean_abs_error_int16=25.875
cosine_to_expected=0.999743
PASSED
```

Vivado placement failed due to large over-utilization:

```text
LUT ~272k vs 53k available
FF  ~240k vs 106k available
BRAM36 176 vs 140 available
```

Conclusion: on-chip feature-map storage was not viable.

### 7.2 `hls_resource_serial/`

A serial/resource-shared version reduced some parallelism but still kept too much data on-chip. It remained too large:

```text
BRAM_18K=895
DSP=79
FF=113344
LUT=205926
```

Conclusion: serializing compute alone was not enough.

### 7.3 `hls_ddr/`

The first viable version moved intermediate activation tensors into external DDR scratch buffers:

```text
scratch0: int16[64 * 56 * 56]
scratch1: int16[64 * 56 * 56]
scratch2: int16[64 * 56 * 56]
```

This version reused generic convolution, depthwise, and dense engines. It fit:

```text
BRAM_18K=7
DSP=21
FF=20523
LUT=39350
```

Board test passed:

```text
elapsed_sec: 3.6342
max_abs_error: 83
mean_abs_error: 25.7734
cosine: 0.9997449
passed: True
```

Conclusion: DDR scratch made the design viable, but latency was still high.

### 7.4 `hls_ddr_fast/`

This version kept the DDR scratch architecture but added:

```text
per-channel weight caching
specialized 1x1 pointwise convolution
two-output-channel pointwise processing
dense input caching
```

It improved estimated latency but did not dramatically improve board latency. The HLS estimate was:

```text
BRAM_18K=10
DSP=26
FF=20116
LUT=38857
top max latency estimate: 15.407 sec
```

Conclusion: useful, but fixed-point conversion overhead was still a major bottleneck.

### 7.5 Final `hls_ddr_intmac/`

The final and best version keeps the DDR scratch architecture but rewrites the convolution hot loops as integer MACs.

Final C simulation:

```text
max_abs_error_int16=16
mean_abs_error_int16=4.69531
cosine_to_expected=0.999991
PASSED
```

Final HLS estimate:

```text
BRAM_18K=10
DSP=50
FF=16709
LUT=27004
top max latency estimate: 8.924 sec
```

This is both faster and much smaller in LUTs than `hls_ddr_fast`.

## 8. Final HLS IP Architecture

The final HLS IP lives in:

```text
hls_ddr_intmac/tinyfacematch_ip.cpp
```

The top function is:

```cpp
void tinyfacematch_ip(
    const ap_uint<8>* image_rgb,
    const ap_int<16>* weights,
    ap_int<16>* scratch0,
    ap_int<16>* scratch1,
    ap_int<16>* scratch2,
    ap_int<16>* embedding_out
);
```

The AXI interfaces are:

```text
image_rgb:      m_axi gmem0
weights:        m_axi gmem1
scratch0:       m_axi gmem2
scratch1:       m_axi gmem2
scratch2:       m_axi gmem2
embedding_out:  m_axi gmem2
control:        s_axilite
```

The AXI-Lite pointer offsets are unchanged across the DDR versions:

```text
CTRL_OFFSET     = 0x00
IMAGE_OFFSET    = 0x10
WEIGHTS_OFFSET  = 0x1c
SCRATCH0_OFFSET = 0x28
SCRATCH1_OFFSET = 0x34
SCRATCH2_OFFSET = 0x40
OUT_OFFSET      = 0x4c
```

The IP expects PYNQ/PS software to allocate DDR buffers and write their physical addresses into these registers.

## 9. Final HLS Compute Kernels

The final HLS has four main kernels:

```cpp
run_conv_image(...)
run_conv1x1(...)
run_depthwise(...)
run_dense_l2(...)
```

### 9.1 `run_conv_image`

This handles the first RGB image convolution.

Input is uint8 RGB HWC. The code converts pixels directly to Q6.10 integer activations:

```cpp
pixel_q10 = pixel * 8 - 1020
```

This corresponds to:

```text
(pixel - 127.5) / 128.0
```

in Q10 scale.

Weights are cached per output channel, then MACs are accumulated in `ap_int<48>`.

### 9.2 `run_conv1x1`

This handles all pointwise convolutions:

```text
expand layers
project layers
dwsep pointwise
conv_final
```

It processes two output channels at a time:

```text
w_cache0[]
w_cache1[]
acc0
acc1
```

For each spatial location and input channel, one input activation is reused for two output accumulators.

Residual connections are added in Q6.10 with saturation.

### 9.3 `run_depthwise`

This handles:

```text
3x3 depthwise convolutions
final 7x7 global depthwise convolution
```

For each channel, it caches the channel's depthwise kernel weights, then computes each output spatial position.

### 9.4 `run_dense_l2`

The final dense layer and L2 normalization are still implemented mostly with fixed-point arithmetic. This is acceptable because it is not the main latency bottleneck.

It:

```text
reads 128 input activations
computes 128 dense outputs
computes sum of squares
computes inverse norm with hls::sqrt
writes int16 Q14 normalized embedding
```

## 10. Integer MAC Arithmetic

The key final optimization is replacing repeated `ap_fixed` conversions in the convolution hot loops with integer arithmetic.

The main arithmetic relationship is:

```text
activation_q10 * weight_q12 = accumulator_q22
bias_q20 << 2               = accumulator_q22
output_q10                  = round_shift(accumulator_q22, 12)
```

For PReLU:

```text
negative_activation_q10 * alpha_q12 = q22
round_shift(q22, 12)                = q10
```

Helper functions include:

```cpp
read_act_raw()
read_image_q10()
saturate_i16()
round_shift_sat_i16()
apply_prelu_q10()
add_q10_sat()
```

This improves both resource usage and latency because HLS can map integer multiply-accumulate operations efficiently onto DSP slices. It also removes LUT-heavy fixed-point conversion, alignment, rounding, and saturation logic from the hot loop.

The tradeoff is expected and healthy:

```text
hls_ddr_fast:
  DSP=26
  LUT=38857

hls_ddr_intmac:
  DSP=50
  LUT=27004
```

DSP usage increased, but LUT usage dropped significantly. On Zynq-7020 this is a good tradeoff because the LUT budget was the limiting factor.

## 11. Layer Execution Order in HLS

Inside the top function, layers execute in this order:

```text
conv0 image -> scratch0

dwsep0 depthwise -> scratch1
dwsep0 pointwise -> scratch0

b0 expand -> scratch1
b0 depthwise stride2 -> scratch2
b0 project -> scratch0

b1 expand -> scratch1
b1 depthwise -> scratch2
b1 project + residual -> scratch0

b2 expand -> scratch1
b2 depthwise -> scratch2
b2 project + residual -> scratch0

b3 expand -> scratch1
b3 depthwise stride2 -> scratch2
b3 project -> scratch0

b4 expand -> scratch1
b4 depthwise -> scratch2
b4 project + residual -> scratch0

b5 expand -> scratch1
b5 depthwise -> scratch2
b5 project + residual -> scratch0

b6 expand -> scratch1
b6 depthwise stride2 -> scratch2
b6 project -> scratch0

b7 expand -> scratch1
b7 depthwise -> scratch2
b7 project + residual -> scratch0

conv_final -> scratch1
global_dw7 -> scratch0
dense + L2 normalize -> embedding_out
```

The scratch buffers are reused to avoid on-chip storage explosion.

## 12. HLS Build and Packaging

The final HLS build scripts are in:

```text
hls_ddr_intmac/
  run_hls_csim.tcl
  package_ip.tcl
  package_ip_no_csim.tcl
```

Useful commands:

```sh
source /mnt/OS/vivado_linux/2025.2/Vitis/settings64.sh
export LD_LIBRARY_PATH=/mnt/OS/vivado_linux/2025.2/Vitis/lib/lnx64.o/Ubuntu/24:$LD_LIBRARY_PATH

cd /mnt/OS/ramse/hls4ml_projects/tinyFaceNet/hls_ddr_intmac

vitis-run --mode hls --tcl run_hls_csim.tcl
vitis-run --mode hls --tcl package_ip_no_csim.tcl
```

The `LD_LIBRARY_PATH` line was needed because this Linux environment needed Vitis' bundled `libncurses.so.5`.

To check resources:

```sh
grep -E "BRAM|DSP|FF|LUT|URAM|Total|Available|Utilization" \
  tinyfacematch_hls_ddr_intmac/solution1/syn/report/tinyfacematch_ip_csynth.rpt
```

Packaged IP:

```text
hls_ddr_intmac/tinyfacematch_hls_ddr_intmac/solution1/impl/ip
```

IP archive:

```text
hls_ddr_intmac/tinyfacematch_hls_ddr_intmac/solution1/impl/ip/xilinx_com_hls_tinyfacematch_ip_1_0.zip
```

## 13. Vivado Block Design Integration

In Vivado:

1. Add the HLS IP repository:

```text
/mnt/OS/ramse/hls4ml_projects/tinyFaceNet/hls_ddr_intmac/tinyfacematch_hls_ddr_intmac/solution1/impl/ip
```

2. Refresh the IP catalog.

3. Add or replace the TinyFaceMatch IP in the block design.

4. Connect:

```text
s_axi_control -> Zynq GP AXI control path
m_axi_gmem0   -> PS HP/DDR path
m_axi_gmem1   -> PS HP/DDR path
m_axi_gmem2   -> PS HP/DDR path
ap_clk        -> PL clock
ap_rst_n      -> active-low reset
```

5. Use a clock constraint around 10 ns if targeting 100 MHz:

```tcl
create_clock -period 10.000 [get_ports ap_clk]
```

6. Generate bitstream and export `.bit`/`.hwh` for PYNQ.

The IP instance found by PYNQ was:

```text
tinyfacematch_ip_0
```

## 14. PYNQ / PS-Side Runtime

The standalone PYNQ helper is:

```text
scripts/pynq_hls_camera_verify.py
```

The core PS-side tasks are:

1. Load overlay:

```python
ol = Overlay("hvCNN_INT.bit")
```

2. Resolve the HLS IP:

```python
ip = ol.tinyfacematch_ip_0
```

or auto-search `ol.ip_dict` for `"tinyfacematch"`.

3. Load `weights.bin`:

```python
weights = np.fromfile("weights.bin", dtype=np.int16)
```

4. Allocate physically contiguous PYNQ buffers:

```python
image_buf   = allocate((112 * 112 * 3,), dtype=np.uint8)
weights_buf = allocate((129344,), dtype=np.int16)
scratch0    = allocate((64 * 56 * 56,), dtype=np.int16)
scratch1    = allocate((64 * 56 * 56,), dtype=np.int16)
scratch2    = allocate((64 * 56 * 56,), dtype=np.int16)
out_buf     = allocate((128,), dtype=np.int16)
```

5. Write physical addresses into IP registers:

```python
write_u64(ip, 0x10, image_buf.physical_address)
write_u64(ip, 0x1c, weights_buf.physical_address)
write_u64(ip, 0x28, scratch0.physical_address)
write_u64(ip, 0x34, scratch1.physical_address)
write_u64(ip, 0x40, scratch2.physical_address)
write_u64(ip, 0x4c, out_buf.physical_address)
```

6. Run inference:

```python
image_buf[:] = image_flat
image_buf.flush()
out_buf.flush()

ip.write(0x00, 0x01)
while (ip.read(0x00) & 0x2) == 0:
    pass

out_buf.invalidate()
embedding = np.array(out_buf, dtype=np.float32) / 16384.0
embedding = embedding / np.linalg.norm(embedding)
```

The IP output is already Q14 normalized, but the Python side normalizes again for safety before cosine comparison.

## 15. Camera and Face Verification Flow

The camera/demo flow uses OpenCV.

For DroidCam, the stream URL was:

```text
http://192.168.0.114:4747/video
```

The camera path uses:

```text
threaded latest-frame grabber
Haar face detection on downscaled frames
face crop + resize to 112x112
BGR -> RGB conversion
HLS embedding inference
cosine similarity match
```

The face crop is converted for HLS using:

```python
rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_AREA)
image_flat = np.ascontiguousarray(rgb.astype(np.uint8).reshape(-1))
```

The notebook flow used throttling:

```text
DETECTION_SCALE = 0.25
DETECTION_EVERY = 2
HDMI_INTERVAL / loop interval limiting
```

This keeps camera capture responsive and avoids running face detection every raw frame.

When no face is found, the system can optionally run a dummy cosine comparison using orthogonal vectors:

```python
dummy_a[0] = 1.0
dummy_b[1] = 1.0
```

The dummy cosine should be approximately 0 and below the match threshold.

## 16. Enrollment and Matching

Enrollment collects several embeddings of an authorized face and averages them:

```text
capture face crops
run HLS embedding
average embeddings
L2 normalize average
save as .npz or .npy
```

Verification computes:

```text
score = dot(live_embedding, authorized_embedding)
```

Since embeddings are normalized, dot product equals cosine similarity.

The user also added a separate cosine similarity IP block in the Vivado design. The Python matching loop can use either:

```python
score = float(np.dot(cam_emb, auth_emb))
```

or the PL cosine IP:

```python
match, score = cs.compare(
    cam_emb.astype(np.float32),
    auth_emb.astype(np.float32),
    threshold=MATCH_THRESHOLD,
)
```

The CNN embedding IP and cosine IP are independent accelerators. The CNN produces a 128-D embedding; the cosine IP compares two 128-D float vectors.

## 17. Hardware Benchmark Results

The final measured benchmark on `hvCNN_INT.bit` was:

```text
Loading overlay: hvCNN_INT.bit
Using TinyFaceMatch IP: tinyfacematch_ip_0
weights words: 129344

CNN calc only / HLS IP ap_start to ap_done
  runs: 20
  min:  1287.00 ms
  mean: 1287.28 ms
  p50:  1287.10 ms
  p95:  1287.45 ms
  max:  1290.65 ms

Embedding calc / buffers + HLS CNN + output normalize
  runs: 20
  min:  1288.31 ms
  mean: 1288.66 ms
  p50:  1288.47 ms
  p95:  1288.85 ms
  max:  1292.07 ms

embedding norm = 1.0
```

Summary:

```text
CNN-only mean:       1287.28 ms
Full embedding mean: 1288.66 ms
PS overhead:         ~1.38 ms
Throughput:          ~0.78 embeddings/sec
```

The low min/max spread shows the accelerator timing is stable.

## 18. Final HLS Resource Summary

Final `hls_ddr_intmac` HLS estimate:

```text
BRAM_18K = 10 / 280   = 3%
DSP      = 50 / 220   = 22%
FF       = 16709 / 106400 = 15%
LUT      = 27004 / 53200  = 50%
URAM     = 0
```

Compared against previous HLS versions:

```text
Original static-buffer HLS:
  too large; Vivado placement failed

hls_ddr:
  BRAM_18K=7
  DSP=21
  FF=20523
  LUT=39350

hls_ddr_fast:
  BRAM_18K=10
  DSP=26
  FF=20116
  LUT=38857

hls_ddr_intmac:
  BRAM_18K=10
  DSP=50
  FF=16709
  LUT=27004
```

The final version uses more DSPs but far fewer LUTs, which is exactly the right tradeoff for this board.

## 19. Why IntMAC Improved Speed and Accuracy

The earlier fixed-point HLS kernels did this in the inner loop:

```text
int16 activation -> ap_fixed
int16 weight -> ap_fixed
fixed-point multiply
fixed-point accumulate
fixed-point round/saturate
convert back to int16
```

That created a lot of LUT-heavy conversion and control logic.

The IntMAC kernels do:

```text
int16 activation * int16 weight -> int48 accumulator
round/shift once
saturate once
store int16
```

This maps better to FPGA DSP slices. It also matches the intended quantized math more directly. That is why the CSIM error improved:

```text
hls_ddr / hls_ddr_fast:
  max_abs_error_int16 around 83

hls_ddr_intmac:
  max_abs_error_int16 = 16
```

## 20. Practical End-to-End Flow

The full workflow is:

1. Train the PyTorch model:

```sh
python -m tinyfacematch_pytorch.train \
  --data-dir CASIA_subset_500x25 \
  --output-dir runs/tinyfacematch128_w05_v4_continue \
  --width-mult 0.5 \
  --augment
```

2. Prepare HLS weights:

```sh
python -m tinyfacematch_pytorch.prepare_hls_weights \
  --checkpoint runs/tinyfacematch128_w05_v4_continue/last.pt \
  --output-dir runs/tinyfacematch128_w05_v4_continue/hls_weights_q12
```

3. Copy/update generated `weights_offsets.hpp` into the HLS folder if needed.

4. Run HLS CSIM:

```sh
cd hls_ddr_intmac
vitis-run --mode hls --tcl run_hls_csim.tcl
```

5. Package HLS IP:

```sh
vitis-run --mode hls --tcl package_ip_no_csim.tcl
```

6. Add packaged IP to Vivado:

```text
hls_ddr_intmac/tinyfacematch_hls_ddr_intmac/solution1/impl/ip
```

7. Build bitstream.

8. On PYNQ, copy:

```text
.bit
.hwh
weights.bin
```

9. Run the notebook/Python flow to allocate buffers, write physical addresses, start the IP, and read embeddings.

10. Enroll authorized embeddings and compare live embeddings using cosine similarity.

## 21. Current Final Recommendation

Use:

```text
hls_ddr_intmac
```

as the main CNN IP. Keep the older versions as fallbacks, but the final IntMAC design is currently the best choice because it has:

```text
best board latency
best CSIM accuracy
much lower LUT utilization
stable PYNQ runtime behavior
same register map as previous DDR versions
```

The next meaningful improvements would likely come from architectural changes, not small loop tweaks:

```text
tiling/streaming feature maps to reduce DDR traffic
more parallel pointwise convolution if post-route utilization allows
moving dense/L2 fully to integer arithmetic
using multiple AXI memory bundles/ports for scratch buffers
raising PL clock if timing still closes
```

For the current board and project constraints, the IntMAC DDR-scratch design is the cleanest working balance between resource usage, accuracy, and speed.
