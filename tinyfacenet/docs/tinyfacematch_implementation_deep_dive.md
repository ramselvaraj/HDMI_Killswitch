# TinyFaceMatch Implementation Deep Dive

This document explains the implementation from a technical and architectural point of view. It is not a command reference. The goal is to make the system understandable as a design: what each part is responsible for, how data moves between parts, why the HLS implementation is written the way it is, and what tradeoffs led to the final version.

## 1. High-Level Mental Model

The project is a face verification system built around a compact CNN embedding model.

At a high level, the system does this:

```text
camera/image
  -> face crop
  -> 112x112 RGB image
  -> CNN embedding accelerator on FPGA
  -> 128-D normalized embedding
  -> cosine similarity against enrolled embeddings
  -> match / no match
```

The important separation is:

```text
PyTorch side:
  train the model
  export/fold/quantize weights
  validate that quantized math is close enough

HLS / FPGA side:
  implement only inference
  consume a 112x112 RGB image
  consume packed quantized weights
  produce a 128-D embedding

PS / Python side:
  load the bitstream
  allocate DDR buffers
  write physical addresses into the IP
  feed images
  read embeddings
  handle camera, enrollment, and matching
```

The final FPGA CNN IP does not know anything about cameras, identities, file formats, face detection, or authorization. It is deliberately just:

```text
image + weights + scratch buffers -> embedding
```

Everything user-facing happens on the processor side.

## 2. Main Files and Their Roles

The implementation is split into a few layers.

### 2.1 PyTorch Model and Training

```text
tinyfacematch_pytorch/model.py
```

Defines the neural network architecture. This is the source of truth for the CNN topology.

Important classes:

```text
TinyFaceMatch128
  inference backbone
  outputs a 128-D normalized embedding

TinyFaceMatchTrainingModel
  wraps the backbone with an ArcFace classifier
  used only during training

ArcFaceClassifier
  training-only classification head
  not exported to FPGA

ConvBnPReLU
  Conv2d + BatchNorm2d + PReLU

ConvBn
  Conv2d + BatchNorm2d

DepthwiseSeparableBlock
  depthwise 3x3 + pointwise 1x1

ResidualBottleneckBlock
  1x1 expand + depthwise 3x3 + 1x1 project
  optional residual connection

GlobalDepthwiseConv
  final 7x7 depthwise convolution
```

```text
tinyfacematch_pytorch/data.py
```

Defines the dataset loader and image preprocessing.

```text
tinyfacematch_pytorch/train.py
```

Defines the training loop, optimizer, validation loop, checkpoint saving, and metrics logging.

### 2.2 Weight Preparation

```text
tinyfacematch_pytorch/prepare_hls_weights.py
```

Converts a trained PyTorch checkpoint into a format that the HLS IP can consume. This file is one of the most important bridges in the project.

It:

```text
loads the trained checkpoint
rebuilds the TinyFaceMatch128 backbone
folds BatchNorm into Conv/Linear layers
quantizes weights, biases, and PReLU values
packs everything into weights.bin
generates weights_offsets.hpp
generates manifest.json
```

```text
tinyfacematch_pytorch/eval_hls_prepared.py
```

Builds a PyTorch-side reference model using the prepared quantized weights. It exists to answer: "If the FPGA computes using these folded/quantized weights, does it still behave like the trained model?"

### 2.3 HLS Implementations

```text
hls/
```

Original direct HLS implementation. It was useful for correctness but not viable for the FPGA because it used too much on-chip storage.

```text
hls_ddr/
```

First viable DDR-backed design. It moved large intermediate tensors into external DDR scratch buffers.

```text
hls_ddr_fast/
```

Improved DDR version with weight caching and a specialized 1x1 convolution path.

```text
hls_ddr_intmac/
```

Final best version. It keeps the DDR scratch architecture but rewrites the convolution hot loops using integer MAC arithmetic. This is the current recommended implementation.

Important files inside `hls_ddr_intmac/`:

```text
tinyfacematch_ip.hpp
  public top-function declaration
  constants for image size, scratch size, embedding size
  fixed-point typedefs used by the dense/L2 tail

tinyfacematch_ip.cpp
  actual HLS implementation
  top-level AXI interfaces
  layer execution order
  integer convolution kernels
  dense/L2 output stage

weights_offsets.hpp
  generated constants describing the packed weights layout
  quantization frac-bit definitions
  byte offsets for every layer

testbench.cpp
  C simulation testbench
  loads input vector, weights.bin, expected output
  checks max error, mean error, cosine similarity
```

### 2.4 Processor-Side Runtime

```text
scripts/pynq_hls_camera_verify.py
```

Standalone PYNQ/OpenCV script showing how the PS side uses the IP. It is not the only way to run the accelerator, but it captures the important runtime pattern:

```text
load overlay
resolve IP
allocate buffers
write physical addresses
prepare image input
start IP
wait for completion
read output embedding
compare embedding against profile
```

Notebook cells used during development follow the same logic.

## 3. Model Architecture in Implementation Terms

The model is intentionally small because it has to fit on a Zynq-7020 class FPGA.

The inference backbone is `TinyFaceMatch128`.

With `width_mult=0.5`, the effective channel sizes are:

```text
c32  -> 16
c64  -> 32
c128 -> 64
c256 -> 128
```

The model begins with a stride-2 convolution:

```text
input:  112 x 112 x 3
conv0:   56 x  56 x 16
```

Then it uses a depthwise-separable block:

```text
depthwise 3x3, per-channel spatial filtering
pointwise 1x1, channel mixing
```

After that, the backbone is mostly residual bottleneck blocks. Each bottleneck has:

```text
expand:   1x1 conv, increases channels
dw:       3x3 depthwise conv
project:  1x1 conv, reduces channels
residual: optional add back to identity
```

The residual is used only when:

```text
stride == 1 and input_channels == output_channels
```

This is important for the HLS implementation because residual blocks require preserving the input tensor while computing the branch. That is why the HLS design has a third scratch buffer.

The final part is:

```text
conv_final: 1x1 conv to 128 channels
global_dw7: 7x7 depthwise conv over final 7x7 map
dense: 128 -> 128 linear projection
L2 normalize
```

The output is always a 128-D normalized vector.

## 4. Training Design

Training and inference are intentionally different.

During training, the model is:

```text
TinyFaceMatch128 backbone
ArcFace classifier head
```

The backbone learns embeddings. The ArcFace head turns embeddings into classification logits with angular margin. That margin makes same-identity embeddings cluster together and different-identity embeddings separate angularly.

The FPGA does not need the classifier because deployment does not classify among training identities. Deployment compares embeddings using cosine similarity.

So the export path only uses:

```text
checkpoint["backbone_state_dict"]
```

and ignores:

```text
classifier_state_dict
optimizer_state_dict
scheduler_state_dict
```

The image preprocessing during training is:

```text
RGB conversion
resize to 112x112
optional augmentation
normalize: (pixel - 127.5) / 128.0
CHW tensor layout
```

This matters because the FPGA input path must reproduce the same normalization. In the final HLS IP, the normalization is implemented in integer Q-format inside `run_conv_image()`.

## 5. Why BatchNorm Folding Is Necessary

The PyTorch model contains Conv-BN-PReLU and Conv-BN blocks.

In software, a block like this is:

```text
conv output
  -> batchnorm
  -> optional PReLU
```

On the FPGA, implementing separate BatchNorm layers would add unnecessary memory passes and logic. BatchNorm is a linear transform during inference, so it can be folded into the previous convolution.

For each Conv-BN pair:

```text
scale = gamma / sqrt(running_var + eps)

folded_weight = conv_weight * scale
folded_bias   = beta + (conv_bias - running_mean) * scale
```

Because the PyTorch convolutions usually have `bias=False`, the `conv_bias` term is effectively zero.

After folding, the HLS layer only needs:

```text
convolution weights
one bias per output channel
optional PReLU alpha
```

This is why the packed HLS weights contain only folded weights, folded biases, and PReLU values. There is no explicit BatchNorm in the HLS code.

## 6. Weight Packing and Quantization

The HLS IP reads one flat `weights.bin` array from DDR.

The packing script produces:

```text
weights.bin
weights_offsets.hpp
manifest.json
weights_quantized.npz
weights_folded_fp32.npz
```

The important runtime file is:

```text
weights.bin
```

The important compile-time file is:

```text
weights_offsets.hpp
```

The model uses these quantization formats:

```text
weights:      int16 with 12 fractional bits
biases:       int32 with 20 fractional bits
PReLU alpha:  int16 with 12 fractional bits
activations:  int16 with 10 fractional bits
output:       int16 with 14 fractional bits
```

In other words:

```text
weight_float approx weight_int / 2^12
bias_float   approx bias_int   / 2^20
act_float    approx act_int    / 2^10
out_float    approx out_int    / 2^14
```

The packed file is ordered layer-by-layer in inference order. For each layer:

```text
write quantized weights
write quantized bias
write PReLU alpha if the layer has PReLU
```

Biases are int32. Since the HLS top pointer is `ap_int<16>* weights`, the HLS reconstructs each int32 bias from two consecutive int16 words:

```cpp
static ap_int<32> read_i32_words(const ap_int<16>* words, int index) {
    ap_uint<16> lo = (ap_uint<16>)words[index];
    ap_uint<16> hi = (ap_uint<16>)words[index + 1];
    ap_uint<32> combined = ((ap_uint<32>)hi << 16) | lo;
    return (ap_int<32>)combined;
}
```

The offsets header tells HLS where each layer begins:

```cpp
TFM_B4_EXPAND_OFFSET_BYTES
TFM_B4_DW_OFFSET_BYTES
TFM_B4_PROJECT_OFFSET_BYTES
...
```

The HLS code divides by 2 when passing offsets:

```cpp
TFM_B4_EXPAND_OFFSET_BYTES / 2
```

because the pointer is indexed in 16-bit words, not bytes.

## 7. Why the Final HLS Uses DDR Scratch Buffers

The first intuitive HLS implementation kept intermediate tensors as arrays inside the IP. That is attractive because on-chip access is fast, but it was impossible to place on the Zynq-7020.

The feature maps are large. For example:

```text
64 x 56 x 56 int16 = 200704 words
```

One such tensor is already about:

```text
200704 * 2 bytes = about 401 KB
```

Several of those on-chip quickly exceed available BRAM/LUT resources.

So the viable architecture changed to:

```text
store intermediate activations in external DDR
reuse a small number of DDR scratch buffers
make compute kernels resource-shared
```

The final IP uses:

```text
scratch0
scratch1
scratch2
```

Each scratch buffer is large enough for the largest intermediate activation tensor:

```text
64 * 56 * 56 int16 words
```

The third scratch buffer is needed for residual blocks. During a residual block:

```text
scratch0 may hold the identity input
scratch1 may hold expand output
scratch2 may hold depthwise output
project output is written back to scratch0 with residual add
```

Without the third buffer, the project path could overwrite the identity before the residual add.

## 8. HLS Top-Level Interface

The final top function is:

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

Conceptually:

```text
image_rgb:
  input image in HWC RGB uint8
  size = 112 * 112 * 3

weights:
  flat packed weights.bin interpreted as int16 words

scratch0, scratch1, scratch2:
  DDR activation workspaces
  int16 Q6.10

embedding_out:
  output vector
  128 int16 Q14 values
```

The HLS pragmas expose these as AXI master ports:

```text
image_rgb      -> m_axi gmem0
weights        -> m_axi gmem1
scratch/output -> m_axi gmem2
```

Control and pointer registers use AXI-Lite.

The HLS IP does not allocate memory. It only receives physical addresses. This is why the PS/PYNQ code must allocate buffers first and then write the buffer addresses into the IP control registers.

## 9. Register Map

The register map is generated by HLS and remained stable across the DDR versions:

```text
0x00  control register
0x10  image_rgb pointer low
0x14  image_rgb pointer high
0x1c  weights pointer low
0x20  weights pointer high
0x28  scratch0 pointer low
0x2c  scratch0 pointer high
0x34  scratch1 pointer low
0x38  scratch1 pointer high
0x40  scratch2 pointer low
0x44  scratch2 pointer high
0x4c  embedding_out pointer low
0x50  embedding_out pointer high
```

The control register uses the standard `ap_ctrl_hs` protocol:

```text
bit 0: ap_start
bit 1: ap_done
bit 2: ap_idle
bit 3: ap_ready
```

The PS starts the IP by writing:

```text
control = 0x01
```

and waits until:

```text
control & 0x2 != 0
```

## 10. Final HLS Layer Flow

The final HLS does not implement a dynamic graph interpreter. It is a hand-written static schedule.

The top function explicitly calls every layer in order:

```text
conv0
dwsep0 depthwise
dwsep0 pointwise
b0 expand
b0 depthwise
b0 project
...
conv_final
global_dw7
dense_l2
```

This is deliberate. A static schedule avoids the overhead and complexity of describing layers as runtime metadata. The model topology is fixed, so hard-coding the layer calls gives HLS more compile-time structure.

The scratch buffer flow is also manually scheduled. For example:

```text
run_conv_image(image_rgb, scratch0, ...)
run_depthwise(scratch0, scratch1, ...)
run_conv1x1(scratch1, scratch0, ...)
```

This means:

```text
scratch0 holds conv0 output
scratch1 holds depthwise output
scratch0 is reused for pointwise output
```

Residual blocks use `scratch0` as the identity when appropriate:

```text
expand:    scratch0 -> scratch1
depthwise: scratch1 -> scratch2
project:   scratch2 -> scratch0, adding residual from old scratch0
```

That pattern is why the project layer accepts both:

```text
input tensor
output tensor
residual tensor
```

## 11. Final HLS Kernels

The final design has four meaningful compute kernels.

### 11.1 `run_conv_image`

Purpose:

```text
first 3x3 stride-2 convolution from RGB input
```

Input:

```text
uint8 image_rgb[112 * 112 * 3]
```

Output:

```text
int16 Q6.10 tensor: 16 x 56 x 56
```

Special behavior:

```text
converts image pixels to Q6.10 internally
handles zero-padding bounds checks
applies PReLU
```

The pixel conversion is:

```cpp
pixel_q10 = pixel * 8 - 1020
```

Why this works:

```text
training normalization = (pixel - 127.5) / 128.0
Q10 scale = multiply by 1024

(pixel - 127.5) / 128 * 1024
= (pixel - 127.5) * 8
= pixel * 8 - 1020
```

So this integer formula directly produces the Q6.10 activation value.

### 11.2 `run_conv1x1`

Purpose:

```text
all pointwise 1x1 convolutions
```

This includes:

```text
dwsep0 pointwise
bottleneck expand layers
bottleneck project layers
conv_final
```

This is one of the most important kernels because 1x1 convolution does most of the channel mixing and contributes heavily to compute cost.

Implementation details:

```text
process output channels in pairs
cache weights for output channel 0
cache weights for output channel 1
for each spatial location:
  read one input activation
  update acc0 and acc1
round/shift/saturate
apply PReLU if needed
add residual if needed
write output
```

Processing two output channels together reuses the same input activation load for two output accumulators. This spends more DSPs but improves throughput and keeps LUTs reasonable.

### 11.3 `run_depthwise`

Purpose:

```text
3x3 depthwise convolutions
7x7 global depthwise convolution
```

Depthwise convolution is different from normal convolution because each channel is filtered independently:

```text
output[c, y, x] depends only on input[c, :, :]
```

Implementation:

```text
for each channel:
  cache that channel's kernel
  read folded bias
  read optional PReLU
  scan output spatial grid
  accumulate k x k window
  round/shift/saturate
  apply PReLU if present
  write output
```

The same function handles both:

```text
k=3, pad=1, stride=1 or 2
k=7, pad=0, stride=1
```

### 11.4 `run_dense_l2`

Purpose:

```text
final dense projection and L2 normalization
```

The dense input is 128 values after global depthwise. It computes:

```text
dense_out[o] = bias[o] + sum_i input[i] * weight[o, i]
```

Then:

```text
norm = sqrt(sum(dense_out[i]^2))
embedding[i] = dense_out[i] / norm
```

The output is quantized to Q14:

```text
embedding_out[i] = round(embedding[i] * 2^14)
```

This stage still uses fixed-point helper types because it is not the dominant runtime bottleneck. The main performance issue was convolution, not the final 128x128 dense layer.

## 12. Integer MAC Arithmetic

The final HLS version is called IntMAC because convolution arithmetic is done as integer multiply-accumulate.

The core format relationship is:

```text
activation: Q6.10
weight:     Q*.12
bias:       Q*.20
```

When multiplying:

```text
activation_q10 * weight_q12 = product_q22
```

The accumulator is therefore Q22.

Bias is Q20, so before adding it to the accumulator:

```text
bias_q22 = bias_q20 << 2
```

After accumulation, the desired activation output is Q10. Since the accumulator is Q22:

```text
out_q10 = round(acc_q22 >> 12)
```

This is implemented by:

```cpp
round_shift_sat_i16(acc, TFM_WEIGHT_FRAC_BITS)
```

because `TFM_WEIGHT_FRAC_BITS = 12`.

PReLU uses:

```text
negative_value_q10 * alpha_q12 = q22
round shift by 12 -> q10
```

Residual addition is:

```text
out_q10 + residual_q10
```

with int16 saturation.

This arithmetic is simple, explicit, and maps well to FPGA DSP blocks.

## 13. Why IntMAC Is Better Than the Fixed-Point Version

The previous HLS versions represented activations and weights as `ap_fixed` values inside the hot loops.

That is readable, but HLS has to synthesize lots of helper logic:

```text
sign extension
fraction alignment
fixed-point rounding
fixed-point saturation
type conversion
muxing for ap_fixed operators
```

Those operations are not free. They consume LUTs and can slow scheduling.

The IntMAC version keeps the hot loop closer to what the hardware should do:

```text
read int16 activation
read int16 weight
multiply
accumulate in int48
shift once
saturate once
write int16 activation
```

The final resource tradeoff proves the point:

```text
hls_ddr_fast:
  DSP = 26
  LUT = 38857

hls_ddr_intmac:
  DSP = 50
  LUT = 27004
```

DSP usage increased because the arithmetic is now more directly mapped to DSPs. LUT usage dropped because the design no longer builds so much fixed-point conversion logic.

This is the right tradeoff for the target FPGA because LUT utilization was the placement risk.

Accuracy also improved:

```text
hls_ddr_fast:
  max_abs_error_int16 = 83
  mean_abs_error_int16 = 25.7734
  cosine_to_expected = 0.999745

hls_ddr_intmac:
  max_abs_error_int16 = 16
  mean_abs_error_int16 = 4.69531
  cosine_to_expected = 0.999991
```

The integer implementation better matches the intended quantized arithmetic: accumulate in a high precision integer domain, then round once.

## 14. Processor-Side Runtime Design

The PS side is responsible for memory management and orchestration.

The HLS IP is not given Python arrays directly. It receives physical DDR addresses.

The PS flow is:

```text
load overlay
find tinyfacematch_ip_0
load weights.bin as int16
allocate physically contiguous buffers
copy weights into weights buffer
flush caches
write physical addresses into HLS registers
for each inference:
  copy image into image buffer
  flush image buffer
  clear/flush output buffer
  write ap_start
  poll ap_done
  invalidate output buffer
  convert Q14 output to float
  normalize embedding
```

The reason for explicit flush/invalidate is cache coherency. The PS CPU cache and the PL accelerator both access DDR. Before PL reads a buffer, PS writes must be flushed to DDR. After PL writes output, PS must invalidate its cached view before reading.

The weights buffer is written once and reused. The scratch buffers are reused every inference. The image and output buffers are updated every inference.

## 15. Camera and Matching Pipeline

The camera pipeline is intentionally outside the HLS IP. This keeps the FPGA accelerator focused on CNN inference only.

The runtime camera flow is:

```text
DroidCam/OpenCV capture
latest-frame thread
periodic Haar face detection
face crop
resize to 112x112
BGR to RGB
flatten HWC image
HLS CNN inference
cosine similarity
match/no match decision
```

Face detection is throttled because it is CPU-side and does not need to run on every frame. A typical notebook setup used:

```text
DETECTION_SCALE = 0.25
DETECTION_EVERY = 2
```

The CNN inference itself is much slower than a video frame, so the live system naturally operates as periodic verification rather than high-FPS per-frame inference.

Enrollment works by averaging multiple embeddings:

```text
collect N embeddings of authorized user
average them
normalize the average
store as authorized embedding
```

Verification compares live embedding to stored embeddings:

```text
score = dot(live_embedding, authorized_embedding)
```

Because both vectors are normalized, dot product equals cosine similarity.

## 16. Cosine Similarity Accelerator

A separate cosine similarity IP can be added to the Vivado block design.

This block is not part of the CNN HLS IP. It consumes two 128-D embeddings and returns:

```text
match flag
cosine score
```

The Python code can switch from software cosine:

```python
score = float(np.dot(cam_emb, auth_emb))
```

to PL cosine:

```python
match, score = cs.compare(cam_emb, auth_emb, threshold=MATCH_THRESHOLD)
```

This is useful for keeping the matching operation in PL, but the measured CNN latency is still dominated by the TinyFaceMatch IP.

## 17. Hardware Results and Interpretation

The final benchmark measured:

```text
CNN-only mean:       1287.28 ms
Full embedding mean: 1288.66 ms
```

This means:

```text
PS overhead is about 1.38 ms
CNN IP accounts for about 99.89% of embedding latency
```

Therefore, further system-level speedups must mostly target the CNN IP, not Python overhead.

The final HLS resource estimate is:

```text
BRAM_18K = 10
DSP      = 50
FF       = 16709
LUT      = 27004
```

This is much healthier than the earlier versions. In particular, LUT usage is around 50% in HLS estimate, which gives Vivado more room for routing and integration than the `hls_ddr_fast` version.

## 18. Implementation Thought Process

The final implementation was reached through a sequence of constraints and corrections.

### 18.1 Start With Correctness

The first goal was to make the HLS output match the expected quantized embedding. The original implementation had fixed-point saturation problems during pixel normalization and dequantization. Those were fixed first because resource optimization is meaningless if the math is wrong.

### 18.2 Make It Fit

The original static-buffer design was too large. The obvious reason was that full intermediate tensors were being stored on-chip.

The key architectural change was:

```text
move activations to DDR scratch buffers
reuse a small number of buffers
keep only small per-channel caches on chip
```

This made the design placeable.

### 18.3 Improve Latency Without Breaking Fit

The first latency attempt was conservative:

```text
cache weights
specialize 1x1 convolutions
compute two output channels together
```

This improved estimates but did not fully solve board latency. It also showed that the design still carried a lot of fixed-point overhead.

### 18.4 Replace Fixed-Point Hot Loops With Integer MACs

The final insight was that the quantized model is naturally integer math:

```text
int16 activation
int16 weight
int32 bias
int48 accumulator
```

Instead of dequantizing every activation and weight to `ap_fixed`, the final HLS does the convolution in integer Q-format.

This produced the best combined result:

```text
better accuracy
lower LUT usage
faster board latency
still viable DSP usage
same external interface
```

### 18.5 Keep the External Interface Stable

All DDR versions kept the same top-level pointer interface and AXI-Lite offsets. This was intentional because it allowed the Python/PYNQ driver and Vivado block design to stay mostly unchanged while the internals evolved.

The implementation changed under the hood, but the PS contract remained:

```text
write image pointer
write weights pointer
write scratch pointers
write output pointer
start IP
read output
```

## 19. Known Limitations

The final design is good, but it is still a DDR-scratch architecture.

That means every layer writes an activation tensor to DDR and the next layer reads it back. This is much more feasible than storing everything on chip, but it costs latency.

The design is also mostly serial at the layer level:

```text
one layer runs
then next layer runs
then next layer runs
```

There is no full streaming pipeline across layers.

The CNN is therefore not high-FPS. It is suitable for periodic face verification, not real-time dense video inference.

## 20. Possible Future Improvements

The next improvements would require deeper architectural changes.

Most promising:

```text
tile feature maps to reduce DDR traffic
stream rows between depthwise and pointwise where possible
give scratch buffers separate AXI bundles/ports
increase pointwise parallelism if post-route utilization allows
convert dense/L2 tail to integer arithmetic too
raise PL clock if timing closes
```

Less promising:

```text
small Python optimizations
more logging cleanup
minor loop syntax changes
```

The hardware benchmark shows Python overhead is tiny compared with CNN time, so further meaningful speedups need to come from the accelerator architecture.

## 21. Current Final State

The implementation to use is:

```text
hls_ddr_intmac/
```

It is the best version because:

```text
it fits comfortably by HLS estimate
it passed Vitis CSIM
it is more accurate than earlier HLS versions
it is much faster on board
it keeps the same PYNQ-facing interface
it reduces LUT pressure significantly
```

The final system is best understood as:

```text
PyTorch trains an embedding model.
Weight preparation folds and quantizes that model.
The HLS IP implements the fixed inference graph using DDR scratch buffers.
The final HLS kernels compute convolutions using integer Q-format MACs.
PYNQ orchestrates memory, camera input, embedding extraction, and cosine matching.
```

That is the core implementation.
