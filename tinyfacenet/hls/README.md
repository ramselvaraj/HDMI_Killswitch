# TinyFaceMatch128 DDR IntMAC HLS IP

Integer-MAC latency fork of the working `hls_ddr_fast/` IP.

The original IP keeps many full activation tensors as static on-chip arrays. That
passed C simulation but over-utilized Zynq-7020 resources. The DDR version stores
intermediate activations as `int16` Q6.10 tensors in external DDR scratch
buffers. This fork keeps that viable memory layout, but runs the convolution,
depthwise, and pointwise hot loops as integer MACs instead of repeatedly
dequantizing activations and weights into fixed-point objects.

## Interface

```text
image_rgb:      112 * 112 * 3 uint8 HWC RGB input
weights:        weights.bin interpreted as int16 words
scratch0:       int16 scratch buffer, at least 64 * 56 * 56 words
scratch1:       int16 scratch buffer, at least 64 * 56 * 56 words
scratch2:       int16 scratch buffer, at least 64 * 56 * 56 words
embedding_out:  128 int16 values, Q*.14 normalized embedding
```

The extra `scratch2` buffer is required so residual blocks can preserve the
identity tensor while computing expand/depthwise/project intermediates.

## Quick C++ Check

From the repo root:

```sh
g++ -std=c++17 -O2 \
  -I/mnt/OS/vivado_linux/2025.2/Vitis/include \
  hls_ddr_intmac/testbench.cpp hls_ddr_intmac/tinyfacematch_ip.cpp \
  -o /tmp/tinyfacematch_ddr_intmac_csim

/tmp/tinyfacematch_ddr_intmac_csim \
  hls/test_vectors/input_0.bin \
  runs/tinyfacematch128_w05_v4_continue/hls_weights_q12/weights.bin \
  hls/test_vectors/expected_0.bin
```

Expected current result:

```text
max_abs_error_int16=16
mean_abs_error_int16=4.69531
cosine_to_expected=0.999991
PASSED
```

## Vitis HLS

```sh
source /mnt/OS/vivado_linux/2025.2/Vitis/settings64.sh
export LD_LIBRARY_PATH=/mnt/OS/vivado_linux/2025.2/Vitis/lib/lnx64.o/Ubuntu/24:$LD_LIBRARY_PATH
cd /mnt/OS/ramse/hls4ml_projects/tinyFaceNet/hls_ddr_intmac

vitis-run --mode hls --tcl run_hls_csim.tcl
vitis-run --mode hls --tcl package_ip_no_csim.tcl
```

After synthesis, check resources:

```sh
grep -E "BRAM|DSP|FF|LUT|URAM|Total" \
  tinyfacematch_hls_ddr_intmac/solution1/syn/report/tinyfacematch_ip_csynth.rpt
```

Current synthesized estimate:

```text
BRAM_18K=10
DSP=50
FF=16709
LUT=27004
```

Current latency estimate:

```text
top max:        8.924 sec
run_conv1x1:    0.386 sec max
run_depthwise:  0.197 sec max
```

The AXI-Lite pointer offsets match the working DDR IP:

```text
image_rgb=0x10
weights=0x1c
scratch0=0x28
scratch1=0x34
scratch2=0x40
embedding_out=0x4c
```
