# HDMI Killswitch

An FPGA hardware kill switch for HDMI, gated by CNN face authentication, on the
**PYNQ-Z2** (Zynq-7020, `xc7z020clg400-1`).

The board sits inline between a PC and a monitor. HDMI in goes through the PL; a
face-recognition result decides whether pixels pass through or the monitor is
fed black (sync signals stay alive so the monitor keeps lock). The decision
logic lives in the fabric rather than in host software.

```text
PC --HDMI--> [ dvi2rgb -> auth gate / overlay -> rgb2dvi ] --HDMI--> Monitor
                              ^
        camera -> Flask/PYNQ -> tinyFaceNet CNN IP -> 128-d embedding
                              -> cosine-similarity IP vs. enrolled -> auth bit
```

## Demo

Live on the PYNQ-Z2 (2026-05-27): the Flask dashboard scores the camera face with
tinyFaceNet + the cosine IP, and the HDMI output is gated accordingly.
Threshold 65%. Same face scored 0.87 (authorized) and 0.51 (not authorized).

| Authorized | Not authorized |
|---|---|
| ![authorized](media/demo_authorized.jpg) | ![not authorized](media/demo_not_authorized.jpg) |

![web UI](media/web_ui_match.png)

Video: [`media/demo.mp4`](media/demo.mp4) (1080p, 13 MB).

## Repository layout

```text
hdmi/                  video-path iteration chain (each stage builds on the last)
  01_testgen/            colour-bar / white test video generator, timing generator
  02_passthrough_clean/  clean HDMI passthrough, 24-bit pixel gate (handle_auth)
  03_auth_gate/          registered auth gate: CDC-synced, latched at vsync, black on deny
  04_cosine_auth/        gate driven by the cosine-similarity result
system/                final integrated design (hvCNN_INT): HDMI gate + graphics overlay
                       + cosine-similarity AXI IP + tinyFaceNet block design
tinyfacenet/           the face-embedding CNN that feeds the gate
  model/                 PyTorch model, training, ONNX export, HLS weight prep, eval
  hls/                   Vitis HLS integer-MAC IP (hls_ddr_intmac) + test vectors
  host/                  PYNQ Flask demo (camera, enrollment, match loop, dashboard)
                         + cosine_driver.py (MMIO driver for the cosine_sim_axi IP)
  scripts/               test-vector export, on-board verification
  weights/               final trained weights, ONNX, Q12 HLS weights, eval JSONs
  docs/                  system report, deep dive, LaTeX/PDF implementation report
  hw/                    tfn1.xsa  (CNN-only Vivado export)
```

Each `hdmi/NN_*` and `system/` folder holds `rtl/`, `constraints/`, `bd/` (block
design, `.bd` and/or exported Tcl) and `hw/` (`.bit`, `.hwh`, `.xsa`).

## Results (measured)

| Item | Result |
|---|---|
| Clean HDMI passthrough | 1,561 LUT, 2,560 FF, 1 BRAM; WNS **+1.254 ns**, timing met |
| tinyFaceNet, first HLS version | 205,926 LUT: cannot be placed on the 53k-LUT part |
| tinyFaceNet, DDR-scratch version | 39,350 LUT |
| tinyFaceNet, final integer-MAC | **27,004 LUT, 50 DSP, 10 BRAM_18K**; cosine 0.99999 vs. reference |
| Embedding latency on PYNQ | ~1,287 ms (earlier measurement), nearly all in the CNN accelerator |
| Web UI runtime panel (2026-05-27) | CNN 691 ms, Haar detect 523-613 ms, cosine 13 ms, loop 330 ms, **8.3 FPS** |

The XDC files declare the HDMI pixel clock and fabric clock as asynchronous
(`set_clock_groups -asynchronous`). The auth bit is double-flop synchronized into
the pixel domain and latched at vsync to avoid torn frames.

## tinyFaceNet

`TinyFaceMatch128`: 112x112 RGB in, 128-d L2-normalized embedding out. Trained in
PyTorch (ArcFace-style), BatchNorm folded, weights quantized to int16 (Q12), then
implemented in Vitis HLS. The latency work is in `hls/` (the final fork is the
integer-MAC one; see `tinyfacenet/hls/README.md`). Full details in
`tinyfacenet/docs/`.

### Reproduce

```bash
# Train / evaluate (needs CASIA-WebFace, not included; ~3 GB)
cd tinyfacenet/model && pip install -r requirements.txt && python train.py --help

# HLS C-sim (adjust Vitis path), then package the IP
cd tinyfacenet/hls
vitis-run --mode hls --tcl run_hls_csim.tcl
vitis-run --mode hls --tcl package_ip_no_csim.tcl

# On the PYNQ board
cd tinyfacenet/host && pip install -r requirements.txt && python app.py
```

Note: the scripts and READMEs carried over from the original work contain
absolute paths from the author's machine (e.g. `/mnt/OS/...`, `runs/...`);
adjust them to this layout (`tinyfacenet/weights/hls_weights_q12/weights.bin`).
`host/app.py` defaults to the bitstream name `hvCNN_q8.bit`; the final design
is `system/hw/hvCNN_INT.bit` (+ `.hwh`), so pass or set that.

## Software drivers (MMIO)

- `tinyfacenet/host/cosine_driver.py`: Python class `CosineSimilarity` for the
  `cosine_sim_axi` IP (register map in its docstring: CTRL, STATUS, THRESHOLD in
  Q2.30, vector select/data in Q1.15).
- `tinyfacenet/host/app.py`: drives the tinyFaceNet HLS IP directly over MMIO via
  PYNQ (`ip.write`/`ip.read` on the CTRL/image/weights/scratch/out pointer
  registers) and DMA buffers from `pynq.allocate`.
- The HDMI auth bit itself is set from software over a GPIO/AXI register in the
  block designs under `hdmi/03_auth_gate` and `system/`.

## What is committed vs. not

Committed: hand-written RTL, constraints, block designs, bitstreams/`.xsa`/`.hwh`
for the final pipeline, HLS sources, model code, final weights, docs.
Not committed: Vivado project directories (`*.runs`, `*.cache`, ...), the
CASIA-WebFace dataset, older training runs, and the earlier near-duplicate
Vivado project copies.

## Status / known gaps

Ran end to end on the PYNQ-Z2 with a real monitor (see Demo). Per-frame cost is
now dominated by the CNN (~0.7 s) and the CPU-side Haar face detector (~0.5-0.6 s),
which is also the weakest part for accuracy (off-angle / low-light faces).
