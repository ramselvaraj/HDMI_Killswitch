# TinyFaceMatch PYNQ Flask Demo

This folder turns the working notebook flow into a browser demo.

The Flask app runs on the PYNQ board. It owns the camera, HLS CNN IP, cosine IP,
DDR buffers, face detection, enrollment, and match loop. A browser connects to
the board and receives an MJPEG video stream plus JSON status updates.

## What Runs Where

```text
PYNQ board:
  Flask app
  OpenCV DroidCam capture
  latest-frame camera grabber thread
  low-latency MJPEG publisher thread
  Haar face detection
  TinyFaceMatch HLS CNN IP
  cosine similarity IP
  authorized embedding storage

Browser:
  live video
  match/no-match dashboard
  enrollment controls
  threshold controls
  latency/status display
```

## Files

```text
app.py
  Flask routes and background FaceService.

templates/index.html
  Single-page dashboard.

static/app.css
  Dashboard styling.

static/app.js
  Status polling and control actions.
```

## Run

Put the bitstream, weights file, and `cosine_driver.py` where Python can find
them. Then, from this directory on the PYNQ board:

```sh
python3 -m pip install flask

python3 app.py \
  --bitstream hvCNN_q8.bit \
  --weights weights.bin \
  --video http://192.168.0.114:4747/video \
  --threshold 0.5 \
  --display-fps 15 \
  --host 0.0.0.0 \
  --port 5000
```

OpenCV and PYNQ are usually already installed on the board image. Avoid
reinstalling OpenCV unless imports fail.

Open this from your laptop:

```text
http://<pynq-board-ip>:5000
```

Press `Start` in the UI to load the overlay and begin the live loop.

## Useful Environment Variables

The same settings can be supplied with environment variables:

```text
TFN_BITSTREAM
TFN_WEIGHTS
TFN_VIDEO
TFN_AUTH_DIR
TFN_THRESHOLD
TFN_LOOP_FPS
TFN_DISPLAY_FPS
TFN_HOST
TFN_PORT
TFN_IP
TFN_COSINE_IP
TFN_AUTOSTART=1
```

Use `TFN_IP` or `TFN_COSINE_IP` only if auto-detection chooses the wrong IP.

## Endpoints

```text
GET  /
GET  /video_feed
GET  /api/status
POST /api/start
POST /api/stop
POST /api/enroll
POST /api/reload_profiles
POST /api/threshold
```

Enrollment is non-blocking. When you press `Enroll Face`, the background loop
collects the requested number of face embeddings from live detections, averages
them, saves `<name>.npy` in the authorized embeddings directory, and reloads the
profile list.

## Notes

The app keeps the same HLS register map as the notebook:

```text
image_rgb      0x10
weights        0x1c
scratch0       0x28
scratch1       0x34
scratch2       0x40
embedding_out  0x4c
```

The frontend does not touch hardware directly. All hardware access stays inside
the single background service thread.

The camera is read continuously by a small grabber thread so OpenCV/DroidCam
does not build up stale frames while the CNN is busy. The video stream is also
published by a lightweight thread, so the browser gets the newest frame even
when HLS inference takes longer than one display frame.
