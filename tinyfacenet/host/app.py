from __future__ import annotations

import argparse
import asyncio
import glob
import logging
import os
import threading
import time
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

try:
    from pynq import Overlay, allocate
except ImportError as exc:  # pragma: no cover - this app is meant to run on PYNQ.
    Overlay = None
    allocate = None
    PYNQ_IMPORT_ERROR = exc
else:
    PYNQ_IMPORT_ERROR = None

try:
    from cosine_driver import CosineSimilarity
except ImportError as exc:  # pragma: no cover - board/project dependency.
    CosineSimilarity = None
    COSINE_IMPORT_ERROR = exc
else:
    COSINE_IMPORT_ERROR = None


SCRATCH_WORDS = 64 * 56 * 56
EMBED_WORDS = 128
OUT_SCALE = float(1 << 14)

CTRL_OFFSET = 0x00
IMAGE_OFFSET = 0x10
WEIGHTS_OFFSET = 0x1C
SCRATCH0_OFFSET = 0x28
SCRATCH1_OFFSET = 0x34
SCRATCH2_OFFSET = 0x40
OUT_OFFSET = 0x4C


log = logging.getLogger("tfn_flask")


@dataclass
class AppConfig:
    bitstream: str = "hvCNN_q8.bit"
    weights: str = "weights.bin"
    video: str = "http://192.168.0.114:4747/video"
    authorized_dir: str = "authorized_embeddings_hls"
    host: str = "0.0.0.0"
    port: int = 5000
    threshold: float = 0.5
    loop_fps: float = 12.0
    detection_scale: float = 0.25
    detection_every: int = 2
    haar_scale_factor: float = 1.15
    haar_min_neighbors: int = 4
    haar_min_size: int = 16
    face_hold_sec: float = 2.0
    max_faces: int = 1
    face_padding: float = 0.25
    jpeg_quality: int = 82
    display_fps: float = 15.0
    crop_zoom: float = 1.0
    crop_x: float = 0.0
    crop_y: float = 0.0
    tinyfacematch_ip: str | None = None
    cosine_ip: str | None = None
    autostart: bool = False


class RollingMetric:
    def __init__(self, maxlen: int = 80) -> None:
        self.values: deque[float] = deque(maxlen=maxlen)

    def add_ms(self, elapsed_ms: float) -> None:
        self.values.append(float(elapsed_ms))

    def mean(self) -> float | None:
        if not self.values:
            return None
        return float(sum(self.values) / len(self.values))


def parse_camera(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def write_u64(ip: Any, offset: int, value: int) -> None:
    ip.write(offset, int(value) & 0xFFFFFFFF)
    ip.write(offset + 4, (int(value) >> 32) & 0xFFFFFFFF)


def normalize_embedding(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x / (np.linalg.norm(x) + 1e-12)


def face_to_hls_input(face_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb.astype(np.uint8).reshape(-1))


def center_crop_frame(frame: np.ndarray, zoom: float, offset_x: float = 0.0, offset_y: float = 0.0) -> np.ndarray:
    zoom = max(1.0, float(zoom))
    if zoom <= 1.0001:
        return np.ascontiguousarray(frame)

    height, width = frame.shape[:2]
    crop_w = max(1, int(round(width / zoom)))
    crop_h = max(1, int(round(height / zoom)))

    offset_x = float(max(-1.0, min(1.0, offset_x)))
    offset_y = float(max(-1.0, min(1.0, offset_y)))
    max_x0 = width - crop_w
    max_y0 = height - crop_h
    x0 = int(round((max_x0 / 2.0) * (1.0 + offset_x)))
    y0 = int(round((max_y0 / 2.0) * (1.0 + offset_y)))
    x0 = max(0, min(max_x0, x0))
    y0 = max(0, min(max_y0, y0))
    return np.ascontiguousarray(frame[y0 : y0 + crop_h, x0 : x0 + crop_w])


def find_cascade() -> str:
    candidates: list[str] = []
    candidates += glob.glob("/usr/share/opencv*/haarcascades/haarcascade_frontalface_default.xml")
    candidates += glob.glob("/usr/local/share/opencv*/haarcascades/haarcascade_frontalface_default.xml")
    candidates += glob.glob("/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
    if hasattr(cv2, "data"):
        candidates.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    candidates += glob.glob(os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"))

    for path in candidates:
        if path and os.path.exists(path):
            return path

    local = "haarcascade_frontalface_default.xml"
    if not os.path.exists(local):
        url = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
        log.info("Downloading Haar cascade XML")
        urllib.request.urlretrieve(url, local)
    return local


def load_authorized_embeddings(directory: str) -> list[np.ndarray]:
    embeddings: list[np.ndarray] = []
    root = Path(directory)
    if not root.exists():
        return embeddings
    for path in sorted(root.glob("*.npy")):
        embeddings.append(normalize_embedding(np.load(str(path)).astype(np.float32)))
    return embeddings


def save_authorized_embedding(name: str, embedding: np.ndarray, directory: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_")
    if not safe_name:
        safe_name = "profile"
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{safe_name}.npy"
    np.save(str(path), normalize_embedding(embedding).astype(np.float32))
    return path


class VideoFrameGrabber(threading.Thread):
    def __init__(self, source: str) -> None:
        super().__init__(daemon=True)
        self.source = source
        self.cap = cv2.VideoCapture(parse_camera(source))
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.running = True

    def is_opened(self) -> bool:
        return bool(self.cap.isOpened())

    def run(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            with self.lock:
                self.frame = frame
        self.cap.release()

    def latest(self) -> np.ndarray | None:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self) -> None:
        self.running = False


class FaceService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

        self.overlay = None
        self.ip = None
        self.cs = None
        self.grabber: VideoFrameGrabber | None = None
        self.publisher: threading.Thread | None = None
        self.detector = None
        self.buffers: list[Any] = []

        self.image_buf = None
        self.weights_buf = None
        self.scratch0 = None
        self.scratch1 = None
        self.scratch2 = None
        self.out_buf = None

        self.auth_embs: list[np.ndarray] = []
        self.last_detections: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
        self.last_detection_time = 0.0
        self.last_result: dict[str, Any] = {"match": False, "score": None, "message": "Starting"}
        self.latest_jpeg: bytes | None = None
        self.latest_frame_time = 0.0
        self.frame_count = 0

        self.metrics = {
            "loop_ms": RollingMetric(),
            "detect_ms": RollingMetric(),
            "hls_ms": RollingMetric(),
            "cosine_ms": RollingMetric(),
            "fps": RollingMetric(),
        }

        self.enroll_request: dict[str, Any] | None = None
        self.status: dict[str, Any] = {
            "running": False,
            "ready": False,
            "state": "idle",
            "message": "Idle",
            "match": False,
            "face_present": False,
            "last_score": None,
            "best_score": None,
            "threshold": config.threshold,
            "authorized_count": 0,
            "frame_count": 0,
            "cnn_ms": None,
            "cosine_ms": None,
            "detect_ms": None,
            "loop_ms": None,
            "fps": None,
            "tinyfacematch_ip": None,
            "cosine_ip": None,
            "bitstream": config.bitstream,
            "weights": config.weights,
            "video": config.video,
            "error": None,
            "enrollment": None,
        }

    def start(self) -> None:
        with self.lock:
            if self.worker and self.worker.is_alive():
                return
            self.stop_event.clear()
            self.worker = threading.Thread(target=self._run_guarded, daemon=True)
            self.worker.start()
            self.status.update({"running": True, "state": "starting", "message": "Starting hardware and camera"})

    def stop(self) -> None:
        self.stop_event.set()
        worker = self.worker
        if worker and worker.is_alive():
            worker.join(timeout=3.0)
        if worker and worker.is_alive():
            with self.lock:
                self.status.update({"running": True, "state": "stopping", "message": "Stop requested; waiting for CNN loop"})
            return
        self._cleanup()
        with self.lock:
            self.status.update({"running": False, "ready": False, "state": "idle", "message": "Stopped"})

    def request_enrollment(self, name: str, samples: int) -> None:
        samples = max(1, min(int(samples), 20))
        with self.lock:
            self.enroll_request = {"name": name or "me", "samples": samples, "embeddings": []}
            self.status["enrollment"] = {"name": name or "me", "captured": 0, "target": samples}
            self.status["message"] = f"Enrollment armed for {name or 'me'}"

    def set_threshold(self, threshold: float) -> None:
        threshold = float(max(-1.0, min(1.0, threshold)))
        with self.lock:
            self.config.threshold = threshold
            self.status["threshold"] = threshold

    def reload_authorized(self) -> int:
        auth = load_authorized_embeddings(self.config.authorized_dir)
        with self.lock:
            self.auth_embs = auth
            self.status["authorized_count"] = len(auth)
        return len(auth)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            result = dict(self.status)
            result["frame_age_ms"] = None
            if self.latest_frame_time:
                result["frame_age_ms"] = (time.time() - self.latest_frame_time) * 1000.0
            for key, metric in self.metrics.items():
                result[key] = metric.mean()
            return result

    def jpeg_frame(self) -> bytes | None:
        with self.lock:
            return self.latest_jpeg

    def _run_guarded(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            self._initialize()
            self._loop()
        except Exception as exc:
            log.exception("FaceService failed")
            with self.lock:
                self.status.update({"state": "error", "message": str(exc), "error": repr(exc), "running": False, "ready": False})
        finally:
            self._cleanup()
            loop.close()

    def _initialize(self) -> None:
        if Overlay is None or allocate is None:
            raise RuntimeError(f"pynq import failed: {PYNQ_IMPORT_ERROR!r}")
        if CosineSimilarity is None:
            raise RuntimeError(f"cosine_driver import failed: {COSINE_IMPORT_ERROR!r}")

        bitstream = Path(self.config.bitstream)
        weights_path = Path(self.config.weights)
        if not bitstream.exists():
            raise RuntimeError(f"bitstream not found: {bitstream}")
        if not weights_path.exists():
            raise RuntimeError(f"weights file not found: {weights_path}")

        overlay = Overlay(str(bitstream))
        tfm_name = self.config.tinyfacematch_ip or self._find_ip_name(overlay, "tinyfacematch")
        cosine_name = self.config.cosine_ip or self._find_ip_name(overlay, "cos")
        ip = getattr(overlay, tfm_name)
        cosine_base = overlay.ip_dict[cosine_name]["phys_addr"]
        cs = CosineSimilarity(bitfile=None, base_addr=cosine_base)

        weights = np.fromfile(weights_path, dtype=np.int16)
        if weights.size == 0:
            raise RuntimeError(f"empty weights file: {weights_path}")

        image_buf = allocate(shape=(112 * 112 * 3,), dtype=np.uint8)
        weights_buf = allocate(shape=(weights.size,), dtype=np.int16)
        scratch0 = allocate(shape=(SCRATCH_WORDS,), dtype=np.int16)
        scratch1 = allocate(shape=(SCRATCH_WORDS,), dtype=np.int16)
        scratch2 = allocate(shape=(SCRATCH_WORDS,), dtype=np.int16)
        out_buf = allocate(shape=(EMBED_WORDS,), dtype=np.int16)

        weights_buf[:] = weights
        scratch0[:] = 0
        scratch1[:] = 0
        scratch2[:] = 0
        out_buf[:] = 0
        weights_buf.flush()
        scratch0.flush()
        scratch1.flush()
        scratch2.flush()
        out_buf.flush()

        write_u64(ip, IMAGE_OFFSET, image_buf.physical_address)
        write_u64(ip, WEIGHTS_OFFSET, weights_buf.physical_address)
        write_u64(ip, SCRATCH0_OFFSET, scratch0.physical_address)
        write_u64(ip, SCRATCH1_OFFSET, scratch1.physical_address)
        write_u64(ip, SCRATCH2_OFFSET, scratch2.physical_address)
        write_u64(ip, OUT_OFFSET, out_buf.physical_address)

        cascade = find_cascade()
        detector = cv2.CascadeClassifier(cascade)
        if detector.empty():
            raise RuntimeError(f"failed to load Haar cascade: {cascade}")

        grabber = VideoFrameGrabber(self.config.video)
        if not grabber.is_opened():
            raise RuntimeError(f"could not open video source: {self.config.video}")
        grabber.start()
        publisher = threading.Thread(target=self._publish_loop, daemon=True)

        with self.lock:
            self.overlay = overlay
            self.ip = ip
            self.cs = cs
            self.grabber = grabber
            self.publisher = publisher
            self.detector = detector
            self.image_buf = image_buf
            self.weights_buf = weights_buf
            self.scratch0 = scratch0
            self.scratch1 = scratch1
            self.scratch2 = scratch2
            self.out_buf = out_buf
            self.buffers = [image_buf, weights_buf, scratch0, scratch1, scratch2, out_buf]
            self.auth_embs = load_authorized_embeddings(self.config.authorized_dir)
            self.status.update(
                {
                    "ready": True,
                    "running": True,
                    "state": "live",
                    "message": "Live verification running",
                    "authorized_count": len(self.auth_embs),
                    "tinyfacematch_ip": tfm_name,
                    "cosine_ip": cosine_name,
                    "error": None,
                }
            )
        publisher.start()

    @staticmethod
    def _find_ip_name(overlay: Any, needle: str) -> str:
        names = [name for name in overlay.ip_dict if needle.lower() in name.lower()]
        if not names:
            raise RuntimeError(f"could not find IP containing '{needle}'. Available: {list(overlay.ip_dict)}")
        return names[0]

    def _loop(self) -> None:
        interval = 1.0 / max(1.0, self.config.loop_fps)
        next_time = time.time()

        dummy_a = np.zeros(128, dtype=np.float32)
        dummy_b = np.zeros(128, dtype=np.float32)
        dummy_a[0] = 1.0
        dummy_b[1] = 1.0

        while not self.stop_event.is_set():
            now = time.time()
            if now < next_time:
                time.sleep(min(0.01, next_time - now))
                continue
            next_time = time.time() + interval
            loop_start = time.perf_counter()

            frame = self.grabber.latest()
            if frame is None:
                self._set_message("Waiting for camera frame")
                time.sleep(0.03)
                continue
            frame = center_crop_frame(frame, self.config.crop_zoom, self.config.crop_x, self.config.crop_y)

            self.frame_count += 1
            with self.lock:
                detections = list(self.last_detections)
                last_detection_time = self.last_detection_time
            if detections and time.time() - last_detection_time > self.config.face_hold_sec:
                detections = []
            if self.frame_count % max(1, self.config.detection_every) == 0:
                detect_start = time.perf_counter()
                fresh_detections = self._detect_and_embed(frame)
                detect_ms = (time.perf_counter() - detect_start) * 1000.0
                self.metrics["detect_ms"].add_ms(detect_ms)
                with self.lock:
                    now_s = time.time()
                    if fresh_detections:
                        detections = fresh_detections
                        self.last_detections = fresh_detections
                        self.last_detection_time = now_s
                    elif self.last_detections and now_s - self.last_detection_time <= self.config.face_hold_sec:
                        detections = list(self.last_detections)
                    else:
                        detections = []
                        self.last_detections = []

            result = self._match_detections(detections, dummy_a, dummy_b)
            self._consume_enrollment(detections)

            loop_ms = (time.perf_counter() - loop_start) * 1000.0
            self.metrics["loop_ms"].add_ms(loop_ms)
            with self.lock:
                self.status.update(
                    {
                        "frame_count": self.frame_count,
                        "face_present": bool(detections),
                        "match": result["match"],
                        "last_score": result["score"],
                        "best_score": result["score"],
                        "cnn_ms": self.metrics["hls_ms"].mean(),
                        "cosine_ms": self.metrics["cosine_ms"].mean(),
                        "detect_ms": self.metrics["detect_ms"].mean(),
                        "loop_ms": self.metrics["loop_ms"].mean(),
                        "fps": self.metrics["fps"].mean(),
                        "message": result["message"],
                    }
                )
                self.last_result = result

    def _publish_loop(self) -> None:
        interval = 1.0 / max(1.0, self.config.display_fps)
        last_fps_time = time.time()
        frames = 0
        while not self.stop_event.is_set():
            try:
                frame = self.grabber.latest() if self.grabber is not None else None
                if frame is None:
                    time.sleep(0.03)
                    continue
                frame = center_crop_frame(frame, self.config.crop_zoom, self.config.crop_x, self.config.crop_y)
                with self.lock:
                    detections = list(self.last_detections)
                    result = dict(self.last_result)
                    last_detection_time = self.last_detection_time
                if detections and time.time() - last_detection_time > self.config.face_hold_sec:
                    detections = []
                self._publish_frame(self._annotate(frame, detections, result))
                frames += 1
                elapsed = time.time() - last_fps_time
                if elapsed >= 1.0:
                    self.metrics["fps"].add_ms(frames / elapsed)
                    frames = 0
                    last_fps_time = time.time()
                time.sleep(interval)
            except Exception as exc:
                log.exception("Frame publisher failed")
                self._set_message(f"Frame publisher error: {exc}")
                time.sleep(0.25)

    def _detect_and_embed(self, frame_bgr: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
        small = cv2.resize(
            frame_bgr,
            (0, 0),
            fx=self.config.detection_scale,
            fy=self.config.detection_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.config.haar_scale_factor,
            minNeighbors=self.config.haar_min_neighbors,
            minSize=(self.config.haar_min_size, self.config.haar_min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        output: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
        inv = 1.0 / self.config.detection_scale
        height, width = frame_bgr.shape[:2]

        for x, y, w, h in faces[: self.config.max_faces]:
            xf, yf = int(x * inv), int(y * inv)
            wf, hf = int(w * inv), int(h * inv)
            pad = int(self.config.face_padding * max(wf, hf))
            x0 = max(0, xf - pad)
            y0 = max(0, yf - pad)
            x1 = min(width, xf + wf + pad)
            y1 = min(height, yf + hf + pad)
            crop = frame_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            embedding = self._hls_infer_face(crop)
            output.append(((x0, y0, x1 - x0, y1 - y0), embedding))

        return output

    def _hls_infer_face(self, face_bgr: np.ndarray) -> np.ndarray:
        inp = face_to_hls_input(face_bgr)
        self.image_buf[:] = inp
        self.out_buf[:] = 0
        self.image_buf.flush()
        self.out_buf.flush()

        start = time.perf_counter()
        self.ip.write(CTRL_OFFSET, 0x01)
        while (self.ip.read(CTRL_OFFSET) & 0x2) == 0:
            pass
        self.metrics["hls_ms"].add_ms((time.perf_counter() - start) * 1000.0)

        self.out_buf.invalidate()
        emb = np.array(self.out_buf, dtype=np.float32) / OUT_SCALE
        return normalize_embedding(emb)

    def _match_detections(
        self,
        detections: list[tuple[tuple[int, int, int, int], np.ndarray]],
        dummy_a: np.ndarray,
        dummy_b: np.ndarray,
    ) -> dict[str, Any]:
        if not detections:
            start = time.perf_counter()
            _, score = self.cs.compare(dummy_a, dummy_b, threshold=self.config.threshold)
            self.metrics["cosine_ms"].add_ms((time.perf_counter() - start) * 1000.0)
            return {"match": False, "score": float(score), "message": "No user found"}

        auth = list(self.auth_embs)
        if not auth:
            return {"match": False, "score": None, "message": "Face detected, no enrolled profiles"}

        best_score = -1.0
        best_match = False
        start = time.perf_counter()
        for _, cam_emb in detections:
            for auth_emb in auth:
                match, score = self.cs.compare(
                    cam_emb.astype(np.float32),
                    auth_emb.astype(np.float32),
                    threshold=self.config.threshold,
                )
                score = float(score)
                if score > best_score:
                    best_score = score
                if match:
                    best_match = True
                    break
            if best_match:
                break
        self.metrics["cosine_ms"].add_ms((time.perf_counter() - start) * 1000.0)

        if best_match:
            return {"match": True, "score": best_score, "message": f"Match found: {best_score:.4f}"}
        return {"match": False, "score": best_score, "message": f"Match not found: {best_score:.4f}"}

    def _consume_enrollment(self, detections: list[tuple[tuple[int, int, int, int], np.ndarray]]) -> None:
        with self.lock:
            request_obj = self.enroll_request
        if request_obj is None or not detections:
            return

        request_obj["embeddings"].append(detections[0][1])
        captured = len(request_obj["embeddings"])
        target = int(request_obj["samples"])
        with self.lock:
            self.status["enrollment"] = {"name": request_obj["name"], "captured": captured, "target": target}

        if captured >= target:
            profile = normalize_embedding(np.mean(np.stack(request_obj["embeddings"]), axis=0))
            saved_path = save_authorized_embedding(request_obj["name"], profile, self.config.authorized_dir)
            count = self.reload_authorized()
            with self.lock:
                self.enroll_request = None
                self.status["enrollment"] = None
                self.status["message"] = f"Saved {saved_path.name}; {count} profile(s) loaded"

    def _annotate(
        self,
        frame: np.ndarray,
        detections: list[tuple[tuple[int, int, int, int], np.ndarray]],
        result: dict[str, Any],
    ) -> np.ndarray:
        annotated = frame.copy()
        match = bool(result["match"])
        color = (48, 220, 112) if match else (56, 92, 245)
        if not detections:
            color = (58, 183, 255)

        for bbox, _ in detections:
            x, y, w, h = bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        return annotated

    def _publish_frame(self, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality])
        if not ok:
            return
        with self.lock:
            self.latest_jpeg = encoded.tobytes()
            self.latest_frame_time = time.time()

    def _set_message(self, message: str) -> None:
        with self.lock:
            self.status["message"] = message

    def _cleanup(self) -> None:
        with self.lock:
            grabber = self.grabber
            publisher = self.publisher
            buffers = list(self.buffers)
            self.grabber = None
            self.publisher = None
            self.buffers = []
        if grabber is not None:
            grabber.stop()
            grabber.join(timeout=1.0)
        if publisher is not None and publisher.is_alive() and publisher is not threading.current_thread():
            publisher.join(timeout=1.0)
        for buf in buffers:
            try:
                buf.freebuffer()
            except Exception:
                pass


def create_app(config: AppConfig) -> Flask:
    app = Flask(__name__)
    service = FaceService(config)
    app.config["FACE_SERVICE"] = service

    @app.route("/")
    def index() -> str:
        return render_template("index.html", config=asdict(config))

    @app.route("/video_feed")
    def video_feed() -> Response:
        def generate():
            while True:
                frame = service.jpeg_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.03)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/status")
    def api_status() -> Response:
        return jsonify(service.snapshot())

    @app.post("/api/start")
    def api_start() -> Response:
        service.start()
        return jsonify({"ok": True, "status": service.snapshot()})

    @app.post("/api/stop")
    def api_stop() -> Response:
        service.stop()
        return jsonify({"ok": True, "status": service.snapshot()})

    @app.post("/api/enroll")
    def api_enroll() -> Response:
        payload = request.get_json(silent=True) or {}
        name = str(payload.get("name") or "me")
        samples = int(payload.get("samples") or 3)
        service.request_enrollment(name, samples)
        return jsonify({"ok": True, "status": service.snapshot()})

    @app.post("/api/reload_profiles")
    def api_reload_profiles() -> Response:
        count = service.reload_authorized()
        return jsonify({"ok": True, "authorized_count": count, "status": service.snapshot()})

    @app.post("/api/threshold")
    def api_threshold() -> Response:
        payload = request.get_json(silent=True) or {}
        service.set_threshold(float(payload.get("threshold", config.threshold)))
        return jsonify({"ok": True, "status": service.snapshot()})

    if config.autostart:
        service.start()

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TinyFaceMatch PYNQ Flask demo.")
    parser.add_argument("--bitstream", default=os.environ.get("TFN_BITSTREAM", "hvCNN_q8.bit"))
    parser.add_argument("--weights", default=os.environ.get("TFN_WEIGHTS", "weights.bin"))
    parser.add_argument("--video", default=os.environ.get("TFN_VIDEO", "http://192.168.0.114:4747/video"))
    parser.add_argument("--authorized-dir", default=os.environ.get("TFN_AUTH_DIR", "authorized_embeddings_hls"))
    parser.add_argument("--host", default=os.environ.get("TFN_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TFN_PORT", "5000")))
    parser.add_argument("--threshold", type=float, default=float(os.environ.get("TFN_THRESHOLD", "0.5")))
    parser.add_argument("--loop-fps", type=float, default=float(os.environ.get("TFN_LOOP_FPS", "12")))
    parser.add_argument("--display-fps", type=float, default=float(os.environ.get("TFN_DISPLAY_FPS", "15")))
    parser.add_argument("--detection-scale", type=float, default=float(os.environ.get("TFN_DETECTION_SCALE", "0.25")))
    parser.add_argument("--detection-every", type=int, default=int(os.environ.get("TFN_DETECTION_EVERY", "2")))
    parser.add_argument("--haar-scale-factor", type=float, default=float(os.environ.get("TFN_HAAR_SCALE_FACTOR", "1.15")))
    parser.add_argument("--haar-min-neighbors", type=int, default=int(os.environ.get("TFN_HAAR_MIN_NEIGHBORS", "4")))
    parser.add_argument("--haar-min-size", type=int, default=int(os.environ.get("TFN_HAAR_MIN_SIZE", "16")))
    parser.add_argument("--face-hold-sec", type=float, default=float(os.environ.get("TFN_FACE_HOLD_SEC", "2.0")))
    parser.add_argument("--crop-zoom", type=float, default=float(os.environ.get("TFN_CROP_ZOOM", "1.0")))
    parser.add_argument("--crop-x", type=float, default=float(os.environ.get("TFN_CROP_X", "0.0")))
    parser.add_argument("--crop-y", type=float, default=float(os.environ.get("TFN_CROP_Y", "0.0")))
    parser.add_argument("--tinyfacematch-ip", default=os.environ.get("TFN_IP"))
    parser.add_argument("--cosine-ip", default=os.environ.get("TFN_COSINE_IP"))
    parser.add_argument("--autostart", action="store_true", default=os.environ.get("TFN_AUTOSTART") == "1")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    config = AppConfig(
        bitstream=args.bitstream,
        weights=args.weights,
        video=args.video,
        authorized_dir=args.authorized_dir,
        host=args.host,
        port=args.port,
        threshold=args.threshold,
        loop_fps=args.loop_fps,
        display_fps=args.display_fps,
        detection_scale=args.detection_scale,
        detection_every=args.detection_every,
        haar_scale_factor=args.haar_scale_factor,
        haar_min_neighbors=args.haar_min_neighbors,
        haar_min_size=args.haar_min_size,
        face_hold_sec=args.face_hold_sec,
        crop_zoom=args.crop_zoom,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
        tinyfacematch_ip=args.tinyfacematch_ip,
        cosine_ip=args.cosine_ip,
        autostart=args.autostart,
    )
    app = create_app(config)
    app.run(host=config.host, port=config.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
