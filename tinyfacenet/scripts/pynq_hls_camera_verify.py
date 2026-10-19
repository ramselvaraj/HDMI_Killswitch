from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from pynq import Overlay, allocate


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


def resolve_ip(overlay: Overlay, ip_name: str | None):
    if ip_name:
        return getattr(overlay, ip_name)

    matches = [name for name in overlay.ip_dict if "tinyfacematch" in name.lower()]
    if not matches:
        raise RuntimeError(f"could not find tinyfacematch IP in overlay; available IPs: {list(overlay.ip_dict)}")
    if len(matches) > 1:
        print(f"multiple tinyfacematch IPs found, using {matches[0]}: {matches}")
    return getattr(overlay, matches[0])


def write_u64(ip, offset: int, value: int) -> None:
    ip.write(offset, int(value) & 0xFFFFFFFF)
    ip.write(offset + 4, (int(value) >> 32) & 0xFFFFFFFF)


def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"could not load Haar cascade: {cascade_path}")
    return detector


def largest_face(
    frame: np.ndarray,
    detector: cv2.CascadeClassifier,
    padding: float,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 70),
    )
    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    pad = int(padding * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(frame.shape[1], x + w + pad)
    y1 = min(frame.shape[0], y + h + pad)
    return frame[y0:y1, x0:x1], (x0, y0, x1, y1)


def center_crop(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    x1 = x0 + side
    y1 = y0 + side
    return frame[y0:y1, x0:x1], (x0, y0, x1, y1)


def face_to_hls_input(face_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb.astype(np.uint8).reshape(-1))


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    emb = embedding.astype(np.float32).reshape(-1)
    norm = np.linalg.norm(emb) + 1e-12
    return emb / norm


class TinyFaceMatchHLS:
    def __init__(self, bitfile: Path, weights_path: Path, ip_name: str | None) -> None:
        self.overlay = Overlay(str(bitfile))
        self.ip = resolve_ip(self.overlay, ip_name)

        weights = np.fromfile(weights_path, dtype=np.int16)
        if weights.size == 0:
            raise RuntimeError(f"empty weights file: {weights_path}")

        self.image_buf = allocate(shape=(112 * 112 * 3,), dtype=np.uint8)
        self.weights_buf = allocate(shape=(weights.size,), dtype=np.int16)
        self.scratch0 = allocate(shape=(SCRATCH_WORDS,), dtype=np.int16)
        self.scratch1 = allocate(shape=(SCRATCH_WORDS,), dtype=np.int16)
        self.scratch2 = allocate(shape=(SCRATCH_WORDS,), dtype=np.int16)
        self.out_buf = allocate(shape=(EMBED_WORDS,), dtype=np.int16)

        self.weights_buf[:] = weights
        self.scratch0[:] = 0
        self.scratch1[:] = 0
        self.scratch2[:] = 0
        self.out_buf[:] = 0

        self.weights_buf.flush()
        self.scratch0.flush()
        self.scratch1.flush()
        self.scratch2.flush()
        self.out_buf.flush()

        write_u64(self.ip, IMAGE_OFFSET, self.image_buf.physical_address)
        write_u64(self.ip, WEIGHTS_OFFSET, self.weights_buf.physical_address)
        write_u64(self.ip, SCRATCH0_OFFSET, self.scratch0.physical_address)
        write_u64(self.ip, SCRATCH1_OFFSET, self.scratch1.physical_address)
        write_u64(self.ip, SCRATCH2_OFFSET, self.scratch2.physical_address)
        write_u64(self.ip, OUT_OFFSET, self.out_buf.physical_address)

    def infer(self, image_hwc_rgb_flat: np.ndarray) -> tuple[np.ndarray, float]:
        self.image_buf[:] = image_hwc_rgb_flat
        self.out_buf[:] = 0
        self.image_buf.flush()
        self.out_buf.flush()

        start = time.time()
        self.ip.write(CTRL_OFFSET, 0x01)
        while (self.ip.read(CTRL_OFFSET) & 0x2) == 0:
            pass
        elapsed = time.time() - start

        self.out_buf.invalidate()
        embedding = np.array(self.out_buf, dtype=np.float32) / OUT_SCALE
        return normalize_embedding(embedding), elapsed

    def close(self) -> None:
        for buf in (self.image_buf, self.weights_buf, self.scratch0, self.scratch1, self.scratch2, self.out_buf):
            try:
                buf.freebuffer()
            except Exception:
                pass


def draw_status(
    frame: np.ndarray,
    box: tuple[int, int, int, int] | None,
    text: str,
    color: tuple[int, int, int],
) -> None:
    if box is not None:
        x0, y0, x1, y1 = box
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        anchor = (x0, max(30, y0 - 10))
    else:
        anchor = (20, 40)
    cv2.putText(frame, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def get_face_or_crop(
    frame: np.ndarray,
    detector: cv2.CascadeClassifier | None,
    padding: float,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    if detector is None:
        return center_crop(frame)
    return largest_face(frame, detector, padding)


def collect_profile(
    cap: cv2.VideoCapture,
    accelerator: TinyFaceMatchHLS,
    detector: cv2.CascadeClassifier | None,
    seconds: float,
    min_samples: int,
    padding: float,
    display: bool,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("could not read from camera")

        face, box = get_face_or_crop(frame, detector, padding)
        if face is None:
            draw_status(frame, None, "No face detected", (0, 0, 255))
        else:
            emb, elapsed = accelerator.infer(face_to_hls_input(face))
            embeddings.append(emb)
            draw_status(frame, box, f"Enrolling {len(embeddings)} infer={elapsed:.2f}s", (0, 220, 255))

        remaining = max(0.0, seconds - (time.time() - start))
        cv2.putText(frame, f"Keep face visible: {remaining:.1f}s", (20, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if display:
            cv2.imshow("TinyFaceMatch HLS enrollment", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt

        if time.time() - start >= seconds:
            break

    if display:
        cv2.destroyWindow("TinyFaceMatch HLS enrollment")
    if len(embeddings) < min_samples:
        raise RuntimeError(f"captured {len(embeddings)} samples, need at least {min_samples}")
    return normalize_embedding(np.mean(np.stack(embeddings), axis=0))


def save_profile(path: Path, profile: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, embedding=profile.astype(np.float32), embedding_dim=np.array(128, dtype=np.int32))
    print(f"saved profile: {path}")


def load_profile(path: Path) -> np.ndarray:
    data = np.load(path)
    profile = data["embedding"].astype(np.float32).reshape(-1)
    if profile.size != 128:
        raise RuntimeError(f"profile has {profile.size} dims, expected 128")
    print(f"loaded profile: {path}")
    return normalize_embedding(profile)


def verify_loop(
    cap: cv2.VideoCapture,
    accelerator: TinyFaceMatchHLS,
    detector: cv2.CascadeClassifier | None,
    profile: np.ndarray,
    threshold: float,
    padding: float,
    smoothing: float,
    display: bool,
) -> None:
    smooth_score: float | None = None
    print(f"live HLS verification started; threshold={threshold:.3f}")
    print("press q in the OpenCV window to quit" if display else "press Ctrl-C to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("could not read from camera")

        face, box = get_face_or_crop(frame, detector, padding)
        if face is None:
            smooth_score = None
            draw_status(frame, None, "No face detected", (0, 0, 255))
        else:
            embedding, elapsed = accelerator.infer(face_to_hls_input(face))
            score = float(np.dot(profile, embedding))
            smooth_score = score if smooth_score is None else smoothing * smooth_score + (1.0 - smoothing) * score
            is_match = smooth_score >= threshold
            color = (0, 220, 0) if is_match else (0, 0, 255)
            label = "MATCH" if is_match else "NOT MATCH"
            draw_status(frame, box, f"{label} cos={smooth_score:.3f} infer={elapsed:.2f}s", color)

        cv2.putText(frame, f"threshold={threshold:.3f}", (20, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if display:
            cv2.imshow("TinyFaceMatch HLS live verification", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        else:
            if smooth_score is not None:
                print(f"cos={smooth_score:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenCV camera verification using the TinyFaceMatch DDR HLS IP on PYNQ.")
    parser.add_argument("--bitfile", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--ip-name", default=None, help="Overlay IP instance name; auto-detects tinyfacematch if omitted.")
    parser.add_argument("--camera", default="0", help="OpenCV camera index or stream URL.")
    parser.add_argument("--threshold", type=float, default=0.14)
    parser.add_argument("--enroll-seconds", type=float, default=20.0)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--save-profile", type=Path, default=Path("profiles/tinyfacematch_hls_me.npz"))
    parser.add_argument("--load-profile", type=Path, default=None)
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--no-face-detect", action="store_true", help="Use a center crop instead of Haar face detection.")
    parser.add_argument("--no-display", action="store_true", help="Run without cv2.imshow windows.")
    return parser


def parse_camera(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    args = build_parser().parse_args()
    accelerator = TinyFaceMatchHLS(args.bitfile, args.weights, args.ip_name)
    detector = None if args.no_face_detect else load_face_detector()
    display = not args.no_display

    cap = cv2.VideoCapture(parse_camera(args.camera))
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera: {args.camera}")

    try:
        if args.load_profile is not None:
            profile = load_profile(args.load_profile)
        else:
            print("enrollment starting; look at the camera")
            profile = collect_profile(
                cap=cap,
                accelerator=accelerator,
                detector=detector,
                seconds=args.enroll_seconds,
                min_samples=args.min_samples,
                padding=args.padding,
                display=display,
            )
            if args.save_profile is not None:
                save_profile(args.save_profile, profile)

        verify_loop(
            cap=cap,
            accelerator=accelerator,
            detector=detector,
            profile=profile,
            threshold=args.threshold,
            padding=args.padding,
            smoothing=args.smoothing,
            display=display,
        )
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        accelerator.close()
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
