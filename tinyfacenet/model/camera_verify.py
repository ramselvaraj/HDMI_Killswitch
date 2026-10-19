from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data import PROJECT_MEAN, PROJECT_SCALE
from .model import TinyFaceMatch128


def load_model(checkpoint_path: Path, device: torch.device) -> TinyFaceMatch128:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    width_mult = float(checkpoint.get("config", {}).get("width_mult", 0.5))
    model = TinyFaceMatch128(width_mult=width_mult)
    model.load_state_dict(checkpoint["backbone_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"could not load Haar cascade: {cascade_path}")
    return detector


def largest_face(frame: np.ndarray, detector: cv2.CascadeClassifier, padding: float) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
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


def preprocess_face(face_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    image = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    image = (image - PROJECT_MEAN) / PROJECT_SCALE
    chw = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0).to(device)


@torch.no_grad()
def embed_face(
    model: TinyFaceMatch128,
    face_bgr: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    embedding = model(preprocess_face(face_bgr, device)).squeeze(0).cpu()
    return F.normalize(embedding.unsqueeze(0), p=2, dim=1).squeeze(0)


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

    cv2.putText(
        frame,
        text,
        anchor,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )


def collect_embeddings(
    cap: cv2.VideoCapture,
    detector: cv2.CascadeClassifier,
    model: TinyFaceMatch128,
    device: torch.device,
    seconds: float,
    min_samples: int,
    padding: float,
) -> torch.Tensor:
    embeddings: list[torch.Tensor] = []
    start = time.time()
    window_name = "TinyFaceMatch enrollment"

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("could not read from camera")

        elapsed = time.time() - start
        face, box = largest_face(frame, detector, padding)
        if face is None:
            draw_status(frame, None, "No face detected", (0, 0, 255))
        else:
            embeddings.append(embed_face(model, face, device))
            draw_status(frame, box, f"Enrolling {len(embeddings)} samples", (0, 220, 255))

        remaining = max(0.0, seconds - elapsed)
        cv2.putText(
            frame,
            f"Keep face visible: {remaining:.1f}s",
            (20, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise KeyboardInterrupt
        if elapsed >= seconds:
            break

    cv2.destroyWindow(window_name)
    if len(embeddings) < min_samples:
        raise RuntimeError(f"captured {len(embeddings)} samples, need at least {min_samples}")
    return torch.stack(embeddings)


def save_profile(path: Path, profile: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, embedding=profile.numpy().astype(np.float32), embedding_dim=np.array(128, dtype=np.int32))
    print(f"saved profile: {path}")


def load_profile(path: Path) -> torch.Tensor:
    data = np.load(path)
    embedding = torch.from_numpy(data["embedding"].astype(np.float32)).flatten()
    if embedding.numel() != 128:
        raise RuntimeError(f"profile has {embedding.numel()} dims, expected 128")
    profile = F.normalize(embedding.unsqueeze(0), p=2, dim=1).squeeze(0)
    print(f"loaded profile: {path}")
    return profile


def enroll_profile(
    cap: cv2.VideoCapture,
    detector: cv2.CascadeClassifier,
    model: TinyFaceMatch128,
    device: torch.device,
    seconds: float,
    min_samples: int,
    padding: float,
) -> torch.Tensor:
    print("Enrollment starting. Look at the camera.")
    embeddings = collect_embeddings(
        cap=cap,
        detector=detector,
        model=model,
        device=device,
        seconds=seconds,
        min_samples=min_samples,
        padding=padding,
    )
    profile = embeddings.mean(dim=0, keepdim=True)
    return F.normalize(profile, p=2, dim=1).squeeze(0)


def verify_loop(
    cap: cv2.VideoCapture,
    detector: cv2.CascadeClassifier,
    model: TinyFaceMatch128,
    device: torch.device,
    profile: torch.Tensor,
    threshold: float,
    padding: float,
    smoothing: float,
) -> None:
    window_name = "TinyFaceMatch live verification"
    smooth_score: float | None = None

    print(f"Live verification started with threshold={threshold:.3f}")
    print("Press q in the camera window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("could not read from camera")

        face, box = largest_face(frame, detector, padding)
        if face is None:
            smooth_score = None
            draw_status(frame, None, "No face detected", (0, 0, 255))
        else:
            embedding = embed_face(model, face, device)
            score = float(torch.dot(profile, embedding).item())
            if smooth_score is None:
                smooth_score = score
            else:
                smooth_score = smoothing * smooth_score + (1.0 - smoothing) * score

            is_match = smooth_score >= threshold
            color = (0, 220, 0) if is_match else (0, 0, 255)
            label = "MATCH" if is_match else "NOT MATCH"
            draw_status(frame, box, f"{label} cos={smooth_score:.3f}", color)

        cv2.putText(
            frame,
            f"threshold={threshold:.3f}",
            (20, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live camera verification with TinyFaceMatch128.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.14)
    parser.add_argument("--enroll-seconds", type=float, default=7.0)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--save-profile", type=Path, default=Path("profiles/tinyfacematch_me.npz"))
    parser.add_argument("--load-profile", type=Path, default=None)
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--smoothing", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = load_model(args.checkpoint, device)
    detector = load_face_detector()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {args.camera}")

    try:
        print(f"device: {device}")
        print("model input: 112x112 RGB")
        if args.load_profile is not None:
            profile = load_profile(args.load_profile)
        else:
            profile = enroll_profile(
                cap=cap,
                detector=detector,
                model=model,
                device=device,
                seconds=args.enroll_seconds,
                min_samples=args.min_samples,
                padding=args.padding,
            )
            if args.save_profile is not None:
                save_profile(args.save_profile, profile)

        verify_loop(
            cap=cap,
            detector=detector,
            model=model,
            device=device,
            profile=profile,
            threshold=args.threshold,
            padding=args.padding,
            smoothing=args.smoothing,
        )
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
