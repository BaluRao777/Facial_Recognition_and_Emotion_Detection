"""Realtime face + emotion recognition from a webcam or video file."""

import argparse
import time

import cv2
import numpy as np

from src.inference import FaceEmotionPredictor

SMOOTHING = 0.6  # weight of the previous frames in the running average


class _Track:
    """Running average of one face's probabilities across frames."""

    def __init__(self, box, emotion_probs, face_probs):
        self.box = box
        self.emotion_probs = emotion_probs
        self.face_probs = face_probs

    def update(self, box, emotion_probs, face_probs):
        self.box = box
        if emotion_probs is not None:
            self.emotion_probs = SMOOTHING * self.emotion_probs + (1 - SMOOTHING) * emotion_probs
        if face_probs is not None:
            self.face_probs = SMOOTHING * self.face_probs + (1 - SMOOTHING) * face_probs


def _centre(box):
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def _match_tracks(tracks, boxes, emotion_probs, face_probs):
    """Match detections to the nearest previous face so labels do not flicker."""
    new_tracks = []
    unused = list(tracks)
    for i, box in enumerate(boxes):
        e = None if emotion_probs is None else emotion_probs[i]
        f = None if face_probs is None else face_probs[i]
        cx, cy = _centre(box)
        best = None
        for track in unused:
            tx, ty = _centre(track.box)
            dist = np.hypot(cx - tx, cy - ty)
            if dist < max(box[2], box[3]) and (best is None or dist < best[0]):
                best = (dist, track)
        if best is None:
            new_tracks.append(_Track(box, e, f))
        else:
            unused.remove(best[1])
            best[1].update(box, e, f)
            new_tracks.append(best[1])
    return new_tracks


def _draw(frame, track, result):
    x, y, w, h = track.box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    lines = []
    if "identity" in result:
        lines.append(f"{result['identity']} {result['identity_confidence']:.0%}")
    if "emotion" in result:
        lines.append(f"{result['emotion']} {result['emotion_confidence']:.0%}")
    for i, text in enumerate(reversed(lines)):
        pos = (x, max(15, y - 8 - 20 * i))
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index or video file (default: 0)")
    parser.add_argument("--output", help="Write the annotated video to this file")
    parser.add_argument("--no-display", action="store_true", help="Do not open a window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames")
    parser.add_argument("--emotion-only", action="store_true")
    parser.add_argument("--face-only", action="store_true")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        parser.error(f"Could not open video source: {args.source}")

    predictor = FaceEmotionPredictor(
        emotion=not args.face_only, face=not args.emotion_only
    )

    writer = None
    tracks = []
    frames = 0
    fps = 0.0
    start = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            tick = time.perf_counter()

            boxes, emotion_probs, face_probs = predictor.predict_probabilities(frame)
            tracks = _match_tracks(tracks, boxes, emotion_probs, face_probs)
            for track in tracks:
                _draw(frame, track, predictor.describe(track.emotion_probs, track.face_probs))

            frames += 1
            instant = 1.0 / max(time.perf_counter() - tick, 1e-6)
            fps = instant if frames == 1 else 0.9 * fps + 0.1 * instant
            cv2.putText(frame, f"{fps:.0f} FPS", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if args.output:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                        capture.get(cv2.CAP_PROP_FPS) or 30.0, (w, h),
                    )
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Face + Emotion (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    if frames:
        print(f"Processed {frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} FPS average)")


if __name__ == "__main__":
    main()
