import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


AGE_GENDER_ACTIONS = ["age", "gender"]

# Age correction: DeepFace tends to overestimate age.
# Subtract this value from detected age before displaying.
AGE_OFFSET = 5


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _local_deepface_path() -> Path:
    return Path(__file__).resolve().parent / "deepface"


def ensure_local_deepface_importable() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    os.environ.setdefault("DEEPFACE_HOME", str(Path(__file__).resolve().parent))
    local_deepface = _local_deepface_path()
    if local_deepface.exists():
        deepface_path = str(local_deepface)
        if deepface_path not in sys.path:
            sys.path.insert(0, deepface_path)


def normalize_analysis_results(
    raw_faces: Iterable[Dict[str, Any]],
    frame_index: int,
    timestamp: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for face_index, face in enumerate(raw_faces):
        gender = str(face.get("dominant_gender", "Unknown"))
        gender_scores = face.get("gender") or {}
        gender_confidence = _number(gender_scores.get(gender))
        region = face.get("region") or {}

        # Apply age correction: subtract AGE_OFFSET from detected age
        raw_age = _int(face.get("age"))
        corrected_age = max(0, raw_age - AGE_OFFSET)

        records.append(
            {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "face_index": face_index,
                "age": corrected_age,
                "gender": gender,
                "gender_confidence": round(gender_confidence, 4),
                "bbox": {
                    "x": _int(region.get("x")),
                    "y": _int(region.get("y")),
                    "w": _int(region.get("w")),
                    "h": _int(region.get("h")),
                },
                "face_confidence": round(_number(face.get("face_confidence")), 6),
            }
        )

    return records


def normalize_no_face_frame(frame_index: int, timestamp: str) -> Dict[str, Any]:
    return {"frame_index": frame_index, "timestamp": timestamp, "faces": []}


@dataclass
class JsonResultWriter:
    output_path: Path
    source: str = "realtime_camera"
    actions: List[str] = field(default_factory=lambda: AGE_GENDER_ACTIONS.copy())
    frames: List[Dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=_timestamp)

    def append_frame(
        self,
        frame_index: int,
        timestamp: str,
        faces: List[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> None:
        frame: Dict[str, Any] = {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "faces": faces,
        }
        if error:
            frame["error"] = error
        self.frames.append(frame)

    def payload(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "actions": self.actions,
            "started_at": self.started_at,
            "finished_at": _timestamp(),
            "frames_analyzed": len(self.frames),
            "faces_detected": sum(len(frame["faces"]) for frame in self.frames),
            "frames": self.frames,
        }

    def write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(self.payload(), indent=2),
            encoding="utf-8",
        )


def draw_faces(frame: Any, faces: List[Dict[str, Any]]) -> Any:
    """Draw green bounding boxes and age/gender labels on the frame."""
    import cv2

    GREEN = (0, 255, 0)

    for face in faces:
        bbox = face["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        label = f'{face["age"]} {face["gender"]} {face["gender_confidence"]:.1f}%'
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        cv2.putText(
            frame,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            GREEN,
            2,
        )
    return frame


def analyze_frame(frame: Any, detector_backend: str) -> List[Dict[str, Any]]:
    ensure_local_deepface_importable()
    from deepface import DeepFace

    raw_faces = DeepFace.analyze(
        img_path=frame,
        actions=AGE_GENDER_ACTIONS,
        detector_backend=detector_backend,
        enforce_detection=False,
        silent=True,
    )
    return list(raw_faces)


def run_realtime_age_gender(
    source: str,
    output_path: Path,
    detector_backend: str,
    frame_interval: int,
    max_frames: Optional[int],
    duration_seconds: Optional[float],
    preview: bool,
) -> Path:
    import cv2

    camera_source: Any = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/video source: {source}")

    writer = JsonResultWriter(output_path=output_path)
    frame_index = 0
    analyzed_frames = 0
    started = time.time()

    # Persistent face data: drawn on EVERY frame so the green box never blinks
    last_faces: List[Dict[str, Any]] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1

            # On non-analysis frames, still draw the LAST detected faces
            if frame_index % frame_interval != 0:
                if preview:
                    display_frame = draw_faces(frame.copy(), last_faces)
                    cv2.imshow("Age/Gender realtime", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            analyzed_frames += 1
            timestamp = _timestamp()
            faces: List[Dict[str, Any]] = []
            error = None

            try:
                raw_faces = analyze_frame(frame=frame, detector_backend=detector_backend)
                faces = normalize_analysis_results(
                    raw_faces=raw_faces,
                    frame_index=frame_index,
                    timestamp=timestamp,
                )
            except Exception as exc:  # keep long-running capture alive
                error = str(exc)

            # Update persistent face cache (only if we got results)
            if faces:
                last_faces = faces

            writer.append_frame(
                frame_index=frame_index,
                timestamp=timestamp,
                faces=faces,
                error=error,
            )
            writer.write()

            # Print live JSON to console for each analyzed frame
            frame_result = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "faces": faces,
            }
            if error:
                frame_result["error"] = error
            print(json.dumps(frame_result, indent=2))

            if preview:
                display_frame = draw_faces(frame=frame.copy(), faces=last_faces)
                cv2.imshow("Age/Gender realtime", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames is not None and analyzed_frames >= max_frames:
                break
            if duration_seconds is not None and time.time() - started >= duration_seconds:
                break
    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()
        writer.write()

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real-time camera age and gender deduction with JSON output."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or video path. Default: 0",
    )
    parser.add_argument(
        "--output",
        default="age_gender_results_v2.json",
        help="JSON output path. Default: age_gender_results_v2.json",
    )
    parser.add_argument(
        "--detector-backend",
        default="opencv",
        help="DeepFace detector backend. Default: opencv",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=10,
        help="Analyze every Nth frame. Default: 10",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N analyzed frames. Default: run until stopped",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after N seconds. Default: run until stopped",
    )
    parser.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        default=True,
        help="Disable the live OpenCV window. By default, preview is ON.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    frame_interval = max(1, args.frame_interval)
    output_path = run_realtime_age_gender(
        source=args.source,
        output_path=Path(args.output),
        detector_backend=args.detector_backend,
        frame_interval=frame_interval,
        max_frames=args.max_frames,
        duration_seconds=args.duration,
        preview=args.preview,
    )
    print(f"Age/gender JSON written to {output_path}")


if __name__ == "__main__":
    main()
