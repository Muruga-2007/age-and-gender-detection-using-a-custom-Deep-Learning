import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from realtime_age_gender import (
    JsonResultWriter,
    ensure_local_deepface_importable,
    normalize_analysis_results,
    normalize_no_face_frame,
)


class NormalizeAnalysisResultsTests(unittest.TestCase):
    def test_normalizes_age_gender_faces_only(self):
        raw_faces = [
            {
                "age": 29,
                "dominant_gender": "Man",
                "gender": {"Man": 97.25, "Woman": 2.75},
                "region": {"x": 10, "y": 20, "w": 100, "h": 120},
                "face_confidence": 0.91,
                "dominant_emotion": "happy",
                "race": {"asian": 10},
            },
            {
                "age": 33.9,
                "dominant_gender": "Woman",
                "gender": {"Man": 4.5, "Woman": 95.5},
                "region": {"x": 140, "y": 30, "w": 80, "h": 90},
                "face_confidence": 0.88,
            },
        ]

        records = normalize_analysis_results(
            raw_faces=raw_faces,
            frame_index=12,
            timestamp="2026-05-03T12:00:00+00:00",
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["frame_index"], 12)
        self.assertEqual(records[0]["face_index"], 0)
        self.assertEqual(records[0]["age"], 29)
        self.assertEqual(records[0]["gender"], "Man")
        self.assertEqual(records[0]["gender_confidence"], 97.25)
        self.assertEqual(records[0]["bbox"], {"x": 10, "y": 20, "w": 100, "h": 120})
        self.assertNotIn("dominant_emotion", records[0])
        self.assertNotIn("race", records[0])

        self.assertEqual(records[1]["face_index"], 1)
        self.assertEqual(records[1]["age"], 34)
        self.assertEqual(records[1]["gender"], "Woman")
        self.assertEqual(records[1]["gender_confidence"], 95.5)

    def test_normalizes_no_face_frame(self):
        record = normalize_no_face_frame(
            frame_index=5,
            timestamp="2026-05-03T12:01:00+00:00",
        )

        self.assertEqual(
            record,
            {
                "frame_index": 5,
                "timestamp": "2026-05-03T12:01:00+00:00",
                "faces": [],
            },
        )


class JsonResultWriterTests(unittest.TestCase):
    def test_writes_json_session_with_records(self):
        output_path = Path.cwd() / "age_gender_results.test.json"
        try:
            writer = JsonResultWriter(output_path=output_path)
            writer.append_frame(
                frame_index=1,
                timestamp="2026-05-03T12:02:00+00:00",
                faces=[{"frame_index": 1, "face_index": 0, "age": 40, "gender": "Man"}],
            )
            writer.append_frame(
                frame_index=2,
                timestamp="2026-05-03T12:02:01+00:00",
                faces=[],
            )
            writer.write()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
        finally:
            if output_path.exists():
                output_path.unlink()

        self.assertEqual(payload["source"], "realtime_camera")
        self.assertEqual(payload["actions"], ["age", "gender"])
        self.assertEqual(payload["frames_analyzed"], 2)
        self.assertEqual(payload["faces_detected"], 1)
        self.assertEqual(payload["frames"][0]["faces"][0]["age"], 40)
        self.assertEqual(payload["frames"][1]["faces"], [])


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_sets_workspace_deepface_home_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            ensure_local_deepface_importable()

            self.assertEqual(os.environ["DEEPFACE_HOME"], str(Path.cwd()))


if __name__ == "__main__":
    unittest.main()
