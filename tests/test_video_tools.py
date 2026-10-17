import base64
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import cv2
import numpy as np

import video_tools as video


class FakeVisionModel:
    def with_structured_output(self, *args, **kwargs):
        return self

    def invoke(self, messages):
        findings = []
        for block in messages[-1].content:
            if block["type"] == "text" and block["text"].startswith("frame_index="):
                index, timestamp = block["text"].split("; ")
                seconds = float(timestamp.split("=")[1].split()[0])
                findings.append(video.FrameFinding(
                    frame_index=int(index.split("=")[1]),
                    description="Synthetic test evidence.",
                    count=2 if seconds >= 2 else 1,
                    uncertain=False,
                ))
            elif block["type"] == "image_url":
                data_url = block["image_url"]["url"]
                assert base64.b64decode(data_url.split(",", 1)[1]).startswith(b"\xff\xd8")
        return video.BatchFindings(findings=findings)


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.path = self.folder / "clip.avi"
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (64, 64))
        self.assertTrue(writer.isOpened())
        try:
            for index in range(16):
                writer.write(np.full((64, 64, 3), index * 12, dtype=np.uint8))
        finally:
            writer.release()

    def test_sampling_covers_both_endpoints_and_respects_budget(self):
        times = video.sample_times(5, 105, 0.1, max_frames=12)
        self.assertEqual(len(times), 12)
        self.assertEqual((times[0], times[-1]), (5, 105))
        for params in ((-1, 5, 1), (5, 1, 1), (0, 1, 0), (0, float("nan"), 1)):
            with self.assertRaises(ValueError):
                video.sample_times(*params)

    def test_real_video_decoding_and_timestamps(self):
        info = video.video_info(self.path)
        self.assertEqual(info.frame_count, 16)
        frames = video.extract_frames(self.path, [0, 2, 3.75, 2])
        self.assertEqual([frame.timestamp_seconds for frame in frames], [0, 2, 3.75])
        self.assertTrue(all(cv2.imread(str(frame.path)) is not None for frame in frames))
        self.assertGreater(cv2.imread(str(frames[-1].path)).mean(), cv2.imread(str(frames[0].path)).mean())

    def test_counts_use_single_frame_maximum_not_union_or_sum(self):
        report = video.analyze_video_file(self.path, "How many types are visible together?", model=FakeVisionModel())
        self.assertEqual(report["observed_maximum"], 2)
        self.assertGreater(report["sampled_frames"], 3)
        self.assertLessEqual(report["largest_sample_gap_seconds"], 2)
        self.assertIn("not a proven maximum", report["limitations"])

    def test_successful_analysis_is_cached_but_new_question_is_not(self):
        with patch.object(video, "ChatOpenAI", return_value=FakeVisionModel()):
            first = video.analyze_video_file(self.path, "How many types are visible together?")
        with patch.object(video, "ChatOpenAI", side_effect=AssertionError("Unexpected model request")):
            cached = video.analyze_video_file(self.path, "How many types are visible together?", start_seconds=0.0, interval_seconds=2.0)
            self.assertEqual(first, cached)
            with self.assertRaises(AssertionError):
                video.analyze_video_file(self.path, "What color is the object?")

    def test_missing_frame_findings_fail_instead_of_implying_full_coverage(self):
        bad_model = SimpleNamespace(
            with_structured_output=lambda *args, **kwargs: SimpleNamespace(
                invoke=lambda messages: video.BatchFindings(findings=[])
            )
        )
        with self.assertRaisesRegex(RuntimeError, "omitted or duplicated"):
            video.analyze_video_file(self.path, "What is visible?", model=bad_model)

    def test_youtube_url_validation_and_local_path_boundary(self):
        self.assertEqual(video.youtube_id("https://youtu.be/L1vXCYZAYYM?t=4"), "L1vXCYZAYYM")
        self.assertEqual(video.youtube_id("https://www.youtube.com/watch?v=L1vXCYZAYYM"), "L1vXCYZAYYM")
        for url in ("https://youtube.com.evil.example/watch?v=L1vXCYZAYYM", "file:///tmp/video.mp4", "https://youtube.com/watch?v=bad"):
            with self.assertRaises(ValueError):
                video.youtube_id(url)
        with patch.object(video, "ATTACHMENTS", self.folder):
            self.assertEqual(video.resolve_video(str(self.path)), self.path.resolve())
            with self.assertRaises(ValueError):
                video.resolve_video(str(self.folder.parent / "outside.mp4"))

    def test_download_reuses_existing_course_file(self):
        cached = self.folder / "L1vXCYZAYYM.mp4"
        cached.write_bytes(b"existing video")
        with patch.object(video, "ATTACHMENTS", self.folder), patch.object(video, "YoutubeDL") as downloader:
            self.assertEqual(video.download_youtube("https://youtu.be/L1vXCYZAYYM"), cached)
            downloader.assert_not_called()

    def test_transcription_extracts_real_audio_and_offsets_segments(self):
        source = self.folder / "speech.wav"
        with wave.open(str(source), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\x00\x00" * 16000)
        def create(**kwargs):
            self.assertEqual(kwargs["response_format"], "verbose_json")
            self.assertGreater(len(kwargs["file"].read()), 100)
            return SimpleNamespace(model_dump=lambda: {"text": "Example speech.", "segments": [{"start": 0.1, "end": 0.3, "text": "Example speech."}]})
        client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
        report = video.transcribe_video_file(source, start_seconds=0.2, end_seconds=0.8, client=client)
        self.assertEqual(report["segments"][0]["start_seconds"], 0.3)
        self.assertEqual(report["segments"][0]["end_seconds"], 0.5)

    def test_tool_returns_download_error_as_evidence_not_answer(self):
        with patch.object(video, "resolve_video", side_effect=RuntimeError("Download unavailable")):
            report = json.loads(video.analyze_video.invoke({"source": "https://youtu.be/L1vXCYZAYYM", "question": "What is visible?"}))
        self.assertFalse(report["answer_available"])
        self.assertEqual(report["error"], "Download unavailable")


if __name__ == "__main__":
    unittest.main()
