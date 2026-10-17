import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests
import httpx
from huggingface_hub.errors import GatedRepoError


class AttachmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unused-offline-test-key"}):
            cls.agent = importlib.import_module("agent")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        patcher = patch.object(self.agent, "BASE_DIR", self.base)
        patcher.start()
        self.addCleanup(patcher.stop)

    def missing_response(self):
        response = requests.Response()
        response.status_code = 404
        return response

    def test_local_attachment_needs_no_network(self):
        folder = self.base / "attachments"
        folder.mkdir()
        path = folder / "example.png"
        path.write_bytes(b"local image")
        with patch.object(self.agent.requests, "get") as get:
            result = self.agent.download_attachment("example-task", path.name)
        self.assertEqual(result, path)
        get.assert_not_called()

    def test_course_404_downloads_only_requested_official_file(self):
        original = self.base / "original.png"
        original.write_bytes(b"official image bytes")
        with patch.object(self.agent.requests, "get", return_value=self.missing_response()), patch.object(self.agent, "hf_hub_download", return_value=str(original)) as download:
            result = self.agent.download_attachment("example-task", "example.png")
        self.assertEqual(result.read_bytes(), original.read_bytes())
        self.assertEqual(download.call_args.kwargs["filename"], "2023/validation/example.png")
        self.assertEqual(download.call_args.kwargs["repo_id"], "gaia-benchmark/GAIA")

    def test_gated_access_explains_user_action(self):
        response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/datasets/example"))
        error = GatedRepoError("Access required", response=response)
        with patch.object(self.agent.requests, "get", return_value=self.missing_response()), patch.object(self.agent, "hf_hub_download", side_effect=error):
            with self.assertRaisesRegex(PermissionError, "Accept its conditions"):
                self.agent.download_attachment("example-task", "example.png")
        self.assertFalse((self.base / "attachments" / "example.png").exists())


if __name__ == "__main__":
    unittest.main()
