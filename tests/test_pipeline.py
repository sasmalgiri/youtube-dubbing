import unittest
from unittest.mock import patch, MagicMock
from src.dubbing.pipeline import DubbingPipeline


class TestDubbingPipeline(unittest.TestCase):

    @patch('src.dubbing.pipeline.setup_logging')
    def test_initialization(self, mock_logging):
        mock_logging.return_value = MagicMock()
        pipeline = DubbingPipeline("https://example.com/video", "es")
        self.assertEqual(pipeline.video_url, "https://example.com/video")
        self.assertEqual(pipeline.target_language, "es")
        self.assertIsNone(pipeline.video_path)
        self.assertIsNone(pipeline.audio_path)
        self.assertIsNone(pipeline.transcription)
        self.assertIsNone(pipeline.translated_text)
        self.assertIsNone(pipeline.dubbed_audio_path)


if __name__ == '__main__':
    unittest.main()
