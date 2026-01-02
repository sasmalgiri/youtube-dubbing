import unittest
from src.dubbing.pipeline import VideoDubbingPipeline

class TestVideoDubbingPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = VideoDubbingPipeline()

    def test_initialization(self):
        self.assertIsNotNone(self.pipeline)

    def test_process_video(self):
        result = self.pipeline.process_video("sample_video.mp4", "en", "es")
        self.assertTrue(result['success'])
        self.assertIn('dubbed_audio', result)
        self.assertIn('subtitles', result)

    def test_invalid_video(self):
        result = self.pipeline.process_video("invalid_video.mp4", "en", "es")
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'Video not found')

    def test_language_translation(self):
        result = self.pipeline.translate_text("Hello, world!", "en", "es")
        self.assertEqual(result, "Hola, mundo!")

if __name__ == '__main__':
    unittest.main()