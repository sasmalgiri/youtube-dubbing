import unittest
from src.subtitles.srt import create_srt, parse_srt

class TestSubtitles(unittest.TestCase):

    def test_create_srt(self):
        subtitles = [
            (0, 2, "Hello, world!"),
            (3, 5, "Welcome to the video dubbing application.")
        ]
        expected_output = "1\n00:00:00,000 --> 00:00:02,000\nHello, world!\n\n" \
                          "2\n00:00:03,000 --> 00:00:05,000\nWelcome to the video dubbing application.\n\n"
        result = create_srt(subtitles)
        self.assertEqual(result, expected_output)

    def test_parse_srt(self):
        srt_content = "1\n00:00:00,000 --> 00:00:02,000\nHello, world!\n\n" \
                      "2\n00:00:03,000 --> 00:00:05,000\nWelcome to the video dubbing application.\n\n"
        expected_output = [
            (0, 2, "Hello, world!"),
            (3, 5, "Welcome to the video dubbing application.")
        ]
        result = parse_srt(srt_content)
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()