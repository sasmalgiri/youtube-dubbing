from gtts import gTTS
import os

class Synthesizer:
    def __init__(self, language='en'):
        self.language = language

    def synthesize(self, text, output_file):
        tts = gTTS(text=text, lang=self.language, slow=False)
        tts.save(output_file)

    def synthesize_from_file(self, input_file, output_file):
        with open(input_file, 'r') as file:
            text = file.read()
        self.synthesize(text, output_file)