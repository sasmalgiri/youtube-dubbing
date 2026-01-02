from typing import List
import speech_recognition as sr

class Transcriber:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def transcribe_audio(self, audio_file: str) -> List[dict]:
        with sr.AudioFile(audio_file) as source:
            audio_data = self.recognizer.record(source)
            try:
                transcription = self.recognizer.recognize_google(audio_data, show_all=True)
                return self._process_transcription(transcription)
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                return []
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                return []

    def _process_transcription(self, transcription: dict) -> List[dict]:
        results = []
        for result in transcription.get('alternative', []):
            results.append({
                'transcript': result.get('transcript'),
                'confidence': result.get('confidence', 0)
            })
        return results
