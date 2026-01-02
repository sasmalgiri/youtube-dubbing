from pydub import AudioSegment

class Mixer:
    def __init__(self, original_audio_path, dubbed_audio_path):
        self.original_audio = AudioSegment.from_file(original_audio_path)
        self.dubbed_audio = AudioSegment.from_file(dubbed_audio_path)

    def mix_audios(self, output_path, volume_adjustment=0):
        # Adjust volume if needed
        if volume_adjustment != 0:
            self.dubbed_audio = self.dubbed_audio + volume_adjustment
        
        # Mix the original and dubbed audio
        mixed_audio = self.original_audio.overlay(self.dubbed_audio)
        mixed_audio.export(output_path, format="mp3")

    def apply_effects(self, effect_function):
        self.dubbed_audio = effect_function(self.dubbed_audio)