"""Quick Google Cloud TTS connectivity test."""
import os
import sys

print("=== Google TTS Connectivity Test ===")
print()

# Check credentials
creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
print(f"GOOGLE_APPLICATION_CREDENTIALS: {creds or '(not set)'}")
print(f"GOOGLE_CLOUD_PROJECT: {project or '(not set)'}")
print()

# Check library
try:
    from google.cloud import texttospeech
    print("google-cloud-texttospeech: INSTALLED")
except ImportError:
    print("google-cloud-texttospeech: NOT INSTALLED")
    print("Install with: pip install google-cloud-texttospeech")
    sys.exit(1)

print()
# Try a test synthesis
print("Attempting test synthesis (Hindi, short text)...")
try:
    client = texttospeech.TextToSpeechClient()
    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text="\u0928\u092e\u0938\u094d\u0924\u0947, \u092f\u0939 \u090f\u0915 \u092a\u0930\u0940\u0915\u094d\u0937\u0923 \u0939\u0948\u0964"),
        voice=texttospeech.VoiceSelectionParams(
            language_code="hi-IN",
            name="hi-IN-Neural2-A",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
        ),
    )
    audio_bytes = len(response.audio_content)
    print(f"SUCCESS! Got {audio_bytes} bytes of audio")
    # Save test file
    test_path = os.path.join("backend", "work", "google_tts_test.mp3")
    os.makedirs(os.path.join("backend", "work"), exist_ok=True)
    with open(test_path, "wb") as f:
        f.write(response.audio_content)
    print(f"Test audio saved to: {test_path}")
    print()
    print("RESULT: Google Cloud TTS is WORKING and ready as Edge-TTS fallback!")
except Exception as e:
    print(f"FAILED: {e}")
    print()
    print("Google TTS is NOT working. Check:")
    print("1. GOOGLE_APPLICATION_CREDENTIALS points to a valid service account JSON")
    print("2. The service account has Text-to-Speech API enabled")
    print("3. Billing is enabled on the GCP project")
    sys.exit(1)
