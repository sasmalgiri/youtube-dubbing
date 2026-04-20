"""Quick Sarvam Bulbul v3 TTS connectivity test."""
import os
import sys
import base64
import requests
from dotenv import load_dotenv

# Load .env from backend/
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

print("=== Sarvam Bulbul v3 TTS Connectivity Test ===")
print()

# Check API keys
keys = []
main = os.environ.get("SARVAM_API_KEY", "").strip()
if main:
    keys.append(("SARVAM_API_KEY", main))
for i in range(2, 20):
    k = os.environ.get(f"SARVAM_API_KEY_{i}", "").strip()
    if k:
        keys.append((f"SARVAM_API_KEY_{i}", k))

if not keys:
    print("NO SARVAM API KEYS FOUND in environment!")
    print("Set SARVAM_API_KEY in backend/.env")
    print("Get free key at https://dashboard.sarvam.ai")
    sys.exit(1)

print(f"Found {len(keys)} Sarvam API key(s)")
for name, key in keys:
    print(f"  {name}: {key[:8]}...{key[-4:]}")
print()

# Test synthesis with first key
api_key = keys[0][1]
test_text = "\u0928\u092e\u0938\u094d\u0924\u0947, \u092f\u0939 \u090f\u0915 \u092a\u0930\u0940\u0915\u094d\u0937\u0923 \u0939\u0948\u0964"
print(f"Test text: {test_text}")
print(f"Calling Sarvam Bulbul v3 API...")
print()

try:
    resp = requests.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "text": test_text,
            "target_language_code": "hi-IN",
            "speaker": "shubh",
            "model": "bulbul:v3",
            "pace": 1.0,
            "temperature": 0.6,
            "speech_sample_rate": 24000,
            "output_audio_codec": "mp3",
        },
        timeout=30,
    )

    print(f"HTTP Status: {resp.status_code}")

    if resp.status_code == 429:
        print("QUOTA EXHAUSTED on this key. Try next key or wait.")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"ERROR: {resp.text[:300]}")
        sys.exit(1)

    result = resp.json()
    audio_b64 = result.get("audios", [None])[0]
    if not audio_b64:
        print("ERROR: API returned empty audio")
        sys.exit(1)

    audio_bytes = base64.b64decode(audio_b64)
    print(f"SUCCESS! Got {len(audio_bytes)} bytes of audio")

    # Save test file
    os.makedirs("backend/work", exist_ok=True)
    test_path = "backend/work/sarvam_tts_test.mp3"
    with open(test_path, "wb") as f:
        f.write(audio_bytes)
    print(f"Test audio saved to: {test_path}")
    print()
    print("RESULT: Sarvam Bulbul v3 is WORKING and ready as Edge-TTS fallback!")

except requests.exceptions.Timeout:
    print("TIMEOUT: Sarvam API did not respond within 30s")
    sys.exit(1)
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
