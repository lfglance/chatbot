import pyaudio
from vosk import Model, KaldiRecognizer
import json

# Audio settings
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Load Vosk model
model = Model("model/vosk-model-small-en-us-0.15")  # Path to your model
rec = KaldiRecognizer(model, RATE)

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Listening... (Ctrl+C to stop)")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            print(rec.Result())
            result = json.loads(rec.Result())
            if result.get("text"):
                print("Transcription:", result["text"])
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()