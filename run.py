import json
from pathlib import Path

import pyaudio
import requests
import subprocess
import numpy as np
import pyttsx3 as tts
from thefuzz import fuzz
from vosk import Model, KaldiRecognizer


# Audio settings
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
AMP_THRESHOLD = 600

# Ollama API settings
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:1b"
BOT_NAME = "jimbo"

# Conversation context
SYSTEM_PROMPT = f"Your name is {BOT_NAME}. You are a helpful assistant. Keep your responses very brief. Be as concise as possible. Only use as few words as necessary. Laconic."

# Load Vosk model
model = Model("model/vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(model, RATE)

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def find_artist(query) -> str|bool:
    top = ("", 0)
    threshold = 0
    result = subprocess.run(
        ["mpc", "list", "artist"],
        capture_output=True,
        text=True,
        check=True
    )
    artists = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for i in artists:
        score = fuzz.ratio(query.lower(), i.lower())
        if score > top[1]:
            top = (i, score)
    if top[1] > threshold:
        return top[0]
    else:
        return None

def find_any(query) -> str|bool:
    top = ("", 0)
    result = subprocess.run(
        ["mpc", "list", "title"],
        capture_output=True,
        text=True,
        check=True
    )
    tracks = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    print(tracks)
    for track in tracks:
        score = fuzz.ratio(query.lower(), track.lower())
        if score > top[1]:
            top = (track, score)
    return top[0]

def find_song(artist, query) -> str|bool:
    top = ("", 0)
    artist = find_artist(artist)
    if not artist:
        return None
    result = subprocess.run(
        ["mpc", "list", "title", "artist", artist],
        capture_output=True,
        text=True,
        check=True
    )
    tracks = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for track in tracks:
        score = fuzz.ratio(query.lower(), track.lower())
        if score > top[1]:
            top = (track, score)
    return top[0]

def query_ollama(user_prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        ollama_response = result.get("message", "No response from Ollama")
        return ollama_response["content"]
    except requests.RequestException as e:
        return f"Error querying Ollama: {e}"
    except Exception as e:
        return e

print("Listening... (Ctrl+C to stop)\n")

try:
    pending_text = None
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        # audio_data = np.frombuffer(data, dtype=np.int16)
        # amp = np.max(np.abs(audio_data))
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            transcribed_text = result.get("text")
            if transcribed_text:
                pending_text = transcribed_text

        if pending_text:
            print("\n> ", pending_text)
            parts = pending_text.split()
            try:
                if pending_text == "clear":
                    conversation_history = []
                    print("\n----- cleared session context -----\n")
                elif pending_text == "volume up":
                    subprocess.run(["mpc", "volume", "100"])
                elif pending_text == "volume down":
                    subprocess.run(["mpc", "volume", "60"])
                elif pending_text == "stop":
                    subprocess.run(["mpc", "stop"])
                elif pending_text == "pause":
                    subprocess.run(["mpc", "pause"])
                elif pending_text == "play" or pending_text == "resume":
                    subprocess.run(["mpc", "play"])
                elif pending_text == "shuffle all songs":
                    subprocess.run(["mpc", "clear"])
                    subprocess.run(["mpc", "add", "/"])
                    subprocess.run(["mpc", "shuffle"])
                    subprocess.run(["mpc", "play"])
                elif pending_text == "skip":
                    subprocess.run(["mpc", "next"])
                elif pending_text == "rewind" or pending_text == "go back":
                    subprocess.run(["mpc", "prev"])
                elif parts[0].startswith("play"):
                    subquery = " ".join(parts[1:]).split(" by ")
                    if len(subquery) > 1:
                        title = subquery[0]
                        artist = find_artist(subquery[1])
                        res = find_song(artist, title)
                        subprocess.run(["mpc", "clear"])
                        subprocess.run(["mpc", "findadd", "artist", artist, "title", res])
                        subprocess.run(["mpc", "play"])
                    else:
                        res = find_any(" ".join(parts[1:]))
                        subprocess.run(["mpc", "clear"])
                        subprocess.run(["mpc", "findadd", "title", res])
                        subprocess.run(["mpc", "play"])
                elif parts[0] == "shuffle":
                    split = " ".join(parts[1:]).split(" by ")
                    artist = find_artist(" ".join(split[1:]))
                    subprocess.run(["mpc", "clear"])
                    subprocess.run(["mpc", "findadd", "artist", artist])
                    subprocess.run(["mpc", "play"])
                elif parts[0] == BOT_NAME:
                    ollama_response = query_ollama(pending_text)
                    print("\n" + ollama_response)
                print()
            except Exception as e:
                print(e)
                pass

        pending_text = None

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
