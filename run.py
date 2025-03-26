import json
from pathlib import Path

import pyaudio
import requests
import subprocess
import numpy as np
from fuzzywuzzy import fuzz
from vosk import Model, KaldiRecognizer


# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Ollama API settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"

# Conversation context
conversation_history = []
INITIAL_PROMPT = "You are a helpful assistant. Respond naturally and maintain context from previous messages. Keep your responses very brief. Be as concise as possible. Only use as few words as necessary. Laconic."

# Music folders
MUSIC_DIR = Path("~/Music").expanduser()

# Load Vosk model
model = Model("model/vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(model, RATE)

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Collect audio file metadata
audio_metadata = {}
for f in MUSIC_DIR.rglob('*.*'):
    if f.is_file() and f.name.endswith((".mp3", ".flac", ".m4a", ".opus")):
        artist = f.parent.parent.name
        album = f.parent.name
        track = f.absolute()
        if artist not in audio_metadata:
            audio_metadata[artist] = {}
        if album not in audio_metadata[artist]:
            audio_metadata[artist][album] = []
        audio_metadata[artist][album].append(str(track))

def find_artist(query) -> str|bool:
    top = ("", 0)
    threshold = 80
    for i in audio_metadata.keys():
        score = fuzz.ratio(query.lower(), i.lower())
        if score > top[1]:
            top = (i, score)
    if top[1] > threshold:
        return top[0]
    else:
        return None

def find_song(artist, query) -> str|bool:
    top = ("", 0)
    artist = find_artist(artist)
    if not artist:
        return None
    for album in audio_metadata[artist].keys():
        for track in audio_metadata[artist][album]:
            name = " ".join(track.split(".")[0:-1])
            score = fuzz.ratio(query.lower(), name.lower())
            if score > top[1]:
                top = (track, score)
    return top[0]

# Function to query Ollama API with context
def query_ollama(text, is_first_run=False):
    global conversation_history

    if is_first_run:
        conversation_history.append({"role": "system", "content": INITIAL_PROMPT})

    conversation_history.append({"role": "user", "content": text})
    full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

    payload = {
        "model": MODEL_NAME,
        "prompt": text,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        ollama_response = result.get("response", "No response from Ollama")
        conversation_history.append({"role": "assistant", "content": ollama_response})
        return ollama_response
    except requests.RequestException as e:
        return f"Error querying Ollama: {e}"

print("Listening... (Ctrl+C to stop)\n")

first_run = True
pending_text = None

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            transcribed_text = result.get("text")
            if transcribed_text:
                pending_text = transcribed_text

        if pending_text:
            print("=" * 50, "\n> ", pending_text)
            parts = pending_text.split()
            if pending_text == "clear":
                conversation_history = []
                print("----- cleared session context -----")
            elif parts[0] == "maestro":
                if parts[1] == "play":
                    if parts[2] == "music":
                        # play all tracks from the artist
                        pass
                    else:
                        query = " ".join(parts[2:])
                        subquery = query.split(" by ")
                        song = subquery[0]
                        artist = subquery[1]
                        res = find_song(artist, song)
                        print(f"Playing {res}")
                        subprocess.Popen(["vlc", res])
            else:
                ollama_response = query_ollama(pending_text, is_first_run=first_run)
                print(ollama_response)

            if first_run:
                first_run = False

            pending_text = None

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

print("\nConversation History:")
for msg in conversation_history:
    print(f"{msg['role']}: {msg['content']}")