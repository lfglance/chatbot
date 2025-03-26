# Jarvis Microphone Listener with Local Speech-to-Text

A Go program that listens to microphone input, displays audio levels, and transcribes speech in real-time using the whisper.cpp library.

## Prerequisites

This program uses the whisper.cpp Go bindings for local speech recognition and requires:

1. **Audio dependencies**:
   ### Ubuntu/Debian
   ```
   sudo apt-get install libasound2-dev
   ```

   ### macOS
   No additional dependencies needed.

   ### Windows
   No additional dependencies needed.

2. **Whisper.cpp model**:
   - Download a model file from https://huggingface.co/ggerganov/whisper.cpp/tree/main
   - Recommended: `ggml-tiny.en.bin` (English only, smallest model) or `ggml-tiny.bin` (multilingual)
   - Place the downloaded model in the `models` directory (will be created automatically)

## Installation

1. Clone this repository
2. Install Go dependencies:
```
go mod download
```
3. Download the Whisper model:
```
mkdir -p models
curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin -o models/ggml-tiny.en.bin
```

## Usage

Run the program:
```
go run main.go
```

The program will:
- Listen to your microphone
- Display audio levels when sound is detected
- When significant audio is detected, process it with the Whisper model
- Display the transcribed text
- Run until you press Ctrl+C to stop it

## How It Works

1. The program captures audio from your microphone using the `malgo` library
2. Audio is buffered in memory until there's a significant amount of speech
3. When speech is detected, the audio buffer is converted to the format required by Whisper
4. The Whisper model processes the audio directly in memory and returns the transcription
5. The transcription is displayed in the console

## Customization

You can modify the code to:
- Adjust the sensitivity of speech detection by changing the `silenceThreshold` constant
- Change the audio buffer size by modifying the `bufferSeconds` constant
- Save transcriptions to a file
- Implement commands or actions based on recognized speech
- Use different Whisper model variants (tiny, base, small, medium, large)