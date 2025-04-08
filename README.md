# jarvis

Voice activated bot and task runner for my personal use. Right now I can do some Ollama API interaction but mostly for playing music, interacting with my music library via `mpd` (music player daemon) library.

## Models

Download the model you want from here: https://alphacephei.com/vosk/models - right now the code is using `vosk-model-small-en-us-0.15` and doesn't support configuration.

```bash
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -O vosk.zip
unzip vosk.zip -d model
rm vosk.zip
```