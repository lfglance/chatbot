package main

import (
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/gen2brain/malgo"
	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"github.com/go-audio/wav"
)

const (
	sampleRate      = 16000 // Whisper works best with 16kHz sample rate
	bufferSeconds   = 3     // Buffer 3 seconds of audio at a time
	audioBufferSize = sampleRate * bufferSeconds
	silenceThreshold = 500  // Threshold for detecting speech
	modelPath      = "models/ggml-tiny.en.bin" // Path to Whisper model file
)

// AudioBuffer holds audio data and handles operations on it
type AudioBuffer struct {
	buffer    []int16
	mutex     sync.Mutex
	threshold int64
}

// NewAudioBuffer creates a new audio buffer
func NewAudioBuffer(capacity int) *AudioBuffer {
	return &AudioBuffer{
		buffer:    make([]int16, 0, capacity),
		threshold: silenceThreshold,
	}
}

// Add appends new samples to the buffer
func (ab *AudioBuffer) Add(samples []int16) {
	ab.mutex.Lock()
	defer ab.mutex.Unlock()
	ab.buffer = append(ab.buffer, samples...)
}

// GetAudioLevel returns the average amplitude of the buffer
func (ab *AudioBuffer) GetAudioLevel() int64 {
	ab.mutex.Lock()
	defer ab.mutex.Unlock()

	if len(ab.buffer) == 0 {
		return 0
	}

	var sum int64
	for _, sample := range ab.buffer {
		if sample < 0 {
			sum -= int64(-sample)
		} else {
			sum += int64(sample)
		}
	}
	return sum / int64(len(ab.buffer))
}

// HasSignificantAudio checks if there's significant audio above threshold
func (ab *AudioBuffer) HasSignificantAudio() bool {
	return ab.GetAudioLevel() > ab.threshold
}

// GetAudioSamples returns a copy of the buffer and clears it
func (ab *AudioBuffer) GetAudioSamples() []int16 {
	ab.mutex.Lock()
	defer ab.mutex.Unlock()

	// Make a copy of the buffer
	samples := make([]int16, len(ab.buffer))
	copy(samples, ab.buffer)

	// Clear the original buffer
	ab.buffer = ab.buffer[:0]

	return samples
}

// Convert int16 PCM to float32 (required by Whisper)
func convertPCMToFloat32(samples []int16) []float32 {
	floatSamples := make([]float32, len(samples))
	for i, sample := range samples {
		floatSamples[i] = float32(sample) / 32768.0 // Normalize to [-1.0, 1.0]
	}
	return floatSamples
}

// SaveWavFile saves the audio buffer to a WAV file
func SaveWavFile(samples []int16, filePath string) error {
	// Create the directory if it doesn't exist
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	// Open the output file
	out, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer out.Close()

	// Create a new WAV encoder
	encoder := wav.NewEncoder(out, sampleRate, 16, 1, 1)
	defer encoder.Close()

	// Convert int16 samples to int
	intSamples := make([]int, len(samples))
	for i, s := range samples {
		intSamples[i] = int(s)
	}

	// Write the samples to the WAV file
	if err := encoder.Write(intSamples); err != nil {
		return err
	}

	return nil
}

// WhisperModel represents the whisper.cpp model
type WhisperModel struct {
	model    whisper.Model
	context  whisper.Context
}

// NewWhisperModel creates a new Whisper model
func NewWhisperModel(modelPath string) (*WhisperModel, error) {
	// Check if model file exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model file not found: %s", modelPath)
	}

	// Load whisper model
	model, err := whisper.New(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load whisper model: %v", err)
	}

	// Create whisper context
	context, err := model.NewContext()
	if err != nil {
		model.Close()
		return nil, fmt.Errorf("failed to create whisper context: %v", err)
	}

	return &WhisperModel{
		model:   model,
		context: context,
	}, nil
}

// Close releases resources used by the model
func (wm *WhisperModel) Close() {
	if wm.context != nil {
		wm.context.Free()
	}
	if wm.model != nil {
		wm.model.Close()
	}
}

// Transcribe performs speech recognition on audio data
func (wm *WhisperModel) Transcribe(samples []float32) (string, error) {
	// Process the audio data with Whisper
	if err := wm.context.Process(samples, nil); err != nil {
		return "", fmt.Errorf("failed to process audio: %v", err)
	}

	// Get the number of segments
	numSegments := wm.context.NumSegments()
	if numSegments == 0 {
		return "", nil
	}

	// Combine all segments into a single transcript
	var transcript string
	for i := 0; i < numSegments; i++ {
		segment, err := wm.context.GetSegment(i)
		if err != nil {
			return transcript, fmt.Errorf("failed to get segment %d: %v", i, err)
		}
		transcript += segment.Text
	}

	return transcript, nil
}

func main() {
	// Create models directory if it doesn't exist
	if err := os.MkdirAll("models", 0755); err != nil {
		fmt.Println("Error creating models directory:", err)
		return
	}

	// Check if model file exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Println("Model file not found:", modelPath)
		fmt.Println("Please download the model file from https://huggingface.co/ggerganov/whisper.cpp/tree/main")
		fmt.Println("and place it in the models directory as 'ggml-tiny.en.bin'")
		return
	}

	// Create temp directory for audio files
	tempDir := filepath.Join(os.TempDir(), "jarvis-audio")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		fmt.Println("Error creating temp directory:", err)
		return
	}
	defer os.RemoveAll(tempDir) // Clean up when done

	// Set up channel to handle termination signals
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)

	// Initialize audio buffer
	audioBuffer := NewAudioBuffer(audioBufferSize)

	// Initialize Whisper model
	whisperModel, err := NewWhisperModel(modelPath)
	if err != nil {
		fmt.Println("Error initializing Whisper model:", err)
		return
	}
	defer whisperModel.Close()

	// Initialize malgo context
	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, nil)
	if err != nil {
		fmt.Println("Error initializing malgo context:", err)
		return
	}
	defer func() {
		_ = ctx.Uninit()
		ctx.Free()
	}()

	// Device configuration for audio capture
	deviceConfig := malgo.DefaultDeviceConfig(malgo.Capture)
	deviceConfig.Capture.Format = malgo.FormatS16
	deviceConfig.Capture.Channels = 1
	deviceConfig.SampleRate = sampleRate
	deviceConfig.Alsa.NoMMap = 1

	// Create a channel for communicating when audio should be processed
	processAudioChan := make(chan struct{}, 1)
	transcriptionChan := make(chan string, 10)
	errorChan := make(chan error, 10)

	// Create a ticker to periodically check for significant audio
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	// Callback function for processing audio data
	onRecvFrames := func(outputSamples, inputSamples []byte, frameCount uint32) {
		// We only care about input samples since we're only capturing
		if len(inputSamples) == 0 {
			return
		}

		// Convert byte array to int16 samples manually
		samples := make([]int16, len(inputSamples)/2)
		for i := 0; i < len(inputSamples); i += 2 {
			if i+1 < len(inputSamples) {
				// Little-endian: LSB first, then MSB
				samples[i/2] = int16(uint16(inputSamples[i]) | uint16(inputSamples[i+1])<<8)
			}
		}

		// Add samples to our audio buffer
		audioBuffer.Add(samples)

		// Get current audio level (for visualization)
		level := audioBuffer.GetAudioLevel()
		if level > silenceThreshold {
			fmt.Printf("Audio level: %d\n", level)
		}
	}

	// Initialize device
	captureCallbacks := malgo.DeviceCallbacks{
		Data: onRecvFrames,
	}
	device, err := malgo.InitDevice(ctx.Context, deviceConfig, captureCallbacks)
	if err != nil {
		fmt.Println("Error initializing device:", err)
		return
	}
	defer device.Uninit()

	// Start the device
	err = device.Start()
	if err != nil {
		fmt.Println("Error starting device:", err)
		return
	}

	fmt.Println("Listening to microphone. Press Ctrl+C to stop.")
	fmt.Println("Speaking will be transcribed using Whisper.cpp")

	// Check if we have significant audio and should process it
	go func() {
		for range ticker.C {
			if audioBuffer.HasSignificantAudio() {
				select {
				case processAudioChan <- struct{}{}:
					// Signal sent successfully
				default:
					// Channel is full, which means we're already processing
				}
			}
		}
	}()

	// Process audio when signaled
	go func() {
		for range processAudioChan {
			samples := audioBuffer.GetAudioSamples()
			if len(samples) == 0 {
				continue
			}

			// For debugging, save a WAV file
			audioFile := filepath.Join(tempDir, fmt.Sprintf("audio_%d.wav", time.Now().UnixNano()))
			if err := SaveWavFile(samples, audioFile); err != nil {
				errorChan <- fmt.Errorf("error saving WAV file: %v", err)
			}

			// Convert to float32 for Whisper
			floatSamples := convertPCMToFloat32(samples)

			// Transcribe using whisper.cpp
			go func(samples []float32, filePath string) {
				transcript, err := whisperModel.Transcribe(samples)
				if err != nil {
					errorChan <- fmt.Errorf("error transcribing audio: %v", err)
					return
				}

				if transcript != "" {
					transcriptionChan <- transcript
				}
			}(floatSamples, audioFile)
		}
	}()

	// Main loop to display transcriptions and errors
	go func() {
		for {
			select {
			case transcript := <-transcriptionChan:
				if transcript != "" {
					fmt.Printf("\nTranscription: %s\n", transcript)
				}
			case err := <-errorChan:
				fmt.Printf("Error: %v\n", err)
			}
		}
	}()

	// Wait for termination signal
	<-sigs
	fmt.Println("\nStopping audio capture...")
}