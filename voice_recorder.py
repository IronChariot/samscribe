import keyboard
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import time
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pyperclip
import pygame

# Global variables
is_recording = False
fs = 44100  # Sample rate
channels = 1  # Mono recording
recording = []
stream = None

def play_wav(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    
    # Wait for the music to play before exiting
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def transcribe_with_whisper(filename):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(filename)
    transcription = result["text"]
    return transcription

def audio_callback(indata, frames, time_info, status):
    global recording
    if status:
        print(f"Error: {status}", flush=True)
    recording.append(indata.copy())

def start_recording():
    global is_recording, recording, stream
    if not is_recording:
        is_recording = True
        recording = []
        # Start the audio stream in a separate thread
        stream = sd.InputStream(samplerate=fs, channels=channels, callback=audio_callback)
        stream.start()
        print("Recording started...")

def stop_recording():
    global is_recording, recording, stream
    if is_recording:
        is_recording = False
        stream.stop()
        stream.close()
        # Concatenate all recorded chunks
        audio_data = np.concatenate(recording, axis=0)
        # Generate a filename with a timestamp
        filename = time.strftime("%Y%m%d-%H%M%S") + '.wav'
        # Save the audio data to a .wav file
        write(filename, fs, audio_data)
        print(f"Recording saved as {filename}")
        transcription = transcribe_with_whisper(filename)
        # Put the transcription onto the clipboard
        pyperclip.copy(transcription)
        print(f"Transcription copied to clipboard: {transcription}")
        os.remove(filename)  # Delete the temporary audio file
        # Play the beep.wav sound file
        beep_sound = 'beep.wav'
        play_wav(beep_sound)

def main():
    # Register key event handlers for F14 key
    keyboard.on_press_key('f14', lambda e: start_recording())
    keyboard.on_release_key('f14', lambda e: stop_recording())
    
    # Keep the script running indefinitely
    print("Press F14 to start recording...")
    keyboard.wait()

if __name__ == "__main__":
    main()
