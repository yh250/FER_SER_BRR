# audio_processor.py

import numpy as np
import sounddevice as sd  # Still needed for callback structure, but won't start live stream
from transformers import pipeline
import time
from collections import deque
import os


class AudioProcessor:
    def __init__(self, samplerate, ser_process_interval, history_length, use_live_mic=True):
        self.samplerate = samplerate
        self.ser_process_interval = ser_process_interval
        self.use_live_mic = use_live_mic
        self.ser_pipeline = None
        self.current_audio_chunk = deque()
        self.total_audio_frames_collected = 0
        self.last_ser_process_time = time.time()
        self.last_speech_emotion = "Initializing..."
        self.speech_emotion_history = deque(maxlen=history_length)
        self.audio_stream = None  # For live mic only

        self._load_ser_model()
        if self.use_live_mic:
            self._start_audio_stream()

    def _load_ser_model(self):
        print("Loading Speech Emotion Recognition model...")
        self.ser_pipeline = pipeline("audio-classification", model="r-f/wav2vec-english-speech-emotion-recognition")
        print("Speech Emotion Recognition model loaded.")

    def _start_audio_stream(self):
        try:
            self.audio_stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                callback=self._audio_input_callback
            )
            self.audio_stream.start()
            print("Live audio stream started.")
        except Exception as e:
            print(f"Error starting live audio stream: {e}. Falling back to no live audio.")
            self.use_live_mic = False  # Disable live mic if it fails

    def _audio_input_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio Callback Warning: {status}")
        self.current_audio_chunk.append(indata[:, 0].copy())
        self.total_audio_frames_collected += frames

    def add_audio_data(self, audio_data):
        """Adds pre-extracted audio data (e.g., from a video file) for processing."""
        # Split into chunks compatible with how the live stream would deliver it
        chunk_size = int(self.samplerate * 0.1)  # Process in smaller chunks for SER
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) > 0:
                self.current_audio_chunk.append(chunk)
                self.total_audio_frames_collected += len(chunk)

    def process_audio(self):
        # Always process, but only if enough audio has accumulated
        if self.total_audio_frames_collected >= int(self.samplerate * self.ser_process_interval):
            if self.current_audio_chunk:
                combined_audio_for_ser = np.concatenate(list(self.current_audio_chunk))
                self.current_audio_chunk.clear()  # Clear buffer
                self.total_audio_frames_collected = 0
                self.last_ser_process_time = time.time()  # Reset timer (even if not live)

                try:
                    ser_results = self.ser_pipeline({"raw": combined_audio_for_ser, "sampling_rate": self.samplerate})
                    if ser_results:
                        print(f"Raw SER Results: {ser_results}")
                        dominant_speech_emotion = ser_results[0]['label']
                        self.last_speech_emotion = dominant_speech_emotion
                    else:
                        self.last_speech_emotion = "No Speech Emotion"
                except Exception as e:
                    self.last_speech_emotion = "Error"
                    #----------------------------------------------------------
                    print(f"SER pipeline error: {e}") # Uncomment for debugging
            else:
                self.last_speech_emotion = "No Speech Input"  # Should not happen if `total_audio_frames_collected` check passes

        self.speech_emotion_history.append(self.last_speech_emotion)
        return self.last_speech_emotion, list(self.speech_emotion_history)

    def stop(self):
        if self.use_live_mic and self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            print("Live audio stream stopped.")


if __name__ == '__main__':
    # Test for AudioProcessor
    from config import AUDIO_SAMPLERATE, SER_PROCESS_INTERVAL, RECENT_EMOTION_HISTORY_LENGTH

    # Test with live mic (default)
    # processor = AudioProcessor(AUDIO_SAMPLERATE, SER_PROCESS_INTERVAL, RECENT_EMOTION_HISTORY_LENGTH, use_live_mic=True)

    # Test with dummy pre-loaded audio (e.g., if extracted from video)
    # Generate some dummy audio (e.g., 5 seconds of random noise)
    dummy_audio = np.random.randn(AUDIO_SAMPLERATE * 5).astype(np.float32)
    processor = AudioProcessor(AUDIO_SAMPLERATE, SER_PROCESS_INTERVAL, RECENT_EMOTION_HISTORY_LENGTH,
                               use_live_mic=False)
    processor.add_audio_data(dummy_audio)

    print("Testing audio processor for 10 seconds...")
    for _ in range(20):
        emotion, history = processor.process_audio()
        print(f"Current Speech Emotion: {emotion}, History: {list(history)}")
        time.sleep(0.5)
    processor.stop()