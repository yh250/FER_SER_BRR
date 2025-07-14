# config.py

# --- System Configuration ---
CAMERA_INDEX = 0 # Only used if USE_LIVE_CAMERA is True
VIDEO_FILE_PATH = "/Users/harshyadav/Desktop/FER_SER_BRRR/sad.mp4" # <--- IMPORTANT: Update this with your video file
USE_LIVE_CAMERA = False # Set to True for live webcam, False for video file

AUDIO_SAMPLERATE = 16000
AUDIO_CHUNK_DURATION = 0.5  # seconds per audio buffer to process
SER_PROCESS_INTERVAL = 5.0  # Process SER every X seconds of audio

# --- Video Playback Configuration ---
VIDEO_LOOPING_ENABLED = False

# --- Authenticity Threshold ---
AUTHENTICITY_THRESHOLD = 0.5 # Adjust based on testing

# --- History Length for Temporal Checks ---
RECENT_EMOTION_HISTORY_LENGTH = 20 # Number of recent emotion frames to store for temporal checks

# --- Emotion Mapping for Congruence ---
# Define which emotions are considered congruent across modalities
EMOTION_CONGRUENCE = {
    "happy": {"happy"},
    "sad": {"sad", "calm", "neutral"},
    "angry": {"angry"},
    "fear": {"fearful", "surprised"}, # Fear and surprise often co-occur or have similar arousal
    "disgust": {"disgust"},
    "surprise": {"surprised", "fearful"},
    "neutral": {"neutral", "calm","sad"}
}