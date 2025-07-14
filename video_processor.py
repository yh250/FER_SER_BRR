# video_processor.py

import cv2
from deepface import DeepFace
from collections import deque
import numpy as np
import time
import librosa
import soundfile as sf
import os # Import os for os.path.exists

# Import VIDEO_LOOPING_ENABLED from config
from config import VIDEO_LOOPING_ENABLED # <--- ADD THIS IMPORT

class VideoProcessor:
    # Remove 'should_loop' from init, use config directly
    def __init__(self, source, history_length):
        self.source = source
        self.history_length = history_length
        self.should_loop = VIDEO_LOOPING_ENABLED # <--- Use the config value here

        self.cap = None
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0

        self.last_facial_emotion = "Initializing..."
        self.facial_emotion_history = deque(maxlen=history_length)

        self._initialize_capture()
        self._load_deepface_model()

    def _initialize_capture(self):
        # ... (no change here from previous version) ...
        if isinstance(self.source, int):  # Live camera
            self.cap = cv2.VideoCapture(self.source)
        elif isinstance(self.source, str):  # Video file path
            if not os.path.exists(self.source):
                raise FileNotFoundError(f"Video file not found: {self.source}")
            self.cap = cv2.VideoCapture(self.source)
            print(f"Video file '{self.source}' opened successfully.")
        else:
            raise ValueError("Invalid video source. Must be camera index (int) or file path (str).")

        if not self.cap.isOpened():
            raise IOError(f"Could not open video source: {self.source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    def _load_deepface_model(self):
        # ... (no change here) ...
        print("Loading DeepFace Facial Emotion model (this may take a moment)...")
        try:
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False, silent=True)
            print("DeepFace model loaded.")
        except Exception as e:
            print(f"Error loading DeepFace model: {e}")
            print("DeepFace analysis may fail later. Ensure dlib/opencv dependencies are met.")


    def process_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            if self.should_loop: # This logic remains the same, but 'self.should_loop' now comes from config
                print(f"End of video file: {self.source}. Looping...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    print(f"Error: Could not loop video or read first frame after reset: {self.source}")
                    return None, self.last_facial_emotion, list(self.facial_emotion_history)
            else:
                return None, self.last_facial_emotion, list(self.facial_emotion_history)

        # ... (rest of the DeepFace analysis and history update, no change) ...
        current_facial_emotion = "No Face"
        try:
            demographies = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            if demographies and isinstance(demographies, list) and len(demographies) > 0:
                dominant_emotion = demographies[0]['dominant_emotion']
                current_facial_emotion = dominant_emotion
                #-----------debug_print----------------
                print(f"VideoProc: Facial Emotion - {current_facial_emotion}, Scores: {demographies[0]['emotion']}")

            else:
                current_facial_emotion = "No Face"
        except ValueError as e:
            if "No facial detections" in str(e):
                current_facial_emotion = "No Face"
            else:
                current_facial_emotion = "Error"
                #----------debug_print----------
                print(f"VideoProc: DeepFace analysis error: {e}")

        except Exception as e:
            current_facial_emotion = "Error"
            #--------------debug_print------------
            print(f"VideoProc: Unexpected DeepFace error: {e}")


        self.last_facial_emotion = current_facial_emotion
        self.facial_emotion_history.append(current_facial_emotion)

        return frame, current_facial_emotion, list(self.facial_emotion_history)

    def get_audio_from_video(self, target_samplerate):
        # ... (no change here) ...
        """Extracts audio from the video file using librosa."""
        if not isinstance(self.source, str):
            print("Audio extraction only supported for video files.")
            return None

        print(f"Extracting audio from '{self.source}'...")
        try:
            y, sr = librosa.load(self.source, sr=target_samplerate, mono=True)
            print("Audio extraction complete.")
            return y
        except Exception as e:
            print(f"Error extracting audio from video: {e}")
            return None

    def stop(self):
        # ... (no change here) ...
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("Video capture released.")

    def get_frame_properties(self):
        # ... (no change here) ...
        return self.fps, self.frame_width, self.frame_height