# authenticity_analyzer.py

from collections import Counter # <--- THIS LINE SHOULD BE AT THE TOP LEVEL
# You might also have other imports here, e.g., import numpy as np if you use it

class AuthenticityAnalyzer:
    def __init__(self, emotion_congruence_map, authenticity_threshold):
        self.emotion_congruence_map = emotion_congruence_map
        self.authenticity_threshold = authenticity_threshold

    def analyze(self, facial_emotion, speech_emotion, facial_emotion_history, speech_emotion_history):
        # --- DEBUG: Initial State ---
        print(f"\n--- Authenticity Analysis for Frame ---")
        print(f"Auth Input: Face='{facial_emotion}', Speech='{speech_emotion}'")
        print(f"Auth History: Face={list(facial_emotion_history)}, Speech={list(speech_emotion_history)}")

        authenticity_score = 1.0

        print(f"Auth DEBUG: Initial score: {authenticity_score:.2f}")


        # 1. Penalize persistent "Error" or "No Face/No Speech Input"
        # This block should be carefully integrated into your existing penalty logic.
        # Ensure it applies a strong penalty if a modality is consistently unavailable.
        if facial_emotion == "Error" or facial_emotion == "No Face":
            # Count errors/no face in history
            error_face_count = sum(1 for e in facial_emotion_history if e in ["Error", "No Face", "Initializing..."])
            # If a significant portion of history is error/no face, penalize heavily
            if len(facial_emotion_history) > 0 and error_face_count / len(facial_emotion_history) > 0.8:
                authenticity_score *= 0.1 # Very strong penalty
                # print("DEBUG (Auth): Face consistently showing error/no detection.")

        if speech_emotion == "Error" or speech_emotion == "No Speech Input":
            # Count errors/no speech in history
            error_speech_count = sum(1 for e in speech_emotion_history if e in ["Error", "No Speech Input", "Initializing..."])
            # If a significant portion of history is error/no speech, penalize heavily
            if len(speech_emotion_history) > 0 and error_speech_count / len(speech_emotion_history) > 0.8:
                authenticity_score *= 0.1 # Very strong penalty
                # print("DEBUG (Auth): Speech consistently showing error/no input.")

        # 2. Derive dominant emotions from history for more stable comparison
        # Filter out non-emotion states like 'No Face', 'Error', 'Initializing...', 'No Speech Input'
        valid_face_hist = [e for e in facial_emotion_history if e not in ["No Face", "Error", "Initializing..."]]
        valid_speech_hist = [e for e in speech_emotion_history if e not in ["No Speech Input", "Error", "Initializing..."]]

        # Get the most common emotion, or default to current if history is empty
        dominant_face_in_hist = Counter(valid_face_hist).most_common(1)[0][0] if valid_face_hist else facial_emotion
        dominant_speech_in_hist = Counter(valid_speech_hist).most_common(1)[0][0] if valid_speech_hist else speech_emotion

        print(f"Auth DEBUG: Dominant in History - Face:'{dominant_face_in_hist}', Speech:'{dominant_speech_in_hist}'")


        # If there's an actual emotion in current frame, use it if history is too short for smoothing
        # This balances responsiveness with stability by giving weight to recent events
        effective_facial_emotion = facial_emotion
        if facial_emotion not in ["No Face", "Error", "Initializing..."] and len(valid_face_hist) >= 3: # Use dominant if enough history
            effective_facial_emotion = dominant_face_in_hist

        effective_speech_emotion = speech_emotion
        if speech_emotion not in ["No Speech Input", "Error", "Initializing..."] and len(valid_speech_hist) >= 3:
            effective_speech_emotion = dominant_speech_in_hist

        print(f"Auth DEBUG: Effective Emotions - Face:'{effective_facial_emotion}', Speech:'{effective_speech_emotion}'")


        # 3. Apply Congruence Check using the *effective* (smoothed or current) emotions
        # Only check congruence if both effective emotions are valid and not errors
        if effective_facial_emotion not in ["No Face", "Error", "Initializing..."] and \
           effective_speech_emotion not in ["No Speech Input", "Error", "Initializing..."]:

            is_congruent = False
            if effective_facial_emotion == effective_speech_emotion:
                is_congruent = True
            elif effective_facial_emotion in self.emotion_congruence_map and \
                 effective_speech_emotion in self.emotion_congruence_map[effective_facial_emotion]:
                is_congruent = True

            print(f"Auth DEBUG: Congruence Check (Effective) - Is Congruent: {is_congruent}")

            if not is_congruent:
                # Soften penalty for incongruence of dominant/effective emotions
                authenticity_score *= 0.5
                print(f"Auth DEBUG: PENALTY - Effective emotions incongruent. Score after: {authenticity_score:.2f}")


        # 4. Re-evaluate Static Modality Checks with dominant emotions if applicable
        # Check if face is consistently neutral/no-face while dominant speech is expressive
        if dominant_face_in_hist in ["neutral", "No Face", "calm"] and \
           dominant_speech_in_hist not in ["neutral", "calm", "No Speech Input", "Error", "Initializing..."]:
            neutral_face_count = sum(1 for e in facial_emotion_history if e in ["neutral", "No Face", "Initializing..."])
            # If a significant portion of face history is neutral/no-face
            if len(facial_emotion_history) > 0 and neutral_face_count / len(facial_emotion_history) > 0.7:
                authenticity_score *= 0.6 # Penalty for static face with expressive voice
                print(f"Auth DEBUG: PENALTY - Static face, expressive voice. Score after: {authenticity_score:.2f}")


        # Check if voice is consistently neutral/no-speech while dominant face is expressive
        if dominant_speech_in_hist in ["neutral", "calm", "No Speech Input"] and \
           dominant_face_in_hist not in ["neutral", "No Face", "Initializing..."]:
            neutral_speech_count = sum(1 for e in speech_emotion_history if e in ["neutral", "calm", "No Speech Input", "Initializing..."])
            # If a significant portion of speech history is neutral/no-speech
            if len(speech_emotion_history) > 0 and neutral_speech_count / len(speech_emotion_history) > 0.7:
                authenticity_score *= 0.6 # Penalty for static voice with expressive face
                print(f"Auth DEBUG: PENALTY - Static voice, expressive face. Score after: {authenticity_score:.2f}")



        # Clamp score between 0.0 and 1.0
        authenticity_score = max(0.0, min(1.0, authenticity_score))
        authenticity_status = "AUTHENTIC" if authenticity_score >= self.authenticity_threshold else "POTENTIAL SPOOF"

        # --- DEBUG: Final Score for Frame ---
        print(f"Auth DEBUG: Final score for frame: {authenticity_score:.2f}, Status: {authenticity_status}")
        print(f"-----------------------------------\n")

        return authenticity_score, authenticity_status