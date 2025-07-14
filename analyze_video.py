# analyze_video.py

import argparse
import os
import cv2
import time
import numpy as np
from collections import deque
import threading
import json  # New import for JSON operations

# Import your modular components
from config import (
    AUDIO_SAMPLERATE, SER_PROCESS_INTERVAL,
    AUTHENTICITY_THRESHOLD, RECENT_EMOTION_HISTORY_LENGTH, EMOTION_CONGRUENCE
)
from audio_processor import AudioProcessor
from video_processor import VideoProcessor
from authenticity_analyzer import AuthenticityAnalyzer

# --- Global state for communication between threads ---
shared_data = {
    "current_speech_emotion": "Initializing...",
    "speech_emotion_history": [],
    "stop_audio_thread": False
}


# Function for processing audio in a separate thread (adapted for single run)
def audio_thread_function_for_file(audio_processor_instance, shared_dict, video_processor_instance, total_audio_data,
                                   video_fps):
    time.sleep_interval = 0.01  # Small sleep

    audio_frame_duration = 1 / video_fps if video_fps > 0 else 0.033
    audio_samples_per_frame = int(audio_frame_duration * audio_processor_instance.samplerate)

    current_audio_index = 0
    while not shared_dict["stop_audio_thread"]:
        if current_audio_index < len(total_audio_data):
            end_index = min(current_audio_index + audio_samples_per_frame, len(total_audio_data))
            audio_chunk = total_audio_data[current_audio_index:end_index]
            audio_processor_instance.add_audio_data(audio_chunk)
            current_audio_index = end_index
        else:
            if audio_processor_instance.current_audio_chunk:
                speech_emo, speech_hist = audio_processor_instance.process_audio()
                shared_dict["current_speech_emotion"] = speech_emo
                shared_dict["speech_emotion_history"] = speech_hist
                time.sleep(time.sleep_interval)
            else:
                shared_dict["stop_audio_thread"] = True
                break

        speech_emo, speech_hist = audio_processor_instance.process_audio()
        shared_dict["current_speech_emotion"] = speech_emo
        shared_dict["speech_emotion_history"] = speech_hist
        time.sleep(time.sleep_interval)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a video file for emotional authenticity and save results to JSON.")
    parser.add_argument("video_path", type=str, help="Path to the video file to analyze.")
    parser.add_argument("--output_json", type=str, default="authenticity_results.json",
                        help="Path to the JSON file where results will be saved.")
    args = parser.parse_args()

    video_path = args.video_path
    output_json_path = args.output_json

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'", file=sys.stderr)  # Print errors to stderr
        sys.exit(1)  # Exit with an error code

    print(f"Starting analysis for video: '{video_path}'")

    video_processor = None
    audio_processor = None
    audio_processing_thread = None

    all_authenticity_scores = []
    final_results = {
        "video_path": video_path,
        "is_authentic": "Error",
        "risk_score": 1.0,  # Default to highest risk in case of error
        "average_authenticity_score": 0.0,
        "total_frames_processed": 0,
        "error_message": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST")
    }

    try:
        video_processor = VideoProcessor(video_path, RECENT_EMOTION_HISTORY_LENGTH)
        audio_processor = AudioProcessor(AUDIO_SAMPLERATE, SER_PROCESS_INTERVAL, RECENT_EMOTION_HISTORY_LENGTH,
                                         use_live_mic=False)
        authenticity_analyzer = AuthenticityAnalyzer(EMOTION_CONGRUENCE, AUTHENTICITY_THRESHOLD)

        total_audio_data = video_processor.get_audio_from_video(AUDIO_SAMPLERATE)
        if total_audio_data is None or len(total_audio_data) == 0:
            print(
                "Warning: No audio data extracted from video or audio is empty. Analysis will rely only on facial emotion.")

        # Start audio processing in a separate thread
        audio_processing_thread = threading.Thread(
            target=audio_thread_function_for_file,
            args=(audio_processor, shared_data, video_processor, total_audio_data, video_processor.fps)
        )
        audio_processing_thread.daemon = True
        audio_processing_thread.start()
        print("Audio processing thread started.")

        frame_count = 0
        while True:
            frame, current_facial_emotion, facial_emotion_history = video_processor.process_frame()
            if frame is None:
                break  # End of video file

            frame_count += 1

            current_speech_emotion = shared_data["current_speech_emotion"]
            speech_emotion_history = shared_data["speech_emotion_history"]

            authenticity_score, _ = authenticity_analyzer.analyze(
                current_facial_emotion, current_speech_emotion,
                facial_emotion_history, speech_emotion_history
            )
            all_authenticity_scores.append(authenticity_score)

    except Exception as e:
        print(f"An error occurred during analysis: {e}", file=sys.stderr)
        final_results["error_message"] = str(e)
        final_results["is_authentic"] = "Error"
        final_results["risk_score"] = 1.0  # High risk on error
    finally:
        # --- Cleanup ---
        shared_data["stop_audio_thread"] = True
        if audio_processing_thread and audio_processing_thread.is_alive():
            audio_processing_thread.join(timeout=5)
        if audio_processor:
            audio_processor.stop()
        if video_processor:
            video_processor.stop()

        print("Analysis complete.")

    # --- Final Output Calculation ---
    if all_authenticity_scores:
        final_authenticity_score = np.mean(all_authenticity_scores)
        is_authentic = "Yes" if final_authenticity_score >= AUTHENTICITY_THRESHOLD else "No"
        risk_score = 1.0 - final_authenticity_score

        final_results.update({
            "is_authentic": is_authentic,
            "risk_score": round(risk_score, 2),  # Round for cleaner JSON output
            "average_authenticity_score": round(final_authenticity_score, 2),
            "total_frames_processed": frame_count
        })
    else:
        final_results["is_authentic"] = "No"  # If no scores, imply not authentic or problematic
        final_results["risk_score"] = 1.0
        final_results["average_authenticity_score"] = 0.0
        if not final_results["error_message"]:  # If no specific error, indicate no scores
            final_results["error_message"] = "No authenticity scores could be generated from video."

    # --- Write results to JSON file ---
    try:
        with open(output_json_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"Analysis results saved to '{output_json_path}'")
    except IOError as e:
        print(f"Error writing results to JSON file: {e}", file=sys.stderr)
        sys.exit(1)  # Exit with an error code

    if final_results["error_message"]:
        sys.exit(1)  # Exit with an error if there was an internal processing error


if __name__ == '__main__':
    import sys  # Import sys for sys.stderr and sys.exit

    main()