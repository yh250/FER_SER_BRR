# analyze_video.py

import argparse
import os
import cv2
import time
import numpy as np
from collections import deque
import threading
import json
import sys  # For sys.stderr and sys.exit

# Import your modular components
from config import (
    AUDIO_SAMPLERATE, SER_PROCESS_INTERVAL,
    AUTHENTICITY_THRESHOLD, RECENT_EMOTION_HISTORY_LENGTH, EMOTION_CONGRUENCE
)
from audio_processor import AudioProcessor
from video_processor import VideoProcessor  # Make sure this is updated to use config.VIDEO_LOOPING_ENABLED
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
    time.sleep_interval = 0.01  # Small sleep for thread management

    # Calculate how many audio samples correspond to one video frame duration
    audio_frame_duration = 1 / video_fps if video_fps > 0 else 0.033  # Default 30 FPS if not available
    audio_samples_per_frame = int(audio_frame_duration * audio_processor_instance.samplerate)

    current_audio_index = 0
    while not shared_dict["stop_audio_thread"]:
        if current_audio_index < len(total_audio_data):
            # Feed audio chunks equivalent to video frames
            end_index = min(current_audio_index + audio_samples_per_frame, len(total_audio_data))
            audio_chunk = total_audio_data[current_audio_index:end_index]
            audio_processor_instance.add_audio_data(audio_chunk)
            current_audio_index = end_index
        else:
            # If all audio data has been fed from the video,
            # ensure any remaining buffered audio is processed.
            if audio_processor_instance.current_audio_chunk:
                speech_emo, speech_hist = audio_processor_instance.process_audio()
                shared_dict["current_speech_emotion"] = speech_emo
                shared_dict["speech_emotion_history"] = speech_hist
                time.sleep(time.sleep_interval)
            else:
                # All audio has been processed, signal thread to stop
                shared_dict["stop_audio_thread"] = True
                break  # Exit the audio thread loop

        # Process audio and update shared data
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
        print(f"Error: Video file not found at '{video_path}'", file=sys.stderr)
        sys.exit(1)  # Exit with an error code

    print(f"Starting analysis for video: '{video_path}'")

    video_processor = None
    audio_processor = None
    audio_processing_thread = None

    all_authenticity_scores = []
    # Initialize final_results with default/error values
    final_results = {
        "video_path": video_path,
        "is_authentic": "Error",
        "risk_score": 1.0,  # Default to highest risk in case of error or no scores
        "average_authenticity_score": 0.0,
        "total_frames_processed": 0,
        "error_message": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST")
    }

    try:
        # VideoProcessor will now use VIDEO_LOOPING_ENABLED from config.py
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
        audio_processing_thread.daemon = True  # Daemon thread exits when main program exits
        audio_processing_thread.start()
        print("Audio processing thread started.")

        frame_count = 0
        while True:
            # Process one video frame
            frame, current_facial_emotion, facial_emotion_history = video_processor.process_frame()
            if frame is None:
                # Video has ended (VideoProcessor returns None when not looping and video ends)
                break

            frame_count += 1

            # Get latest speech emotion data from the audio processing thread
            current_speech_emotion = shared_data["current_speech_emotion"]
            speech_emotion_history = shared_data["speech_emotion_history"]

            # Calculate Emotional Authenticity Measure for this specific frame/interval
            authenticity_score, _ = authenticity_analyzer.analyze(  # We only need the score for final aggregation
                current_facial_emotion, current_speech_emotion,
                facial_emotion_history, speech_emotion_history
            )
            all_authenticity_scores.append(authenticity_score)

            # No cv2.imshow or time.sleep for real-time playback in this script,
            # it processes the video as fast as possible.

    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}", file=sys.stderr)
        final_results["error_message"] = str(e)
        final_results["is_authentic"] = "Error"  # Mark as error in results
        final_results["risk_score"] = 1.0  # Highest risk on error
    finally:
        # --- Cleanup ---
        # Signal the audio thread to stop and wait for it to finish gracefully
        shared_data["stop_audio_thread"] = True
        if audio_processing_thread and audio_processing_thread.is_alive():
            audio_processing_thread.join(timeout=5)  # Give it up to 5 seconds to finish
            if audio_processing_thread.is_alive():
                print("Warning: Audio thread did not terminate gracefully.", file=sys.stderr)

        # Stop and release resources
        if audio_processor:
            audio_processor.stop()
        if video_processor:
            video_processor.stop()

        print("Analysis process finished.")

    # --- Final Output Calculation and JSON Writing ---
    if all_authenticity_scores:
        final_authenticity_score = np.mean(all_authenticity_scores)

        # Determine Yes/No authenticity based on the authenticity threshold
        is_authentic = "Yes" if final_authenticity_score >= AUTHENTICITY_THRESHOLD else "No"

        # Calculate Risk Score (opposite of authenticity score)
        risk_score = 1.0 - final_authenticity_score

        # Update final_results dictionary with computed values
        final_results.update({
            "is_authentic": is_authentic,
            "risk_score": round(risk_score, 2),  # Round for cleaner JSON output
            "average_authenticity_score": round(final_authenticity_score, 2),
            "total_frames_processed": frame_count
        })
    else:
        # If no scores were collected (e.g., very short video, all errors)
        final_results["is_authentic"] = "No"  # Default to not authentic or problematic
        final_results["risk_score"] = 1.0  # Max risk
        final_results["average_authenticity_score"] = 0.0  # Min score
        if not final_results["error_message"]:  # If no specific error, state reason for no scores
            final_results[
                "error_message"] = "No authenticity scores could be generated from video (e.g., no frames processed)."

    # Write the final_results dictionary to the specified JSON file
    try:
        with open(output_json_path, 'w') as f:
            json.dump(final_results, f, indent=4)  # indent=4 for pretty-printing JSON
        print(f"Analysis results saved to '{output_json_path}'")
    except IOError as e:
        print(f"Error writing results to JSON file '{output_json_path}': {e}", file=sys.stderr)
        sys.exit(1)  # Exit with an error code if file writing fails

    # Exit with an error code if there was any significant error during processing
    if final_results["error_message"]:
        sys.exit(1)


if __name__ == '__main__':
    main()