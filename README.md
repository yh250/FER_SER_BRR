# Multimodal Emotional Authenticity System (MEAST)

## Overview

The Multimodal Emotional Authenticity System (MEAST) is a proof-of-concept project designed to enhance biometric security by assessing the "emotional authenticity" of a user's presentation. It analyzes inconsistencies between facial expressions and voice modulation that might indicate a spoofing attempt (e.g., deepfakes, replayed videos, static images with audio).

The system integrates:
1.  **Facial Emotion Recognition (FER):** Identifies emotions from visual cues.
2.  **Speech Emotion Recognition (SER):** Identifies emotions from vocal cues.
3.  **Cross-Modal Authenticity Analysis:** Compares and evaluates the coherence and consistency between detected facial and speech emotions over time.

This analysis yields an "Emotional Authenticity Score." A lower score suggests potential spoofing or an unnatural presentation, serving as an additional layer of security.

## How it Works

MEAST processes either live webcam/microphone input or a pre-recorded video file. It performs the following steps:

1.  **Data Capture:** Acquires video frames and audio segments.
2.  **Emotion Recognition:**
    * `DeepFace` performs Facial Emotion Recognition (FER) on video frames.
    * A fine-tuned `Hugging Face Wav2Vec2` model performs Speech Emotion Recognition (SER) on audio segments.
3.  **Emotional Authenticity Analysis:** A configurable logic-based module (`AuthenticityAnalyzer`) compares the detected facial and speech emotions for:
    * **Direct Congruence:** Do the emotions match or are they logically related (e.g., happy face and happy voice)?
    * **Temporal Consistency:** Are emotions stable over time, or are there unnatural rapid shifts?
    * **Modality Completeness:** Are both modalities consistently detectable?
    * This analysis generates an "Authenticity Score" (0.0 to 1.0, where 1.0 is fully authentic) and a "Risk Score" (1.0 - Authenticity Score).

## Modes of Operation

MEAST offers two primary modes of operation, each suited for different use cases:

### 1. Real-time / Live Demo Mode (`main.py`)

This mode provides a live, interactive demonstration, displaying results directly on a video feed. It's ideal for quick testing, visualization, and understanding the system's real-time behavior.

* **File to Use:** `main.py`
* **Input:**
    * **Live Webcam & Microphone:** Captures real-time video and audio from your default devices.
    * **Recorded Video File:** Plays a specified video file in a loop, processing its frames and extracted audio as if it were live.
* **Output:**
    * **GUI Window:** Displays the live video feed with overlaid text showing detected Facial Emotion, Speech Emotion, Authenticity Score, and Authenticity Status.
    * **Console Output:** Provides detailed debug information (if enabled) and system status messages.
* **Usage:** Run `python main.py` from your terminal.

#### Configuration for Real-time Mode (`config.py`):

To switch between live camera and video file input in `main.py`, modify `config.py`:

```python
# config.py

# ...
USE_LIVE_CAMERA = True      # Set to True for live webcam/mic input
                            # Set to False to use a video file

VIDEO_FILE_PATH = "path/to/your/recorded_video.mp4" # ONLY used if USE_LIVE_CAMERA is False
                                                    # Make sure this path is correct!

VIDEO_LOOPING_ENABLED = True # Set to True to loop the video file indefinitely
                             # Set to False to play the video once and exit
# ...
```
### 2. Batch Analysis Mode (`analyze_video.py`)

This mode is designed for automated processing of a single video file, providing a summarized result in a machine-readable JSON format. It's suitable for integration with other UIs or automated workflows.

* **File to Use:** `analyze_video.py`
* **Input:**
    * **Single Video File Path:** Provided as a command-line argument.
* **Output:**
    * **JSON File:** A single JSON file containing the overall authenticity result (Yes/No), average authenticity score, risk score, and other metadata for the entire video.
    * **Console Output:** Provides status messages and debug information (if enabled).
* **Usage:** from your terminal Run
   ```bash
   python analyze_video.py "path/to/your/video.mp4" --output_json "results.json"
    ```
   * Replace `"path/to/your/video.mp4"` with the actual path to your video file.
    * Replace `"path/to/your/output_json_folder"` with the directory where the analysis results (in JSON format) will be saved. Ensure this directory exists or the script has permission to create it.


#### Configuration for Batch Analysis Mode (`config.py`):

For `analyze_video.py`, the following `config.py` settings are implicitly used or overridden:

```python
# config.py

# ...
USE_LIVE_CAMERA = False      # analyze_video.py always processes a file, this is ignored
VIDEO_FILE_PATH = ""         # This is overridden by the command-line argument
VIDEO_LOOPING_ENABLED = False # analyze_video.py always processes once, this is ignored
# ...
```
---
## Project Structure
```
├── main.py                     # Real-time/Live Demo Mode execution script
├── analyze_video.py            # Batch Analysis Mode execution script
├── config.py                   # Global configuration parameters and thresholds
├── audio_processor.py          # Handles audio capture and Speech Emotion Recognition
├── video_processor.py          # Handles video capture and Facial Emotion Recognition
├── authenticity_analyzer.py    # Implements the core emotional authenticity logic
└── requirements.txt            # Lists all project dependencies
```
---
## Setup and Installation

Follow these steps to get the MEAST system up and running on your local machine:

1.  **Clone the Repository (or create files manually)**
    Create a project directory and place all the provided `.py` files and `requirements.txt` into it.

2.  **Create a Virtual Environment (Recommended)**
    It's highly recommended to use a virtual environment to manage dependencies and avoid conflicts with your system's Python packages.

    ```bash
    # Navigate into your project directory
    cd emotion_authenticity_system # Or whatever you named your folder

    # Create a virtual environment
    python -m venv venv

    # Activate the virtual environment
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Once your virtual environment is active, install the required libraries using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

    **Important Notes for Dependencies:**

    * **`sounddevice`**: Relies on the PortAudio library. On some Linux distributions, you might need to install it separately via your system's package manager before installing `sounddevice` through pip.
        * Debian/Ubuntu: `sudo apt-get install libportaudio2`
        * Fedora: `sudo dnf install portaudio-devel`
        * Arch Linux: `sudo pacman -S portaudio`
    * **`ffmpeg`**: Required by `librosa` to extract audio from video files. Ensure it's installed on your system and accessible in your `PATH`.
        * macOS: `brew install ffmpeg`
        * Ubuntu/Debian: `sudo apt install ffmpeg`
        * Windows: Download from official FFmpeg website and add to PATH, or use a tool like Scoop/Chocolatey.
    * **Initial Model Downloads**: The first time you run either `main.py` or `analyze_video.py`, DeepFace and the Hugging Face SER model will automatically download their pre-trained weights. This might take some time (several minutes) and requires an internet connection. Subsequent runs will use the cached models.

### Customization and Tuning (`config.py`)

The `config.py` file is central to customizing the system's behavior and tuning its authenticity detection.

* **`AUDIO_SAMPLERATE` / `SER_PROCESS_INTERVAL`**: Adjust audio processing parameters. A longer `SER_PROCESS_INTERVAL` (e.g., `4.0` or `5.0` seconds) can lead to more stable SER predictions by averaging over a larger audio chunk, potentially reducing "jumpiness."
* **`AUTHENTICITY_THRESHOLD`**: This is a crucial parameter (default `0.6`). Experiment with different values to define what score constitutes "AUTHENTIC" vs. "POTENTIAL SPOOF."
* **`RECENT_EMOTION_HISTORY_LENGTH`**: Controls the size of the historical window (number of recent emotion readings) used for temporal checks and smoothing. A larger value (e.g., `20` or `30`) can make the system more robust to transient fluctuations in emotion detection.
* **`EMOTION_CONGRUENCE`**: This dictionary is vital for defining what emotional pairs across modalities are considered congruent. This is where you should fine-tune the system based on how your specific models interpret emotions.

    **Example:** If your "sad" voice often gets recognized as "neutral" by the SER model, and you consider this authentic for a sad face, you would adjust:

    ```python
    EMOTION_CONGRUENCE = {
        # ...
        "sad": {"sad", "calm", "neutral"}, # If facial 'sad' can be paired with vocal 'neutral'
        "neutral": {"neutral", "calm", "sad"}, # If facial 'neutral' can be paired with vocal 'sad'
        # ...
    }
    ```

## Debugging

To understand why scores might be low or inconsistent, you can enable verbose debug prints:

* In `video_processor.py`: Uncomment `print(f"VideoProc: Facial Emotion - ...")` and other `print("VideoProc: ...")` lines inside `process_frame`.
* In `audio_processor.py`: Uncomment `print(f"AudioProc: Raw SER Results: {ser_results}")` inside `process_audio`.
* In `authenticity_analyzer.py`: Uncomment all `print(f"Auth DEBUG: ...")` lines inside `analyze`.

These prints will show frame-by-frame (or interval-by-interval) the detected emotions, the inputs to the authenticity analyzer, and how the score is being calculated and penalized. This is invaluable for tuning. Remember to comment them out for cleaner output once debugging is complete.

## Future Enhancements

This project is designed with modularity, making it easier to extend. Future enhancements could include:

* **Learned Authenticity Model**: Replace the rule-based `AuthenticityAnalyzer` with a deep learning model (e.g., LSTM, Transformer) trained on genuine and synthetic audio-visual samples.
* **Dimensional Emotion Models**: Use models that predict continuous Valence and Arousal instead of categorical emotions for finer-grained authenticity checks.
* **Integration with Traditional Biometrics**: Fully integrate this authenticity layer with an identity verification system.
* **Deployment Optimizations**: Optimize models for edge deployment or cloud scalability.

>**Disclaimer:**
>This project is a proof-of-concept and should not be used as a standalone security solution in production environments without extensive testing, validation, and further development. Human emotion is complex, and sophisticated spoofing techniques are constantly evolving.
