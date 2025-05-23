import os
import gradio as gr
from faster_whisper import WhisperModel
import torch
import configparser
from typing import List


def load_config() -> configparser.ConfigParser:
    """Load configuration from config.ini file."""
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), "config.ini")
    config.read(config_path)
    return config


# Load configuration
config = load_config()

# Whisper configuration
DEFAULT_MODEL = config["whisper"]["default_model"]
DEVICE = config["whisper"]["device"] if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = config["whisper"]["compute_type"] if DEVICE == "cuda" else "float32"
BEAM_SIZE = config["whisper"].getint("beam_size")
VAD_FILTER = config["whisper"].getboolean("vad_filter")

# App configuration
MAX_DURATION = config["app"].getint("max_duration")
SERVER_NAME = config["app"]["server_name"]
SERVER_PORT = config["app"].getint("server_port")
SHARE = config["app"].getboolean("share")

# Available models and languages
WHISPER_MODELS = config["models"]["available_models"].split(",")
AVAILABLE_LANGUAGES = config["languages"]["available_languages"].split(",")


def load_model(model_name: str) -> WhisperModel:
    """Load the Whisper model with the specified configuration."""
    return WhisperModel(model_name, device=DEVICE, compute_type=COMPUTE_TYPE)


def transcribe_audio(
    audio_file: str, model_name: str, language: str = None
) -> tuple[str, str]:
    """Transcribe audio using the selected Whisper model."""
    try:
        # Load the model
        model = load_model(model_name)

        # Transcribe the audio
        segments, info = model.transcribe(
            audio_file,
            language=language if language != "Auto-detect" else None,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
        )

        # Combine all segments into one text
        full_text = " ".join([segment.text for segment in segments])

        return full_text, info.language
    except Exception as e:
        return f"Error during transcription: {str(e)}", None


def create_interface():
    """Create and return the Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎙️ Audio/Video Transcription with Whisper")
        gr.Markdown("Upload an audio or video file to transcribe it using Whisper AI.")

        with gr.Row():
            with gr.Column():
                # Input components
                audio_input = gr.Audio(
                    label="Upload Audio/Video", type="filepath", format="mp3"
                )
                model_dropdown = gr.Dropdown(
                    choices=WHISPER_MODELS,
                    value=DEFAULT_MODEL,
                    label="Select Whisper Model",
                )
                language_dropdown = gr.Dropdown(
                    choices=["Auto-detect"] + AVAILABLE_LANGUAGES,
                    value="Auto-detect",
                    label="Language (optional)",
                )
                transcribe_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column():
                # Output components
                output_text = gr.Textbox(label="Transcription", lines=10, max_lines=20)
                detected_language = gr.Textbox(
                    label="Detected Language", interactive=False
                )

        # Set up the event handler
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, model_dropdown, language_dropdown],
            outputs=[output_text, detected_language],
        )

        # Add some helpful information
        gr.Markdown(
            f"""
        ### Tips:
        - For better accuracy, use larger models (medium, large)
        - Processing time increases with model size
        - GPU is recommended for faster processing
        - Maximum audio duration is {MAX_DURATION // 60} minutes
        """
        )

    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(share=SHARE, server_name=SERVER_NAME, server_port=SERVER_PORT)
