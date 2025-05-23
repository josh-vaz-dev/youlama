import os
import gradio as gr
from faster_whisper import WhisperModel
import torch
import configparser
from typing import List, Tuple, Optional
import youtube_handler as yt
from ollama_handler import OllamaHandler


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

# Initialize Ollama handler
ollama = OllamaHandler()
OLLAMA_AVAILABLE = ollama.is_available()
OLLAMA_MODELS = ollama.get_available_models() if OLLAMA_AVAILABLE else []


def load_model(model_name: str) -> WhisperModel:
    """Load the Whisper model with the specified configuration."""
    return WhisperModel(model_name, device=DEVICE, compute_type=COMPUTE_TYPE)


def transcribe_audio(
    audio_file: str,
    model_name: str,
    language: str = None,
    summarize: bool = False,
    ollama_model: str = None,
) -> tuple[str, str, Optional[str]]:
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

        # Generate summary if requested
        summary = None
        if summarize and OLLAMA_AVAILABLE:
            summary = ollama.summarize(full_text, ollama_model)

        return full_text, info.language, summary
    except Exception as e:
        return f"Error during transcription: {str(e)}", None, None


def process_youtube_url(
    url: str,
    model_name: str,
    language: str = None,
    summarize: bool = False,
    ollama_model: str = None,
) -> Tuple[str, str, str, Optional[str]]:
    """Process a YouTube URL and return transcription or subtitles."""
    try:
        # First try to get available subtitles
        available_subs = yt.get_available_subtitles(url)

        if available_subs:
            # Try to download English subtitles first, then fall back to any available
            subtitle_path = yt.download_subtitles(url, "en")
            if not subtitle_path:
                subtitle_path = yt.download_subtitles(url, available_subs[0])

            if subtitle_path:
                with open(subtitle_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    summary = None
                    if summarize and OLLAMA_AVAILABLE:
                        summary = ollama.summarize(text, ollama_model)
                    return text, "en", "Subtitles", summary

        # If no subtitles available, download and transcribe
        audio_path, video_title = yt.download_video(url)
        transcription, detected_lang, summary = transcribe_audio(
            audio_path, model_name, language, summarize, ollama_model
        )

        # Clean up the temporary audio file
        try:
            os.remove(audio_path)
        except:
            pass

        return transcription, detected_lang, "Transcription", summary

    except Exception as e:
        return f"Error processing YouTube video: {str(e)}", None, "Error", None


def create_interface():
    """Create and return the Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎙️ Audio/Video Transcription with Whisper")

        with gr.Tabs() as tabs:
            with gr.TabItem("Local File"):
                gr.Markdown(
                    "Upload an audio or video file to transcribe it using Whisper AI."
                )

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
                        if OLLAMA_AVAILABLE:
                            with gr.Group():
                                summarize_checkbox = gr.Checkbox(
                                    label="Generate Summary", value=False
                                )
                                ollama_model_dropdown = gr.Dropdown(
                                    choices=OLLAMA_MODELS,
                                    value=OLLAMA_MODELS[0] if OLLAMA_MODELS else None,
                                    label="Ollama Model",
                                    visible=False,
                                )
                                summarize_checkbox.change(
                                    fn=lambda x: {"visible": x},
                                    inputs=[summarize_checkbox],
                                    outputs=[ollama_model_dropdown],
                                )
                        transcribe_btn = gr.Button("Transcribe", variant="primary")

                    with gr.Column():
                        # Output components
                        output_text = gr.Textbox(
                            label="Transcription", lines=10, max_lines=20
                        )
                        detected_language = gr.Textbox(
                            label="Detected Language", interactive=False
                        )
                        if OLLAMA_AVAILABLE:
                            summary_text = gr.Textbox(
                                label="Summary", lines=5, max_lines=10, visible=False
                            )

                # Set up the event handler
                def transcribe_with_summary(
                    audio, model, lang, summarize, ollama_model
                ):
                    result = transcribe_audio(
                        audio, model, lang, summarize, ollama_model
                    )
                    if len(result) == 3:
                        text, lang, summary = result
                        return text, lang, summary if summary else ""
                    return result[0], result[1], ""

                transcribe_btn.click(
                    fn=transcribe_with_summary,
                    inputs=[
                        audio_input,
                        model_dropdown,
                        language_dropdown,
                        (
                            summarize_checkbox
                            if OLLAMA_AVAILABLE
                            else gr.Checkbox(value=False)
                        ),
                        (
                            ollama_model_dropdown
                            if OLLAMA_AVAILABLE
                            else gr.Dropdown(value=None)
                        ),
                    ],
                    outputs=[
                        output_text,
                        detected_language,
                        summary_text if OLLAMA_AVAILABLE else gr.Textbox(),
                    ],
                )

            with gr.TabItem("YouTube"):
                gr.Markdown(
                    "Enter a YouTube URL to transcribe the video or extract available subtitles."
                )

                with gr.Row():
                    with gr.Column():
                        # YouTube input components
                        youtube_url = gr.Textbox(
                            label="YouTube URL",
                            placeholder="Enter YouTube URL (youtube.com, youtu.be, or invidious)",
                        )
                        yt_model_dropdown = gr.Dropdown(
                            choices=WHISPER_MODELS,
                            value=DEFAULT_MODEL,
                            label="Select Whisper Model",
                        )
                        yt_language_dropdown = gr.Dropdown(
                            choices=["Auto-detect"] + AVAILABLE_LANGUAGES,
                            value="Auto-detect",
                            label="Language (optional)",
                        )
                        if OLLAMA_AVAILABLE:
                            with gr.Group():
                                yt_summarize_checkbox = gr.Checkbox(
                                    label="Generate Summary", value=False
                                )
                                yt_ollama_model_dropdown = gr.Dropdown(
                                    choices=OLLAMA_MODELS,
                                    value=OLLAMA_MODELS[0] if OLLAMA_MODELS else None,
                                    label="Ollama Model",
                                    visible=False,
                                )
                                yt_summarize_checkbox.change(
                                    fn=lambda x: {"visible": x},
                                    inputs=[yt_summarize_checkbox],
                                    outputs=[yt_ollama_model_dropdown],
                                )
                        yt_process_btn = gr.Button("Process Video", variant="primary")

                    with gr.Column():
                        # YouTube output components
                        yt_output_text = gr.Textbox(
                            label="Result", lines=10, max_lines=20
                        )
                        yt_detected_language = gr.Textbox(
                            label="Detected Language", interactive=False
                        )
                        yt_source = gr.Textbox(label="Source", interactive=False)
                        if OLLAMA_AVAILABLE:
                            yt_summary_text = gr.Textbox(
                                label="Summary", lines=5, max_lines=10, visible=False
                            )

                # Set up the event handler
                def process_yt_with_summary(url, model, lang, summarize, ollama_model):
                    result = process_youtube_url(
                        url, model, lang, summarize, ollama_model
                    )
                    if len(result) == 4:
                        text, lang, source, summary = result
                        return text, lang, source, summary if summary else ""
                    return result[0], result[1], result[2], ""

                yt_process_btn.click(
                    fn=process_yt_with_summary,
                    inputs=[
                        youtube_url,
                        yt_model_dropdown,
                        yt_language_dropdown,
                        (
                            yt_summarize_checkbox
                            if OLLAMA_AVAILABLE
                            else gr.Checkbox(value=False)
                        ),
                        (
                            yt_ollama_model_dropdown
                            if OLLAMA_AVAILABLE
                            else gr.Dropdown(value=None)
                        ),
                    ],
                    outputs=[
                        yt_output_text,
                        yt_detected_language,
                        yt_source,
                        yt_summary_text if OLLAMA_AVAILABLE else gr.Textbox(),
                    ],
                )

        # Add some helpful information
        gr.Markdown(
            f"""
        ### Tips:
        - For better accuracy, use larger models (medium, large)
        - Processing time increases with model size
        - GPU is recommended for faster processing
        - Maximum audio duration is {MAX_DURATION // 60} minutes
        - YouTube videos will first try to use available subtitles
        - If no subtitles are available, the video will be transcribed
        {"- Ollama summarization is available for both local files and YouTube videos" if OLLAMA_AVAILABLE else ""}
        """
        )

    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(share=SHARE, server_name=SERVER_NAME, server_port=SERVER_PORT)
