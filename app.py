import os
import gradio as gr
from faster_whisper import WhisperModel
import torch
import configparser
from typing import List, Tuple, Optional
import youtube_handler as yt
from ollama_handler import OllamaHandler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

logger.info(f"Initialized Whisper with device: {DEVICE}, compute type: {COMPUTE_TYPE}")
logger.info(
    f"Default model: {DEFAULT_MODEL}, beam size: {BEAM_SIZE}, VAD filter: {VAD_FILTER}"
)

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
DEFAULT_OLLAMA_MODEL = ollama.get_default_model() if OLLAMA_AVAILABLE else None


def load_model(model_name: str) -> WhisperModel:
    """Load the Whisper model with the specified configuration."""
    logger.info(f"Loading Whisper model: {model_name}")
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
        logger.info(f"Starting transcription of {audio_file}")
        logger.info(
            f"Model: {model_name}, Language: {language}, Summarize: {summarize}"
        )

        # Load the model
        model = load_model(model_name)

        # Transcribe the audio
        logger.info("Starting audio transcription...")
        segments, info = model.transcribe(
            audio_file,
            language=language if language != "Auto-detect" else None,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
        )

        # Combine all segments into one text
        full_text = " ".join([segment.text for segment in segments])
        logger.info(
            f"Transcription completed. Text length: {len(full_text)} characters"
        )
        logger.info(f"Detected language: {info.language}")

        # Generate summary if requested
        summary = None
        if summarize and OLLAMA_AVAILABLE:
            logger.info(f"Generating summary using Ollama model: {ollama_model}")
            summary = ollama.summarize(full_text, ollama_model)
            if summary:
                logger.info(f"Summary generated. Length: {len(summary)} characters")
            else:
                logger.warning("Failed to generate summary")

        return full_text, info.language, summary
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
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
        logger.info(f"Processing YouTube URL: {url}")
        logger.info(
            f"Model: {model_name}, Language: {language}, Summarize: {summarize}"
        )

        # First try to get available subtitles
        logger.info("Checking for available subtitles...")
        available_subs = yt.get_available_subtitles(url)

        if available_subs:
            logger.info(f"Found available subtitles: {', '.join(available_subs)}")
            # Try to download English subtitles first, then fall back to any available
            subtitle_path = yt.download_subtitles(url, "en")
            if not subtitle_path:
                logger.info(
                    "English subtitles not available, trying first available language"
                )
                subtitle_path = yt.download_subtitles(url, available_subs[0])

            if subtitle_path:
                logger.info(f"Successfully downloaded subtitles to: {subtitle_path}")
                with open(subtitle_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    summary = None
                    if summarize and OLLAMA_AVAILABLE:
                        logger.info(
                            f"Generating summary from subtitles using Ollama model: {ollama_model}"
                        )
                        summary = ollama.summarize(text, ollama_model)
                        if summary:
                            logger.info(
                                f"Summary generated. Length: {len(summary)} characters"
                            )
                        else:
                            logger.warning("Failed to generate summary")
                    return text, "en", "Subtitles", summary

        # If no subtitles available, download and transcribe
        logger.info("No subtitles available, downloading video for transcription...")
        audio_path, video_title = yt.download_video(url)
        logger.info(f"Video downloaded: {video_title}")

        transcription, detected_lang, summary = transcribe_audio(
            audio_path, model_name, language, summarize, ollama_model
        )

        # Clean up the temporary audio file
        try:
            os.remove(audio_path)
            logger.info("Cleaned up temporary audio file")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file: {str(e)}")

        return transcription, detected_lang, "Transcription", summary

    except Exception as e:
        logger.error(f"Error processing YouTube video: {str(e)}")
        return f"Error processing YouTube video: {str(e)}", None, "Error", None


def create_interface():
    """Create and return the Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎙️ Audio/Video Transcription with Whisper")
        gr.Markdown(
            "### A powerful tool for transcribing and summarizing audio/video content"
        )

        with gr.Tabs() as tabs:
            with gr.TabItem("Local File"):
                gr.Markdown(
                    """
                ### Local File Transcription
                Upload an audio or video file to transcribe it using Whisper AI.
                - Supports various audio and video formats
                - Automatic language detection
                - Optional summarization with Ollama
                """
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
                                    value=DEFAULT_OLLAMA_MODEL,
                                    label="Ollama Model",
                                    visible=False,
                                )

                                def toggle_summary(checked):
                                    if checked and not ollama.is_available():
                                        gr.Warning(
                                            "Ollama is not available. Please check your Ollama server."
                                        )
                                        return False
                                    return checked

                                def update_model_visibility(visible):
                                    if visible:
                                        return gr.Dropdown.update(
                                            visible=True, value=DEFAULT_OLLAMA_MODEL
                                        )
                                    return gr.Dropdown.update(visible=False)

                                summarize_checkbox.change(
                                    fn=toggle_summary,
                                    inputs=[summarize_checkbox],
                                    outputs=[summarize_checkbox],
                                ).then(
                                    fn=update_model_visibility,
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
                    """
                ### YouTube Video Processing
                Enter a YouTube URL to transcribe the video or extract available subtitles.
                - Supports youtube.com, youtu.be, and invidious URLs
                - Automatically extracts subtitles if available
                - Falls back to transcription if no subtitles found
                - Optional summarization with Ollama
                """
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
                                    value=DEFAULT_OLLAMA_MODEL,
                                    label="Ollama Model",
                                    visible=False,
                                )

                                def toggle_yt_summary(checked):
                                    if checked and not ollama.is_available():
                                        gr.Warning(
                                            "Ollama is not available. Please check your Ollama server."
                                        )
                                        return False
                                    return checked

                                def update_yt_model_visibility(visible):
                                    if visible:
                                        return gr.Dropdown.update(
                                            visible=True, value=DEFAULT_OLLAMA_MODEL
                                        )
                                    return gr.Dropdown.update(visible=False)

                                yt_summarize_checkbox.change(
                                    fn=toggle_yt_summary,
                                    inputs=[yt_summarize_checkbox],
                                    outputs=[yt_summarize_checkbox],
                                ).then(
                                    fn=update_yt_model_visibility,
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
        
        ### Status:
        - Device: {DEVICE}
        - Compute Type: {COMPUTE_TYPE}
        - Ollama Status: {"Available" if OLLAMA_AVAILABLE else "Not Available"}
        {"- Available Ollama Models: " + ", ".join(OLLAMA_MODELS) if OLLAMA_AVAILABLE else ""}
        """
        )

    return app


if __name__ == "__main__":
    logger.info("Starting Whisper Transcription Web App")
    logger.info(f"Server will be available at http://{SERVER_NAME}:{SERVER_PORT}")
    app = create_interface()
    app.launch(share=SHARE, server_name=SERVER_NAME, server_port=SERVER_PORT)
