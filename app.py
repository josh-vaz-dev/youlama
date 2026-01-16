import os
import gradio as gr
import torch
import configparser
from typing import List, Tuple, Optional
import youtube_handler as yt
from llm_handler import UnifiedLLMHandler
import logging
from faster_whisper import WhisperModel
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_cuda_compatibility():
    """Check if the current CUDA setup is compatible with faster-whisper."""
    logger.info("Checking CUDA compatibility...")

    # Check PyTorch CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available in PyTorch")
        return False

    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()
    device_name = torch.cuda.get_device_name(0)

    logger.info(f"CUDA Version: {cuda_version}")
    logger.info(f"cuDNN Version: {cudnn_version}")
    logger.info(f"GPU Device: {device_name}")

    # Check CUDA version
    try:
        cuda_major = int(cuda_version.split(".")[0])
        if cuda_major > 11:
            logger.warning(
                f"CUDA {cuda_version} might not be fully compatible with faster-whisper. Recommended: CUDA 11.x"
            )
            logger.info(
                "Consider creating a new environment with CUDA 11.x if you encounter issues"
            )
    except Exception as e:
        logger.error(f"Error parsing CUDA version: {str(e)}")

    return True


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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"
BEAM_SIZE = config["whisper"].getint("beam_size")
VAD_FILTER = config["whisper"].getboolean("vad_filter")
# Quality enhancement parameters
BEST_OF = config["whisper"].getint("best_of", fallback=5)
PATIENCE = config["whisper"].getfloat("patience", fallback=1.0)
TEMPERATURE = config["whisper"].getfloat("temperature", fallback=0)
COMPRESSION_RATIO_THRESHOLD = config["whisper"].getfloat("compression_ratio_threshold", fallback=2.4)
LOG_PROB_THRESHOLD = config["whisper"].getfloat("log_prob_threshold", fallback=-1.0)
NO_SPEECH_THRESHOLD = config["whisper"].getfloat("no_speech_threshold", fallback=0.6)
CONDITION_ON_PREVIOUS = config["whisper"].getboolean("condition_on_previous_text", fallback=True)
WORD_TIMESTAMPS = config["whisper"].getboolean("word_timestamps", fallback=True)

# Log device and compute type
logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
logger.info(f"Using device: {DEVICE}, compute type: {COMPUTE_TYPE}")
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

# Initialize Unified LLM handler with multi-backend support
llm = UnifiedLLMHandler()
LLM_AVAILABLE = llm.is_available()
LLM_MODELS = llm.get_available_models() if LLM_AVAILABLE else []
DEFAULT_LLM_MODEL = llm.get_default_model() if LLM_AVAILABLE else None
ACTIVE_BACKEND = llm.get_active_backend() if LLM_AVAILABLE else "None"

logger.info(f"LLM Backend: {ACTIVE_BACKEND}, Available: {LLM_AVAILABLE}")


def load_model(model_name: str):
    """Load the Whisper model with the specified configuration."""
    try:
        logger.info(f"Loading Whisper model: {model_name}")
        return WhisperModel(
            model_name,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=os.path.join(os.path.dirname(__file__), "models"),
        )
    except Exception as e:
        logger.error(f"Error loading model with CUDA: {str(e)}")
        logger.info("Falling back to CPU")
        return WhisperModel(
            model_name,
            device="cpu",
            compute_type="float32",
            download_root=os.path.join(os.path.dirname(__file__), "models"),
        )


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

        # Transcribe the audio with quality-optimized settings
        logger.info("Starting audio transcription with enhanced quality settings...")
        segments, info = model.transcribe(
            audio_file,
            language=language if language != "Auto-detect" else None,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            best_of=BEST_OF,
            patience=PATIENCE,
            temperature=TEMPERATURE,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            log_prob_threshold=LOG_PROB_THRESHOLD,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            condition_on_previous_text=CONDITION_ON_PREVIOUS,
            word_timestamps=WORD_TIMESTAMPS,
        )

        # Get the full text with timestamps
        full_text = " ".join([segment.text for segment in segments])
        logger.info(
            f"Transcription completed. Text length: {len(full_text)} characters"
        )
        logger.info(f"Detected language: {info.language}")

        # Generate summary if requested
        summary = None
        if summarize and LLM_AVAILABLE:
            logger.info(f"Generating summary using LLM model: {ollama_model}")
            summary = llm.summarize(full_text, ollama_model)
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
                    if summarize and LLM_AVAILABLE:
                        logger.info(
                            f"Generating summary from subtitles using LLM model: {ollama_model}"
                        )
                        summary = llm.summarize(text, ollama_model)
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
        gr.Markdown("# 🎥 YouLama")
        gr.Markdown("### AI-powered YouTube video transcription and summarization")

        with gr.Tabs() as tabs:
            with gr.TabItem("YouTube"):
                gr.Markdown(
                    """
                ### YouTube Video Processing
                Enter a YouTube URL to transcribe the video or extract available subtitles.
                - Supports youtube.com, youtu.be, and invidious URLs
                - Automatically extracts subtitles if available
                - Falls back to transcription if no subtitles found
                - Optional AI-powered summarization with Ollama
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
                        with gr.Group():
                            yt_summarize_checkbox = gr.Checkbox(
                                label="Generate AI Summary",
                                value=False,
                                interactive=LLM_AVAILABLE,
                            )
                            yt_ollama_model_dropdown = gr.Dropdown(
                                choices=(
                                    LLM_MODELS
                                    if LLM_AVAILABLE
                                    else ["No models available"]
                                ),
                                value=(
                                    DEFAULT_LLM_MODEL if LLM_AVAILABLE else None
                                ),
                                label="LLM Model",
                                interactive=LLM_AVAILABLE,
                            )

                        # Add status bar
                        yt_status = gr.Textbox(
                            label="Status",
                            value="Waiting for input...",
                            interactive=False,
                            elem_classes=["status-bar"],
                        )

                        yt_process_btn = gr.Button("Process Video", variant="primary")

                    with gr.Column():
                        # YouTube output components
                        yt_output_text = gr.Textbox(
                            label="Transcription", lines=10, max_lines=20
                        )
                        yt_detected_language = gr.Textbox(
                            label="Detected Language", interactive=False
                        )
                        yt_source = gr.Textbox(label="Source", interactive=False)

                # Add summary text box below the main output
                if LLM_AVAILABLE:
                    yt_summary_text = gr.Textbox(
                        label="AI Summary", lines=5, max_lines=10, value=""
                    )

                # Set up the event handler
                def process_yt_with_summary(url, model, lang, summarize, ollama_model):
                    try:
                        # Update status for each step
                        status = "Checking URL and fetching video information..."
                        result = process_youtube_url(
                            url, model, lang, summarize, ollama_model
                        )

                        if len(result) == 4:
                            text, lang, source, summary = result
                            if source == "Subtitles":
                                status = "Processing subtitles..."
                            else:
                                status = "Transcribing video..."

                            if summarize and summary:
                                status = "Generating AI summary..."

                            return (
                                text,
                                lang,
                                source,
                                summary if summary else "",
                                "Processing complete!",
                            )
                        else:
                            return (
                                result[0],
                                result[1],
                                result[2],
                                "",
                                f"Error: {result[0]}",
                            )
                    except Exception as e:
                        logger.error(f"Error in process_yt_with_summary: {str(e)}")
                        return f"Error: {str(e)}", None, None, "", "Processing failed!"

                yt_process_btn.click(
                    fn=process_yt_with_summary,
                    inputs=[
                        youtube_url,
                        yt_model_dropdown,
                        yt_language_dropdown,
                        yt_summarize_checkbox,
                        yt_ollama_model_dropdown,
                    ],
                    outputs=[
                        yt_output_text,
                        yt_detected_language,
                        yt_source,
                        yt_summary_text if LLM_AVAILABLE else gr.Textbox(),
                        yt_status,
                    ],
                )

            with gr.TabItem("Local File"):
                gr.Markdown(
                    """
                ### Local File Transcription
                Upload an audio or video file to transcribe it using Whisper.
                - Supports various audio and video formats
                - Automatic language detection
                - Optional AI-powered summarization with Ollama
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
                        with gr.Group():
                            summarize_checkbox = gr.Checkbox(
                                label="Generate AI Summary",
                                value=False,
                                interactive=LLM_AVAILABLE,
                            )
                            ollama_model_dropdown = gr.Dropdown(
                                choices=(
                                    LLM_MODELS
                                    if LLM_AVAILABLE
                                    else ["No models available"]
                                ),
                                value=(
                                    DEFAULT_LLM_MODEL if LLM_AVAILABLE else None
                                ),
                                label="LLM Model",
                                interactive=LLM_AVAILABLE,
                            )

                        # Add status bar
                        file_status = gr.Textbox(
                            label="Status",
                            value="Waiting for input...",
                            interactive=False,
                            elem_classes=["status-bar"],
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
                        if LLM_AVAILABLE:
                            summary_text = gr.Textbox(
                                label="AI Summary", lines=5, max_lines=10, value=""
                            )

                # Set up the event handler
                def transcribe_with_summary(
                    audio, model, lang, summarize, ollama_model
                ):
                    try:
                        if not audio:
                            return "", None, "", "Please upload an audio file"

                        # Update status for each step
                        status = "Loading model..."
                        model = load_model(model)

                        status = "Transcribing audio..."
                        segments, info = model.transcribe(
                            audio,
                            language=lang if lang != "Auto-detect" else None,
                            beam_size=BEAM_SIZE,
                            vad_filter=VAD_FILTER,
                        )

                        # Get the full text with timestamps
                        full_text = " ".join([segment.text for segment in segments])

                        if summarize and LLM_AVAILABLE:
                            status = "Generating AI summary..."
                            summary = llm.summarize(full_text, ollama_model)
                            return (
                                full_text,
                                info.language,
                                summary if summary else "",
                                "Processing complete!",
                            )
                        else:
                            return (
                                full_text,
                                info.language,
                                "",
                                "Processing complete!",
                            )

                    except Exception as e:
                        logger.error(f"Error in transcribe_with_summary: {str(e)}")
                        return f"Error: {str(e)}", None, "", "Processing failed!"

                transcribe_btn.click(
                    fn=transcribe_with_summary,
                    inputs=[
                        audio_input,
                        model_dropdown,
                        language_dropdown,
                        summarize_checkbox,
                        ollama_model_dropdown,
                    ],
                    outputs=[
                        output_text,
                        detected_language,
                        summary_text if LLM_AVAILABLE else gr.Textbox(),
                        file_status,
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
        {"- AI-powered summarization is available for both local files and YouTube videos" if LLM_AVAILABLE else "- AI-powered summarization is currently unavailable"}
        
        ### Status:
        - Device: {DEVICE}
        - Compute Type: {COMPUTE_TYPE}
        - LLM Backend: {ACTIVE_BACKEND}
        - LLM Status: {"Available" if LLM_AVAILABLE else "Not Available"}
        {"- Available LLM Models: " + ", ".join(LLM_MODELS) if LLM_AVAILABLE else ""}
        """
        )

    return app


if __name__ == "__main__":
    logger.info("Starting Whisper Transcription Web App")

    # Check CUDA compatibility before starting
    if not check_cuda_compatibility():
        logger.warning(
            "CUDA compatibility check failed. The application might not work as expected."
        )

    logger.info(f"Server will be available at http://{SERVER_NAME}:{SERVER_PORT}")
    app = create_interface()
    app.launch(share=SHARE, server_name=SERVER_NAME, server_port=SERVER_PORT)
