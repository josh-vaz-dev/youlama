# Whisper Transcription Web App

A user-friendly web application for transcribing audio and video files using OpenAI's Whisper model, powered by Gradio and faster-whisper.

## Features

- 🎙️ Transcribe audio and video files
- 🚀 GPU acceleration support
- 🌐 Multiple language support
- 📱 Responsive and modern UI
- 🔄 Multiple model options (tiny to large-v3)
- ⚙️ Configurable settings via config.ini
- 📺 YouTube video support with subtitle extraction

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg (for audio/video processing)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd whisperapp
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install uv (recommended package installer):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

4. Install the required packages using uv:
```bash
uv pip install -r requirements.txt
```

## Configuration

The application can be configured through the `config.ini` file. Here are the available settings:

### Whisper Settings
- `default_model`: Default Whisper model to use
- `device`: Device to use (cuda/cpu)
- `compute_type`: Computation type (float16/float32)
- `beam_size`: Beam size for transcription
- `vad_filter`: Enable/disable voice activity detection

### App Settings
- `max_duration`: Maximum audio duration in seconds
- `server_name`: Server hostname
- `server_port`: Server port
- `share`: Enable/disable public sharing

### Models and Languages
- `available_models`: Comma-separated list of available models
- `available_languages`: Comma-separated list of supported languages

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:7860`

3. Choose between two tabs:
   - **Local File**: Upload and transcribe audio/video files
   - **YouTube**: Process YouTube videos with subtitle extraction

### Local File Tab
1. Upload an audio or video file
2. Select your preferred model and language settings
3. Click "Transcribe" and wait for the results

### YouTube Tab
1. Enter a YouTube URL (supports youtube.com, youtu.be, and invidious URLs)
2. Select your preferred model and language settings
3. Click "Process Video"
4. The app will:
   - First try to extract available subtitles
   - If no subtitles are available, download and transcribe the video

## Model Options

- tiny: Fastest, lowest accuracy
- base: Good balance of speed and accuracy
- small: Better accuracy, moderate speed
- medium: High accuracy, slower
- large-v1/v2/v3: Highest accuracy, slowest

## Tips

- For better accuracy, use larger models (medium, large)
- Processing time increases with model size
- GPU is recommended for faster processing
- Maximum audio duration is configurable in config.ini
- Use uv for faster package installation and dependency resolution
- YouTube videos will first try to use available subtitles
- If no subtitles are available, the video will be transcribed

## License

MIT License 