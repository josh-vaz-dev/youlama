# Audio/Video Transcription Web App

A web application for transcribing audio and video files using WhisperX, with support for YouTube videos and optional summarization using Ollama.

## Features

- Transcribe local audio/video files
- Process YouTube videos (with subtitle extraction when available)
- Automatic language detection
- Multiple WhisperX model options
- Optional text summarization using Ollama
- Modern web interface with Gradio
- Configurable settings via config.ini

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- FFmpeg installed on your system
- Ollama (optional, for summarization)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd whisperapp
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg (if not already installed):
- Ubuntu/Debian:
```bash
sudo apt update && sudo apt install ffmpeg
```
- macOS:
```bash
brew install ffmpeg
```
- Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

4. Copy the example configuration file:
```bash
cp .env.example .env
```

5. Edit the configuration files:
- `.env`: Set your environment variables
- `config.ini`: Configure WhisperX, Ollama, and application settings

## Configuration

### Environment Variables (.env)

```ini
# Server configuration
SERVER_NAME=0.0.0.0
SERVER_PORT=7860
SHARE=true
```

### Application Settings (config.ini)

```ini
[whisper]
default_model = base
device = cuda
compute_type = float32
batch_size = 16
vad = true

[app]
max_duration = 3600
server_name = 0.0.0.0
server_port = 7860
share = true

[models]
available_models = tiny,base,small,medium,large-v1,large-v2,large-v3

[languages]
available_languages = en,es,fr,de,it,pt,nl,ja,ko,zh

[ollama]
enabled = false
url = http://localhost:11434
default_model = mistral
summarize_prompt = Please provide a comprehensive yet concise summary of the following text. Focus on the main points, key arguments, and important details while maintaining accuracy and completeness. Here's the text to summarize: 
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:7860
```

3. Use the interface to:
   - Upload and transcribe local audio/video files
   - Process YouTube videos
   - Generate summaries (if Ollama is configured)

## Features in Detail

### Local File Transcription
- Supports various audio and video formats
- Automatic language detection
- Multiple WhisperX model options
- Optional summarization with Ollama

### YouTube Video Processing
- Supports youtube.com, youtu.be, and invidious URLs
- Automatically extracts subtitles if available
- Falls back to transcription if no subtitles found
- Optional summarization with Ollama

### Summarization
- Uses Ollama for text summarization
- Configurable model selection
- Customizable prompt
- Available for both local files and YouTube videos

## Notes

- For better accuracy, use larger models (medium, large)
- Processing time increases with model size
- GPU is recommended for faster processing
- Maximum audio duration is configurable (default: 60 minutes)
- YouTube videos will first try to use available subtitles
- If no subtitles are available, the video will be transcribed
- Ollama summarization is optional and requires Ollama to be running

## License

This project is licensed under the MIT License - see the LICENSE file for details. 