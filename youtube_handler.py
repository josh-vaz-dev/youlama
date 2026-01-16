import re
import os
import tempfile
import shutil
import atexit
from typing import Optional, Tuple, List
import yt_dlp
from urllib.parse import urlparse, parse_qs

# Track temp directories for cleanup
_temp_dirs: List[str] = []

def _cleanup_temp_dirs():
    """Cleanup all temporary directories on exit."""
    for temp_dir in _temp_dirs:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception:
            pass

atexit.register(_cleanup_temp_dirs)


def is_youtube_url(url: str) -> bool:
    """Check if the URL is a valid YouTube URL."""
    youtube_regex = r"(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
    return bool(re.match(youtube_regex, url))


def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats."""
    if not is_youtube_url(url):
        return None

    # Handle youtu.be URLs
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]

    # Handle youtube.com URLs
    parsed_url = urlparse(url)
    if parsed_url.netloc in ["www.youtube.com", "youtube.com"]:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]
        elif parsed_url.path.startswith(("/embed/", "/v/")):
            return parsed_url.path.split("/")[2]

    return None


def get_video_info(url: str) -> dict:
    """Get video information using yt-dlp."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            return ydl.extract_info(url, download=False)
        except Exception as e:
            raise Exception(f"Error fetching video info: {str(e)}")


def download_video(url: str) -> Tuple[str, str]:
    """Download video and return the path to the audio file with high quality extraction."""
    temp_dir = tempfile.mkdtemp()
    _temp_dirs.append(temp_dir)  # Track for cleanup
    output_path = os.path.join(temp_dir, "%(id)s.%(ext)s")

    # High quality audio extraction settings
    ydl_opts = {
        "format": "bestaudio[acodec=opus]/bestaudio[acodec=vorbis]/bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",  # WAV for best Whisper quality
                "preferredquality": "0",  # Lossless
            }
        ],
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        # Quality options
        "prefer_ffmpeg": True,
        "keepvideo": False,
        # Audio normalization for consistent volume
        "postprocessor_args": [
            "-ar", "16000",  # 16kHz sample rate optimal for Whisper
            "-ac", "1",      # Mono channel
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            audio_path = os.path.join(temp_dir, f"{info['id']}.wav")
            # Fallback to mp3 if wav doesn't exist
            if not os.path.exists(audio_path):
                audio_path = os.path.join(temp_dir, f"{info['id']}.mp3")
            return audio_path, info["title"]
        except Exception as e:
            raise Exception(f"Error downloading video: {str(e)}")


def get_available_subtitles(url: str) -> list:
    """Get available subtitles for the video."""
    ydl_opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "skip_download": True,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return list(info.get("subtitles", {}).keys())
        except Exception:
            return []


def download_subtitles(url: str, lang: str = "en") -> Optional[str]:
    """Download subtitles for the video."""
    temp_dir = tempfile.mkdtemp()
    _temp_dirs.append(temp_dir)  # Track for cleanup
    output_path = os.path.join(temp_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "skip_download": True,
        "outtmpl": output_path,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            subtitle_path = os.path.join(temp_dir, f"{info['id']}.{lang}.vtt")
            if os.path.exists(subtitle_path):
                return subtitle_path
            return None
        except Exception:
            return None


def cleanup_temp_file(file_path: str) -> bool:
    """Manually cleanup a temporary file and its parent directory if empty."""
    try:
        if file_path and os.path.exists(file_path):
            parent_dir = os.path.dirname(file_path)
            os.remove(file_path)
            # Remove parent if it's a temp dir and now empty
            if parent_dir.startswith(tempfile.gettempdir()) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)
                if parent_dir in _temp_dirs:
                    _temp_dirs.remove(parent_dir)
            return True
    except Exception:
        pass
    return False
