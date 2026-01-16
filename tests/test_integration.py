"""Integration tests for end-to-end workflows."""
import pytest
from unittest.mock import patch, MagicMock, Mock
import os
import sys

# Check if gradio is available for integration tests
try:
    import gradio
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

gradio_required = pytest.mark.skipif(
    not GRADIO_AVAILABLE,
    reason="Gradio not installed - integration tests require gradio"
)


@gradio_required
class TestTranscriptionWorkflow:
    """Integration tests for complete transcription workflow."""
    
    @patch('app.load_model')
    @patch('app.WhisperModel')
    def test_transcribe_audio_full_workflow(self, mock_whisper_class, mock_load_model, 
                                           sample_audio_path, mock_whisper_model):
        """Test complete audio transcription workflow."""
        from app import transcribe_audio
        
        mock_load_model.return_value = mock_whisper_model
        
        text, lang, summary = transcribe_audio(
            sample_audio_path,
            "large-v3",
            language="en",
            summarize=False
        )
        
        assert text is not None
        assert lang == "en"
        assert summary is None
    
    @patch('app.LLM_AVAILABLE', True)
    @patch('app.load_model')
    @patch('app.llm')
    def test_transcribe_with_summarization(self, mock_llm, mock_load_model, 
                                          sample_audio_path, mock_whisper_model):
        """Test transcription with AI summarization."""
        from app import transcribe_audio
        
        mock_load_model.return_value = mock_whisper_model
        mock_llm.is_available.return_value = True
        mock_llm.summarize.return_value = "Test summary"
        
        text, lang, summary = transcribe_audio(
            sample_audio_path,
            "large-v3",
            language="en",
            summarize=True,
            ollama_model="llama3.1:8b"
        )
        
        assert text is not None
        assert summary == "Test summary"
        mock_llm.summarize.assert_called_once()


@gradio_required
class TestYouTubeWorkflow:
    """Integration tests for YouTube video processing."""
    
    @patch('app.LLM_AVAILABLE', True)
    @patch('app.yt.get_available_subtitles')
    @patch('app.yt.download_subtitles')
    @patch('app.llm')
    def test_youtube_with_subtitles(self, mock_llm, mock_download_subs,
                                    mock_get_subs, sample_youtube_url,
                                    sample_subtitle_content, temp_dir):
        """Test YouTube processing with existing subtitles."""
        from app import process_youtube_url
        
        # Setup mocks
        mock_get_subs.return_value = ["en"]
        subtitle_path = os.path.join(temp_dir, "test.vtt")
        with open(subtitle_path, "w") as f:
            f.write(sample_subtitle_content)
        mock_download_subs.return_value = subtitle_path
        mock_llm.is_available.return_value = True
        mock_llm.summarize.return_value = "Test summary"
        
        text, lang, source, summary = process_youtube_url(
            sample_youtube_url,
            "large-v3",
            summarize=True,
            ollama_model="llama3.1:8b"
        )
        
        assert text is not None
        assert source == "Subtitles"
        assert summary == "Test summary"
    
    @patch('app.yt.get_available_subtitles')
    @patch('app.yt.download_video')
    @patch('app.transcribe_audio')
    @patch('os.remove')
    def test_youtube_without_subtitles(self, mock_remove, mock_transcribe, 
                                       mock_download, mock_get_subs,
                                       sample_youtube_url, sample_audio_path):
        """Test YouTube processing without subtitles (transcription)."""
        from app import process_youtube_url
        
        # Setup mocks
        mock_get_subs.return_value = []
        mock_download.return_value = (sample_audio_path, "Test Video")
        mock_transcribe.return_value = ("Transcribed text", "en", None)
        
        text, lang, source, summary = process_youtube_url(
            sample_youtube_url,
            "large-v3"
        )
        
        assert text == "Transcribed text"
        assert source == "Transcription"
        mock_download.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_remove.assert_called_once()


class TestConfigurationLoading:
    """Integration tests for configuration loading."""
    
    def test_load_config_from_file(self, config_file):
        """Test loading configuration from file."""
        import configparser
        
        config = configparser.ConfigParser()
        config.read(config_file)
        
        assert config["whisper"]["default_model"] == "large-v3"
        assert config["whisper"].getint("beam_size") == 8
        assert config["vllm"]["model"] == "Qwen/Qwen2.5-14B-Instruct"
    
    def test_config_fallback_values(self, config_file):
        """Test configuration fallback values."""
        import configparser
        
        config = configparser.ConfigParser()
        config.read(config_file)
        
        # Test with fallback
        patience = config["whisper"].getfloat("patience", fallback=1.0)
        assert patience == 1.5
        
        # Test missing with fallback
        missing = config["whisper"].get("nonexistent", fallback="default")
        assert missing == "default"


class TestMultiBackendLLM:
    """Integration tests for multi-backend LLM functionality."""
    
    @patch('llm_handler.requests.post')
    @patch('llm_handler.requests.get')
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    def test_vllm_primary_ollama_fallback(self, mock_config, mock_get, mock_post,
                                         mock_vllm_response, sample_config_content):
        """Test vLLM as primary with Ollama fallback."""
        import configparser
        from llm_handler import UnifiedLLMHandler
        
        # Setup config
        config = configparser.ConfigParser()
        config.read_string(sample_config_content)
        mock_config.return_value = config
        
        # vLLM available
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": [{"id": "test-model"}]}
        
        # vLLM succeeds
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_vllm_response
        mock_post.return_value = mock_response
        
        handler = UnifiedLLMHandler()
        assert handler.get_active_backend() == "vllm"
        
        summary = handler.summarize("Test text")
        assert summary is not None


@gradio_required
class TestErrorHandling:
    """Integration tests for error handling across components."""
    
    @patch('app.load_model')
    def test_transcribe_with_invalid_audio(self, mock_load_model):
        """Test graceful error handling with invalid audio file."""
        from app import transcribe_audio
        
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("Invalid audio format")
        mock_load_model.return_value = mock_model
        
        text, lang, summary = transcribe_audio(
            "/nonexistent/file.wav",
            "large-v3"
        )
        
        assert "Error" in text
        assert lang is None
    
    @patch('app.yt.get_available_subtitles')
    def test_youtube_with_invalid_url(self, mock_get_subs):
        """Test error handling with invalid YouTube URL."""
        from app import process_youtube_url
        
        mock_get_subs.side_effect = Exception("Invalid URL")
        
        text, lang, source, summary = process_youtube_url(
            "https://invalid-url.com",
            "large-v3"
        )
        
        assert "Error" in text
        assert source == "Error"


@gradio_required
class TestResourceCleanup:
    """Integration tests for resource cleanup."""
    
    @patch('app.yt.get_available_subtitles', return_value=[])  # No subtitles to force download path
    @patch('app.yt.download_video')
    @patch('app.transcribe_audio')
    def test_temp_file_cleanup_after_transcription(self, mock_transcribe, 
                                                   mock_download, mock_get_subs,
                                                   sample_youtube_url,
                                                   temp_dir):
        """Test that temporary files are cleaned up after processing."""
        from app import process_youtube_url
        
        # Create temp file
        temp_file = os.path.join(temp_dir, "test_audio.wav")
        with open(temp_file, "w") as f:
            f.write("test")
        
        mock_download.return_value = (temp_file, "Test Video")
        mock_transcribe.return_value = ("Text", "en", None)
        
        with patch('app.os.remove') as mock_remove:
            process_youtube_url(sample_youtube_url, "large-v3")
            mock_remove.assert_called_once_with(temp_file)


@gradio_required
class TestConcurrentRequests:
    """Integration tests for handling concurrent requests."""
    
    @patch('app.load_model')
    def test_multiple_transcriptions_concurrent(self, mock_load_model,
                                                mock_whisper_model):
        """Test handling multiple transcription requests."""
        from app import transcribe_audio
        import threading
        
        mock_load_model.return_value = mock_whisper_model
        results = []
        
        def transcribe():
            text, lang, summary = transcribe_audio(
                "/fake/audio.wav",
                "large-v3"
            )
            results.append((text, lang))
        
        # Create multiple threads
        threads = [threading.Thread(target=transcribe) for _ in range(3)]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Verify results
        assert len(results) == 3
        for text, lang in results:
            assert text is not None


@gradio_required
class TestEndToEndWorkflow:
    """Complete end-to-end workflow tests."""
    
    @patch('app.LLM_AVAILABLE', True)
    @patch('app.yt.get_available_subtitles')
    @patch('app.yt.download_video')
    @patch('app.load_model')
    @patch('app.llm')
    def test_complete_youtube_to_summary_workflow(self, mock_llm, mock_load_model,
                                                  mock_download, mock_get_subs,
                                                  sample_youtube_url, sample_audio_path,
                                                  mock_whisper_model):
        """Test complete workflow from YouTube URL to summarized output."""
        from app import process_youtube_url
        
        # Setup: No subtitles, download video, transcribe, summarize
        mock_get_subs.return_value = []
        mock_download.return_value = (sample_audio_path, "Test Video")
        mock_load_model.return_value = mock_whisper_model
        mock_llm.is_available.return_value = True
        mock_llm.summarize.return_value = "Complete summary of the video content"
        
        text, lang, source, summary = process_youtube_url(
            sample_youtube_url,
            "large-v3",
            language="Auto-detect",
            summarize=True,
            ollama_model="llama3.1:8b"
        )
        
        # Verify complete workflow
        assert text is not None
        assert lang == "en"
        assert source == "Transcription"
        assert summary == "Complete summary of the video content"
        
        # Verify call chain
        mock_get_subs.assert_called_once()
        mock_download.assert_called_once()
        mock_load_model.assert_called_once()
        mock_llm.summarize.assert_called_once()
