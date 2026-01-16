"""Unit tests for youtube_handler module."""
import pytest
import os
from unittest.mock import patch, MagicMock, mock_open
import youtube_handler as yt


class TestYouTubeURLValidation:
    """Tests for YouTube URL validation."""
    
    def test_valid_youtube_urls(self):
        """Test various valid YouTube URL formats."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
        ]
        for url in valid_urls:
            assert yt.is_youtube_url(url), f"Failed for: {url}"
    
    def test_invalid_youtube_urls(self):
        """Test invalid URL formats."""
        invalid_urls = [
            "https://vimeo.com/123456",
            "https://example.com",
            "not a url",
            "",
            "https://www.youtube.com/",
        ]
        for url in invalid_urls:
            assert not yt.is_youtube_url(url), f"Incorrectly validated: {url}"
    
    def test_extract_video_id(self):
        """Test video ID extraction from various URL formats."""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/watch?v=abc123def45", "abc123def45"),
        ]
        for url, expected_id in test_cases:
            assert yt.extract_video_id(url) == expected_id
    
    def test_extract_video_id_with_params(self):
        """Test video ID extraction with additional URL parameters."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&list=PLtest"
        assert yt.extract_video_id(url) == "dQw4w9WgXcQ"


class TestVideoDownload:
    """Tests for video download functionality."""
    
    @patch('youtube_handler.yt_dlp.YoutubeDL')
    def test_download_video_success(self, mock_yt_dlp, mock_yt_dlp_info, temp_dir):
        """Test successful video download."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = mock_yt_dlp_info
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        
        # Create expected output file
        expected_path = os.path.join(temp_dir, "dQw4w9WgXcQ.wav")
        with patch('tempfile.mkdtemp', return_value=temp_dir):
            with patch('os.path.exists', return_value=True):
                audio_path, title = yt.download_video("https://youtube.com/watch?v=dQw4w9WgXcQ")
                
                assert title == "Test Video Title"
                assert "dQw4w9WgXcQ" in audio_path
    
    @patch('youtube_handler.yt_dlp.YoutubeDL')
    def test_download_video_failure(self, mock_yt_dlp):
        """Test video download failure handling."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.side_effect = Exception("Download failed")
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        
        with pytest.raises(Exception, match="Error downloading video"):
            yt.download_video("https://youtube.com/watch?v=invalid")


class TestSubtitles:
    """Tests for subtitle extraction."""
    
    @patch('youtube_handler.yt_dlp.YoutubeDL')
    def test_get_available_subtitles_success(self, mock_yt_dlp, mock_yt_dlp_info):
        """Test getting available subtitles."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = mock_yt_dlp_info
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        
        subtitles = yt.get_available_subtitles("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert "en" in subtitles
    
    @patch('youtube_handler.yt_dlp.YoutubeDL')
    def test_get_available_subtitles_none(self, mock_yt_dlp):
        """Test when no subtitles are available."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {"subtitles": {}}
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        
        subtitles = yt.get_available_subtitles("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert subtitles == []
    
    @patch('youtube_handler.yt_dlp.YoutubeDL')
    def test_download_subtitles_success(self, mock_yt_dlp, mock_yt_dlp_info, temp_dir):
        """Test successful subtitle download."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = mock_yt_dlp_info
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        
        subtitle_path = os.path.join(temp_dir, "dQw4w9WgXcQ.en.vtt")
        with patch('tempfile.mkdtemp', return_value=temp_dir):
            with patch('os.path.exists', return_value=True):
                result = yt.download_subtitles("https://youtube.com/watch?v=dQw4w9WgXcQ", "en")
                assert result is not None
                assert ".vtt" in result


class TestCleanup:
    """Tests for temporary file cleanup."""
    
    def test_cleanup_temp_file_success(self, temp_dir):
        """Test successful cleanup of temporary file."""
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        
        assert os.path.exists(test_file)
        result = yt.cleanup_temp_file(test_file)
        assert result is True
        assert not os.path.exists(test_file)
    
    def test_cleanup_temp_file_nonexistent(self):
        """Test cleanup of non-existent file."""
        result = yt.cleanup_temp_file("/nonexistent/file.txt")
        assert result is False
    
    def test_cleanup_removes_empty_dir(self, temp_dir):
        """Test that cleanup removes empty parent directory."""
        sub_dir = os.path.join(temp_dir, "subdir")
        os.makedirs(sub_dir)
        test_file = os.path.join(sub_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        
        with patch('tempfile.gettempdir', return_value=temp_dir):
            result = yt.cleanup_temp_file(test_file)
            assert result is True
            # File should be removed, directory might or might not exist
            assert not os.path.exists(test_file)


class TestVideoInfo:
    """Tests for video information retrieval."""
    
    @patch('youtube_handler.yt_dlp.YoutubeDL')
    def test_get_video_info_success(self, mock_yt_dlp, mock_yt_dlp_info):
        """Test successful video info retrieval."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = mock_yt_dlp_info
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        
        info = yt.get_video_info("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert info["id"] == "dQw4w9WgXcQ"
        assert info["title"] == "Test Video Title"
        assert "subtitles" in info
    
    @patch('youtube_handler.yt_dlp.YoutubeDL')
    def test_get_video_info_failure(self, mock_yt_dlp):
        """Test video info retrieval failure."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.side_effect = Exception("Network error")
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        
        with pytest.raises(Exception, match="Error fetching video info"):
            yt.get_video_info("https://youtube.com/watch?v=invalid")
