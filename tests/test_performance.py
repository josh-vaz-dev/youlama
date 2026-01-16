"""Performance and load tests for YouLama."""
import pytest
import time
from unittest.mock import patch, MagicMock
import concurrent.futures

# Check if gradio is available for performance tests
try:
    import gradio
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

gradio_required = pytest.mark.skipif(
    not GRADIO_AVAILABLE,
    reason="Gradio not installed - performance tests require gradio"
)


@gradio_required
class TestPerformance:
    """Performance tests for critical components."""
    
    @patch('app.load_model')
    def test_transcription_performance(self, mock_load_model, mock_whisper_model, 
                                      sample_audio_path):
        """Test transcription performance metrics."""
        from app import transcribe_audio
        
        mock_load_model.return_value = mock_whisper_model
        
        start_time = time.time()
        text, lang, summary = transcribe_audio(
            sample_audio_path,
            "large-v3",
            summarize=False
        )
        elapsed = time.time() - start_time
        
        # Should complete quickly for mock
        assert elapsed < 1.0
        assert text is not None
    
    @patch('llm_handler.requests.post')
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    @patch('llm_handler.UnifiedLLMHandler._check_vllm_availability')
    @patch('llm_handler.UnifiedLLMHandler._check_ollama_availability')
    def test_summarization_performance(self, mock_ollama, mock_vllm, mock_config,
                                      mock_post, mock_vllm_response,
                                      sample_transcript, sample_config_content):
        """Test summarization performance."""
        import configparser
        from llm_handler import UnifiedLLMHandler
        
        config = configparser.ConfigParser()
        config.read_string(sample_config_content)
        mock_config.return_value = config
        mock_vllm.return_value = True
        mock_ollama.return_value = False
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_vllm_response
        mock_post.return_value = mock_response
        
        handler = UnifiedLLMHandler()
        
        start_time = time.time()
        summary = handler.summarize(sample_transcript)
        elapsed = time.time() - start_time
        
        assert elapsed < 1.0  # Mock should be fast
        assert summary is not None


@gradio_required
class TestLoadHandling:
    """Load tests for concurrent request handling."""
    
    @patch('app.load_model')
    def test_concurrent_transcriptions(self, mock_load_model, mock_whisper_model):
        """Test handling multiple concurrent transcription requests."""
        from app import transcribe_audio
        
        mock_load_model.return_value = mock_whisper_model
        
        def transcribe_task(task_id):
            return transcribe_audio(
                f"/fake/audio_{task_id}.wav",
                "large-v3"
            )
        
        # Simulate 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(transcribe_task, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should complete successfully
        assert len(results) == 10
        for text, lang, summary in results:
            assert text is not None
    
    @patch('llm_handler.requests.post')
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    @patch('llm_handler.UnifiedLLMHandler._check_vllm_availability')
    @patch('llm_handler.UnifiedLLMHandler._check_ollama_availability')
    def test_concurrent_summarizations(self, mock_ollama, mock_vllm, mock_config,
                                      mock_post, mock_vllm_response,
                                      sample_config_content):
        """Test handling multiple concurrent summarization requests."""
        import configparser
        from llm_handler import UnifiedLLMHandler
        
        config = configparser.ConfigParser()
        config.read_string(sample_config_content)
        mock_config.return_value = config
        mock_vllm.return_value = True
        mock_ollama.return_value = False
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_vllm_response
        mock_post.return_value = mock_response
        
        handler = UnifiedLLMHandler()
        
        def summarize_task(text):
            return handler.summarize(text)
        
        texts = [f"Sample text {i} for summarization" for i in range(5)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(summarize_task, text) for text in texts]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert len(results) == 5
        assert all(r is not None for r in results)


@gradio_required
class TestMemoryUsage:
    """Tests for memory usage and resource management."""
    
    @patch('app.yt.get_available_subtitles', return_value=[])  # No subtitles to force download path
    @patch('app.yt.download_video')
    @patch('app.transcribe_audio')
    def test_memory_cleanup_after_processing(self, mock_transcribe, mock_download,
                                            mock_get_subs, sample_youtube_url, temp_dir):
        """Test that memory is properly cleaned up after processing."""
        from app import process_youtube_url
        import os
        
        # Create large temp file
        temp_file = os.path.join(temp_dir, "large_audio.wav")
        with open(temp_file, "wb") as f:
            f.write(b"0" * (10 * 1024 * 1024))  # 10MB file
        
        mock_download.return_value = (temp_file, "Test Video")
        mock_transcribe.return_value = ("Text", "en", None)
        
        with patch('app.os.remove') as mock_remove:
            process_youtube_url(sample_youtube_url, "large-v3")
            
            # Verify cleanup was called
            mock_remove.assert_called_once()
            
            # In real scenario, file should be removed
            assert not os.path.exists(temp_file) or mock_remove.called


@gradio_required
class TestScalability:
    """Scalability tests for high-volume scenarios."""
    
    @patch('app.load_model')
    def test_sequential_large_batch(self, mock_load_model, mock_whisper_model):
        """Test processing a large batch of requests sequentially."""
        from app import transcribe_audio
        
        mock_load_model.return_value = mock_whisper_model
        
        batch_size = 50
        start_time = time.time()
        
        for i in range(batch_size):
            text, lang, summary = transcribe_audio(
                f"/fake/audio_{i}.wav",
                "large-v3"
            )
            assert text is not None
        
        elapsed = time.time() - start_time
        avg_time = elapsed / batch_size
        
        # With mocks, should be very fast
        assert avg_time < 0.1


class TestRateLimiting:
    """Tests for rate limiting and throttling."""
    
    @patch('llm_handler.requests.post')
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    @patch('llm_handler.UnifiedLLMHandler._check_vllm_availability')
    @patch('llm_handler.UnifiedLLMHandler._check_ollama_availability')
    def test_api_rate_limiting(self, mock_ollama, mock_vllm, mock_config,
                              mock_post, sample_config_content):
        """Test behavior under API rate limiting scenarios."""
        import configparser
        from llm_handler import UnifiedLLMHandler
        
        config = configparser.ConfigParser()
        config.read_string(sample_config_content)
        mock_config.return_value = config
        mock_vllm.return_value = True
        mock_ollama.return_value = False
        
        # Simulate rate limiting (429 status)
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_post.return_value = mock_response
        
        handler = UnifiedLLMHandler()
        summary = handler.summarize("Test text")
        
        # Should handle gracefully
        assert summary is None


class TestStressTest:
    """Stress tests for extreme conditions."""
    
    @patch('app.load_model')
    def test_rapid_fire_requests(self, mock_load_model, mock_whisper_model):
        """Test system under rapid-fire request scenario."""
        from app import transcribe_audio
        
        mock_load_model.return_value = mock_whisper_model
        
        errors = 0
        successes = 0
        
        # Send 100 rapid requests
        for i in range(100):
            try:
                text, lang, summary = transcribe_audio(
                    f"/fake/audio_{i}.wav",
                    "large-v3"
                )
                if text:
                    successes += 1
            except Exception:
                errors += 1
        
        # Most should succeed with mocks
        assert successes > 90
        assert errors < 10
    
    @patch('llm_handler.requests.post')
    @patch('llm_handler.UnifiedLLMHandler._load_config')
    @patch('llm_handler.UnifiedLLMHandler._check_vllm_availability')
    @patch('llm_handler.UnifiedLLMHandler._check_ollama_availability')
    def test_large_text_summarization(self, mock_ollama, mock_vllm, mock_config,
                                     mock_post, mock_vllm_response,
                                     sample_config_content):
        """Test summarization with very large text input."""
        import configparser
        from llm_handler import UnifiedLLMHandler
        
        config = configparser.ConfigParser()
        config.read_string(sample_config_content)
        mock_config.return_value = config
        mock_vllm.return_value = True
        mock_ollama.return_value = False
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_vllm_response
        mock_post.return_value = mock_response
        
        handler = UnifiedLLMHandler()
        
        # Create very large text (100K characters)
        large_text = "Sample text. " * 10000
        
        start_time = time.time()
        summary = handler.summarize(large_text)
        elapsed = time.time() - start_time
        
        assert summary is not None
        # Should handle large input efficiently
        assert elapsed < 2.0
