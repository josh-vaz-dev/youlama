FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Define the path to PyTorch's bundled NVIDIA libraries (adjust if necessary for your specific Python version/setup)
# This path assumes nvidia-cudnn-cuXX or similar packages install here.
ENV PYTORCH_NVIDIA_LIBS_DIR /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib
# Prepend PyTorch's NVIDIA library directory to LD_LIBRARY_PATH
# Also include the standard NVIDIA paths that the base image might set for other CUDA components.
ENV LD_LIBRARY_PATH=${PYTORCH_NVIDIA_LIBS_DIR}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64


# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 7860

# Set entrypoint
ENTRYPOINT ["python3", "app.py"] 