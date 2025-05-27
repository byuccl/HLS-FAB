# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt update && apt install -y python3.12 python3.12-venv python3.12-dev curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Set up a virtual environment
RUN python3.12 -m venv /app/venv

# Activate virtual environment and upgrade pip
RUN /app/venv/bin/pip install --upgrade pip

# Copy the requirements file
COPY requirements.txt .

# Install required Python packages inside the virtual environment
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Use the virtual environment's Python to run the app
CMD ["/app/venv/bin/python", "hls_completions.py"]