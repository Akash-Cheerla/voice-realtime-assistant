FROM python:3.10-slim

# Install ffmpeg for Whisper compatibility
RUN apt-get update && apt-get install -y ffmpeg

# Set working directory
WORKDIR /app

# Copy files and install dependencies
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8080

# Start the server
CMD sh -c 'uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}'
