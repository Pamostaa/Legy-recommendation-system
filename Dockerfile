# Use minimal Python base
FROM python:3.11-slim-buster

# Install system dependencies (only what’s needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install everything in one layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader stopwords

# Copy the rest of the app
COPY . .

# Expose Flask app port
EXPOSE 8000

# Run the app
CMD ["python", "api.py"]
