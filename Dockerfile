FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser && \
    mkdir -p /var/log/supervisor /var/run/supervisor /app/nltk_data && \
    chown -R appuser:appuser /var/log/supervisor /var/run/supervisor /app && \
    chmod -R 755 /var/log/supervisor /var/run/supervisor

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=100 --retries=5 --upgrade pip
RUN pip install --no-cache-dir --timeout=100 --retries=5 -r requirements.txt

RUN pip install --no-cache-dir --timeout=100 --retries=5 \
    flask pymongo kafka-python nltk numpy pandas scikit-learn supervisor gunicorn

RUN python -c "import nltk; nltk.download('vader_lexicon', download_dir='/app/nltk_data'); nltk.download('stopwords', download_dir='/app/nltk_data')" && \
    chown -R appuser:appuser /app/nltk_data

COPY . .

RUN mkdir -p /app/models && \
    chown -R appuser:appuser /app/models && \
    chmod -R 755 /app/models

COPY supervisord.conf /etc/supervisor/supervisord.conf

RUN chown -R appuser:appuser /var/log/supervisor /var/run/supervisor /app && \
    chmod -R 755 /var/log/supervisor /var/run/supervisor /app

EXPOSE 8000

USER appuser
ENV NLTK_DATA=/app/nltk_data
CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]