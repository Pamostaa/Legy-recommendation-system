FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV NLTK_DATA=/app/nltk_data
RUN mkdir -p /app/nltk_data
RUN python -c "import nltk; nltk.download('vader_lexicon', download_dir='/app/nltk_data'); nltk.download('stopwords', download_dir='/app/nltk_data', quiet=True); nltk.download('punkt', download_dir='/app/nltk_data', quiet=True)"

RUN mkdir -p /app/models/bert_preference_model

COPY . .
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

VOLUME ["/app/models"]

EXPOSE 8000

CMD ["/app/start.sh"]