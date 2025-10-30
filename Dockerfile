# ==============================
#  SUPPORT CHATBOT DOCKERFILE
# ==============================
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Download TextBlob corpora (optional, won't fail build if it doesn't work)
RUN python -m textblob.download_corpora || true

EXPOSE 8000

CMD ["uvicorn", "chatbot:app", "--host", "0.0.0.0", "--port", "8000"]
