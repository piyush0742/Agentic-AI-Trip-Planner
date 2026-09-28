FROM python:3.11-slim

WORKDIR /app

# Dependencies pehle copy/install karo (Docker caching ke liye behtar)
COPY langchain_requirements.txt .
RUN pip install --no-cache-dir -r langchain_requirements.txt

# Ab baaki sab kuch copy karo - code, chroma_db, PDF, sab
COPY . .

# Cloud Run 8080 port expect karta hai by default
EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "langchain_chat:app", "--host", "0.0.0.0", "--port", "8080"]