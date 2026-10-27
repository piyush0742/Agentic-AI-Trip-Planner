FROM python:3.11-slim

WORKDIR /app

COPY langchain_requirements.txt .
RUN pip install --no-cache-dir -r langchain_requirements.txt

COPY . .

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "langchain_chat:app", "--host", "0.0.0.0", "--port", "8080"]