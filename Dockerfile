FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

# Render will inject its own PORT
ENV PORT=10000

EXPOSE 10000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "10000"]
