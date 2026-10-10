FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -e . 2>/dev/null || true

EXPOSE 8765

CMD ["python", "-m", "dmadforge", "serve", "--host", "0.0.0.0", "--port", "8765"]
