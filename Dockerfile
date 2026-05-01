FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/package.json ./frontend/
RUN cd frontend && npm install

COPY . .
RUN cd frontend && npm run build

WORKDIR /app/backend
ENV PORT=7860
EXPOSE 7860

CMD ["python", "main.py"]
