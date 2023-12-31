FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["gunicorn","wsgi:app","-b","0.0.0.0:8080"]