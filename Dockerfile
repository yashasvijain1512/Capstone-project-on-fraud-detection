FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	MODEL_PATH=final_model.joblib \
	SCALER_PATH=scaler_for_api.joblib \
	MONITORING_DB_PATH=/app/data/monitoring.db

COPY requirements.docker.txt /tmp/requirements.docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements.docker.txt

COPY . /app
RUN mkdir -p /app/data

EXPOSE 5000 8501

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
