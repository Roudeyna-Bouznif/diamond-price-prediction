FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
ENV PYTHONPATH=/app/src MODEL_PATH=/app/models/model.joblib

# Train inside the image: the pickled model always matches the installed scikit-learn.
# (downloads the public dataset at build time)
RUN python -m diamonds.train --out /app --no-figures

EXPOSE 8000
CMD ["uvicorn", "diamonds.api:app", "--host", "0.0.0.0", "--port", "8000"]
