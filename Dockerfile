FROM python:3.12-slim

WORKDIR /app

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

COPY models/model.pkl /app/models/model.pkl

COPY flask_app/ /app/

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 8000

CMD ["python", "app.py"]