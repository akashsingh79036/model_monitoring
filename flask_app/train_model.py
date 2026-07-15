import os
import pickle
import pandas as pd
import numpy as np
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data (only first time)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ====================================================
# TEXT PREPROCESSING FUNCTIONS
# ====================================================

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text):
    text = "".join([char for char in text if not char.isdigit()])
    return text


def lower_case(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)


def removing_punctuations(text):
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace("'", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)


def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text


df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])

# Dataset should contain:
# text      sentiment

df = df.dropna()

df["text"] = df["content"].astype(str).apply(normalize_text)


X = df["text"]
y = df["sentiment"]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(X)

# ====================================================
# TRAIN TEST SPLIT
# ====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ====================================================
# TRAIN MODEL
# ====================================================

model = MultinomialNB()

model.fit(X_train, y_train)

# ====================================================
# EVALUATION
# ====================================================

predictions = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, predictions))
print()
print(classification_report(y_test, predictions))

# ====================================================
# SAVE MODEL
# ====================================================

os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel saved successfully.")
print("models/model.pkl")
print("models/vectorizer.pkl")