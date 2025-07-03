# updated app.py

from flask import Flask, render_template,request
import pickle
import pandas as pd
import numpy as np
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from prometheus_client import Summary, Counter
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text


# define the metrics
RESPONSE_LATENCY = Summary(name="response_latency_seconds",
                  documentation="Latency of prediction requests in seconds")

TOKENS_COUNTER = Counter(name="input_tokens_total",
                  documentation="Total number of tokens processed in prediction requests")

HAPPY_COUNTER = Counter(name="happy_predictions_total",
                  documentation="Total number of happy predictions made")

SAD_COUNTER = Counter(name="sad_predictions_total",
                  documentation="Total number of sad predictions made")


# make the flask app
app = Flask(__name__)

# Add prometheus wsgi middleware to route /metrics requests
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

# load model 
model = pickle.load(open('models/model.pkl','rb'))

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

def count_tokens(tokens):
    """Process tokens and update the token counter"""
    number_of_tokens = len(tokens)
    # Increment the tokens counter
    TOKENS_COUNTER.inc(number_of_tokens)  
    
    
@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
@RESPONSE_LATENCY.time()  # Measure the latency of the prediction request
def predict():

    text = request.form['text']

    # clean
    text = normalize_text(text)
    
    # split the text into tokens
    tokens = text.split(" ")
    # count tokens
    count_tokens(tokens)

    # bow
    features = vectorizer.transform([text])

    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # prediction
    result = model.predict(features_df)
    
    # get the result
    if result[0] == 1:
        HAPPY_COUNTER.inc()  # Increment happy predictions counter
    elif result[0] == 0:
        SAD_COUNTER.inc()  # Increment sad predictions counter

    # show
    return render_template('index.html', result=result[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)