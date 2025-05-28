

from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open('model2.py', 'rb'))
tfidf_vect = pickle.load(open('tfidfvect2.pkl', 'rb'))

# Preprocess function
def preprocess_text(news_text):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', news_text)  # Remove non-alphabet characters
    review = review.lower()  # Convert to lowercase
    review = review.split()  # Tokenize into words
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]  # Remove stopwords and stem words
    return ' '.join(review)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get news text from form input
    news_text = request.form.get('news_text')

    # Preprocess the input news
    processed_text = preprocess_text(news_text)

    # Transform the text using the vectorizer
    val = tfidf_vect.transform([processed_text]).toarray()

    # Predict using the model
    prediction = model.predict(val)

    # Return the result
    if prediction[0] == 0:
        result = "Fake News!"
    else:
        result = "Real News!"

    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == '__main__':
    app.run(debug=True)
