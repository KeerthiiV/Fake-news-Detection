import numpy as np # linear algebra
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

true = pd.read_csv(r'D:\fake news detection\archive\true.csv')

true.head()

true.shape

fake = pd.read_csv(r'D:\fake news detection\archive\fake.csv')

fake.shape

true['label'] = 1
fake['label'] = 0

frames = [true.loc[:5000][:], fake.loc[:5000][:]]

df = pd.concat(frames)

df.shape

X = df. drop('label', axis=1)
y = df['label']

df = df.dropna()
df2 = df.copy()

df2.reset_index(inplace=True)
df2.head()

pip install nltk


import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

import re
import nltk
nltk.download('stopwords')

corpus = []
for i in range(0, len(df2)):
    review = re.sub('[^a-zA-Z]', ' ', df2['text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

pip install scikit-learn


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v = TfidfVectorizer(max_features=5000, ngram_range=(1,3))

X = tfidf_v.fit_transform(corpus).toarray()
y = df2['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import PassiveAggressiveClassifier
classifier = PassiveAggressiveClassifier(max_iter=1000)

from sklearn import metrics
import numpy as np
import itertools

classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)

score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

review = re.sub('[^a-zA-Z]', ' ', fake['text'][13070])
review = review.lower()
review = review.split()
    
review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
review = ' '.join(review)
review

val = tfidf_v.transform([review]).toarray()

classifier.predict(val)

import pickle
pickle.dump(classifier, open('model2.py', 'wb'))
pickle.dump(tfidf_v, open('tfidfvect2.pkl', 'wb'))

joblib_model = pickle.load(open('model2.py', 'rb'))
joblib_vect = pickle.load(open('tfidfvect2.pkl', 'rb'))
val_pkl = joblib_vect.transform([review]).toarray()
test_pred = joblib_model.predict(val_pkl)

if test_pred == 0:
    print("Fake News!")
else:
    print("Real News")

import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def test_news(news_text):
    # Initialize Porter Stemmer
    ps = PorterStemmer()

    # Preprocess the input news
    review = re.sub('[^a-zA-Z]', ' ', news_text)  # Remove non-alphabet characters
    review = review.lower()  # Convert to lowercase
    review = review.split()  # Tokenize into words
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]  # Remove stopwords and stem words
    review = ' '.join(review)  # Recombine into a single string

    # Load the trained model and vectorizer
    joblib_model = pickle.load(open('model2.py', 'rb'))
    joblib_vect = pickle.load(open('tfidfvect2.pkl', 'rb'))

    # Transform using the loaded vectorizer
    val = joblib_vect.transform([review]).toarray()

    # Predict using the loaded model
    prediction = joblib_model.predict(val)

    # Output the result
    if prediction == 0:
        return "Fake News!"
    else:
        return "Real News!"


news_text = "Breaking news: Scientists discover a new element that could revolutionize technology!"

# Test the news
result = test_news(news_text)
print(result)

# Test a sample news article
news_text = """
NASA's latest mission has successfully landed on Mars, bringing back valuable data 
about the Red Planet's atmosphere and surface conditions. Scientists are excited 
about the findings, which could pave the way for future human exploration.
"""
result = test_news(news_text)
print(f"Prediction for the sample news: {result}")

news_text = """
World leaders gathered in New York for the annual United Nations General Assembly to discuss global issues, 
including climate change, international security, and economic development. The Secretary-General urged 
countries to work together to meet the targets set by the Paris Agreement, emphasizing the importance of 
collaborative efforts in combating global warming. The assembly also highlighted the ongoing challenges 
faced by developing nations and the need for increased support from the international community.
"""
result = test_news(news_text)
print(f"Prediction for the real news: {result}")


news_text = """
LDF and UDF find themselves on the same side as legal and political battle lines emerge over draft UGC regulations
"""
result = test_news(news_text)
print(f"Prediction for the real news: {result}")


news_text = """
Jaishankar to represent India at Donald Trump’s swearing-in as US President on January 20
 """
result = test_news(news_text)
print(f"Prediction for the real news: {result}")


news_text = """
Game Changer Box Office Collection Day 2: Amid Accusations Of Inflating Numbers, Ram Charan And Kiara Advani's Film Faces A Roadblock
 """
result = test_news(news_text)
print(f"Prediction for the real news: {result}")

news_text = """
A thick blanket of fog covered the national capital on Sunday (January 12, 2025), as the city experienced a cold spell.
 """
result = test_news(news_text)
print(f"Prediction for the real news: {result}")

