import numpy as np
import re
from Stemmer import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


# категорії які є: Технології, Транспорт, Історія, Спорт та Ігра
# чистимо текст
def text_cleaner(text):
    text = text.lower()
    stemmer = Stemmer('ukrainian')
    text = ' '.join(stemmer.stemWords(text.split()))
    text = re.sub(r'\b\d+\b', 'digit', text)


# завантаження даних
def load_data():
    data = {'text': [], 'tag': []}
    for line in open('dataset.txt'):
        if not ('#' in line):
            row = line.split("@")
            data['text'] += [row[0]]
            data['tag'] += [row[1]]
    return data


# навчання
def train_test_split(data, validation_split=0.1):
    sz = len(data['text'])
    indices = np.arange(sz)
    X = [data['text'][i] for i in indices]
    Y = [data['tag'][i] for i in indices]
    nb_validation_samples = int(validation_split * sz)
    return {
        'train': {'x': X[:-nb_validation_samples], 'y': Y[:-nb_validation_samples]},
        'test': {'x': X[-nb_validation_samples:], 'y': Y[-nb_validation_samples:]}
    }


def openai():
    data = load_data()
    D = train_test_split(data)
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SGDClassifier(loss='hinge'))])
    text_clf.fit(D['train']['x'], D['train']['y'])
    predicted = text_clf.predict(D['train']['x'])
    # тест
    z = input('Категорії які є: Технології, Транспорт, Історія, Спорт та Ігра\nВведіть текст для класифікації: ')
    zz = []
    zz.append(z)
    predicted = text_clf.predict(zz)
    print(predicted[0])

openai()
