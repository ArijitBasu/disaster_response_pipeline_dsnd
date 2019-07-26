import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle
nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table('message_categories', engine)
    X = df["message"]
    Y = df.iloc[:, 4:]
    return X, Y, Y.columns


def tokenize(text):
    tokens = WhitespaceTokenizer().tokenize(text)
    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(w).lower().strip() for w in tokens]


def compute_text_length(data):
    return np.array([len(text) for text in data]).reshape(-1, 1)


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([('text', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                                                      ('tfidf', TfidfTransformer()),
                                                      ])),
                                   ('length',
                                    Pipeline([('count', FunctionTransformer(compute_text_length, validate=False))]))]
                                  )),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    parameters = {'features__text__vect__ngram_range': [(1, 2), (2, 2)],
                  'clf__estimator__n_estimators': [50, 100]
                  }
    return GridSearchCV(pipeline, parameters, n_jobs=-1, return_train_score=True, verbose=5)


def multioutput_classification_report(y_true, y_pred, categories):
    for i in range(0, len(categories)):
        print(categories[i] + ":")
        print("\tAccuracy: {:.4f}\tPrecision: {:.4f}\tRecall: {:.4f}\tF1_score: {:.4f}".format(
            accuracy_score(y_true[:, i], y_pred[:, i]),
            precision_score(y_true[:, i], y_pred[:, i], average='weighted'),
            recall_score(y_true[:, i], y_pred[:, i], average='weighted'),
            f1_score(y_true[:, i], y_pred[:, i], average='weighted')
        ))


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    multioutput_classification_report(Y_test.values, y_pred, categories=category_names)

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()