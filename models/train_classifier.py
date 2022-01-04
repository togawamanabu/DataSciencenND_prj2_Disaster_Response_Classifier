import sys
import pandas as pd
import numpy as np
import time

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.utils import parallel_backend

import pickle

import nltk
nltk.download('omw-1.4')
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('CategorizedMessage', engine)

    print('df shape: ', df.shape)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    return X, y, list(y.columns)

def tokenize(text):
    stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
 
    clean_tokens = []
    for tok in filtered_tokens:
        clean_tokens.append(lemmatizer.lemmatize(tok).lower().strip())

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters =  {
         'clf__estimator__n_estimators':  [10, 50, 100],
        'clf__estimator__min_samples_split': [2, 3],
    }

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=2, verbose=3)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))

    
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as files:
        pickle.dump(model, files)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        with parallel_backend('multiprocessing'):
            print('Training model...')
            model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()