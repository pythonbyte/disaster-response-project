"""Module responsible for training the Classifier."""
import pickle
import re
import sys
from typing import List

import nltk
import numpy as np
import pandas as pd
import sqlalchemy

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

nltk.download(['punkt', 'stopwords', 'wordnet'])


def load_data(database_filepath: str):
    """
    Load database from the database_filepath.

    Args:
        database_filepath (str): Filepath of the database.

    Returns:
        X (numpy.ndarray[str]): Array of disaster messages
        y (pd.DataFrame): DF with all categories and its 0 and 1 values
        category_names (list[str]): List of all the categories columns
    """
    engine = sqlalchemy.create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_messages', engine)
    X = df.message.values
    y = df.loc[::, 'related':]
    category_names = y.columns

    return X, y, category_names


def tokenize(text: str) -> List[str]:
    """
    Clean, normalize, tokenize and process text parameter.

    Args:
        text (str): String phrase with text.

    Returns:
        stemmed_tokens (list[str]): Cleaned text
    """
    # set to lower case
    text = text.lower()

    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # replace double empty spaces to one
    text = text.replace('  ', ' ').strip()

    # tokenize the text
    tokens = word_tokenize(text)

    # remove stop words
    cleaned_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # lemmatize the tokens
    lemmatized_tokens = [WordNetLemmatizer().lemmatize(token) for token in cleaned_tokens]

    # stemming the tokens
    stemmed_tokens = [PorterStemmer().stem(token) for token in lemmatized_tokens]

    return stemmed_tokens


def build_model() -> Pipeline:
    """
    Build pipeline model.

    Uses CountVectorizer, TfidfTransformer and a
    MultiOutputClassifier with RandomForestClassifier.

    Returns:
        pipeline (sklearn.Pipeline): Pipeline model.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model: Pipeline, X_test: np.ndarray,
                   Y_test: pd.DataFrame, category_names: str):
    """
    Evaluate and prints the model precision, recall and F1 score.

    Args:
        model (sklearn.Pipeline): Pipeline model.
        X_test (numpy.ndarray): Test data
        Y_test (pd.DataFrame): Df for testing
        category_names (List[str]): List of all the categories columns
    """
    y_pred = model.predict(X_test)
    # classification report
    print(classification_report(
        Y_test,
        y_pred,
        target_names=category_names
    ))


def save_model(model: Pipeline, model_filepath: str):
    """
    Save the model on a pickle file.

    Args:
        model (sklearn.Pipeline): Pipeline model.
        model_filepath (str): Filepath to save the pkl file
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    """Main function to process the classifier logic."""
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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
