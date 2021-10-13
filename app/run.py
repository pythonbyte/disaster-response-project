import json
import plotly
import pandas as pd
import joblib
from collections import Counter

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    """Function to tokenize text."""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
database_filepath = '../data/DisasterResponse.db'
engine = create_engine(f'sqlite:///{database_filepath}')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    categories_counts = df.loc[::, 'related':].sum().sort_values(ascending=False)
    categories_names = [i.replace('_', ' ').capitalize() for i in categories_counts.keys()]

    # Logic to return most common words, but takes too much time so
    # I'll pass direct values and I manually filtered the most used ones
    # df['tokens'] = df['message'].apply(lambda x: tokenize(x))
    # results = Counter()
    # df['tokens'].apply(results.update)
    # sorted(results.items(), key=lambda x: -x[1])

    results = [('peopl', 2999),
               ('water', 2943),
               ('food', 2811),
               ('help', 2810),
               ('need', 2747),
               ('pleas', 1954),
               ('earthquak', 1868),
               ('area', 1655),
               ('flood', 1336),
               ('countri', 1220),
               ('thank', 1112),
               ('govern', 1074),
               ('hous', 1040),
               ('inform', 1028),
               ('rain', 1017)]

    graphs = [
        {
            'data': [
                Bar(
                    x=[i[0] for i in results],
                    y=[i[1] for i in results]
                )
            ],

            'layout': {
                'title': 'Distribution of Most Used Words (Normalized)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word text",
                    'tickangle': -40,
                },
            }
        },
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category Name",
                    'tickangle': -40,
                },
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker_color=['steelblue', 'firebrick', 'green']
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    classification_results = dict(
        sorted(classification_results.items(), key=lambda item: -item[1])
    )
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
