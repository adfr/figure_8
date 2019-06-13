import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split,GridSearchCV 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.sklearn_api import W2VTransformer
import tensorflow as tf
import tensorflow.keras
from gensim.models import Word2Vec
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
import pickle
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
# from the lesson's notebook
    def starting_verb(self, text):
        try:
            sentence_list = sent_tokenize(text)
            for sentence in sentence_list:
                pos_tags = pos_tag(tokenize(sentence))
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        except:
            return False
                    
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
	engine = create_engine('sqlite:///'+database_filepath)
	df = pd.read_sql_table(con = engine , table_name = 'Message_label')
	X = df.message 
	Y = df.drop(['id', 'message','original','genre'], axis=1)
	category_names= Y.columns
	return X,Y,category_names

def tokenize(text):
    # First remove punctation and lowercase all letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    clean_tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    # lemmatize, stem and remove stop words
    clean_tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in clean_tokens]
    return clean_tokens


def build_model():
    pipeline  = Pipeline(
            [('vect',CountVectorizer(tokenizer=tokenize)),
             ('tfidf',TfidfTransformer()),
             ('clf' , MultiOutputClassifier(MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(6, 3), random_state=1)) )  
            ]
        )
    parameters ={
        'clf__estimator__hidden_layer_sizes':((32,3), (32,4)),
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (5000, 10000),
        #'clf__estimator__alpha' : (0.00001 ,0.0001 , 0.001 )

    }
    cv  = GridSearchCV(pipeline, param_grid=parameters,verbose=3,n_jobs=4,cv=4)
    return cv
def build_model_verb():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(6, 3), random_state=1)))
        ])
    parameters ={
        'clf__estimator__hidden_layer_sizes':((32,3) ,(32,4)),
#        'vect__ngram_range': ((1, 1), (1, 2)),
#        'vect__max_df': (0.5, 0.75, 1.0),
#        'vect__max_features': (5000, 10000),
#        'clf__estimator__alpha' : (0.00001 ,0.0001 , 0.001 )

    }
    cv  = GridSearchCV(pipeline, param_grid=parameters,verbose=3,n_jobs=4,cv=4)
    return cv

def display_results(y_test, y_pred):
    for i,j in enumerate(y_test.columns):
        print(j)
        confusion_mat = confusion_matrix(y_test[j], y_pred[:,i])
        accuracy = (y_pred[:,i] == y_test[j]).mean()
        print(classification_report(y_test[j], y_pred[:,i]))
        print("Labels:", j)
        print("Confusion Matrix:\n", confusion_mat)
        print("Accuracy:", accuracy)
def evaluate_model(model, X_test, Y_test, category_names):
    y_pred_perceptron2 = model.predict(X_test)


def save_model(model, model_filepath):
	filename = model_filepath
	pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model_verb()
        
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