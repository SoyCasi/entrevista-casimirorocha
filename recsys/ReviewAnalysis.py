#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
import gensim
import spacy

nltk.download("stopwords")
nltk.download("punkt")
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


# El archivo usado se encuentra en la ruta Recommender Systems/recsys/Data. En la siguiente celda se muestra un ejemplo de lo que contiene.


# ### Preprocesamiento de la Data:
# Para poder aprovechar mejor la data, primero se limpiarán los textos de tal forma que solo se evalue la información relevante que contienen.
# Por tal motivo, se buscan eliminar palabras que no son necesarias como por ejemplo aquellas que esten en otros idiomas, palabras innecesarias y signos de puntuación. Utilizaremos el modulo stopwords de NLTK para este proceso.

# In[6]:


class TextPreprocessor(object):

    stop = set(stopwords.words("spanish"))
    black_list = [
        "más",
        "mas",
        "unir",
        "paises",
        "pais",
        "espa",
        "no",
        "os",
        "a",
        "compa",
        "acompa",
        "off",
        "and",
        "grecia",
        "the",
        "it",
        "to",
        "d",
        "et",
        "dame",
        "il",
        "dans",
        "that",
        "as",
        "for",
        "it",
        "elections",
        "would",
        "this",
        "with",
        "york",
        "obama",
        "chavez",
        "gadafi",
    ]
    additional_stopwords = set(black_list)
    stopwords = stop.union(additional_stopwords)

    def __init__(self):
        self.df_news = pd.read_csv("Data/training_data.csv")
        self.nlp = spacy.load("es_core_news_sm")
        self.bigram = gensim.models.Phrases(self.df_news.review.to_list())

    def cleaner(self, word):
        """Build clean text.
        Input:
            word: a string of tweets

        Output:
            out_text: a list with lemmatize and stemmed and eliminated unnecesary words

        """
        word = re.sub(
            r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
            "",
            word,
            flags=re.MULTILINE,
        )
        word = re.sub(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", "", word)
        word = re.sub(r"ee.uu", "eeuu", word)
        word = re.sub(r"\#\.", "", word)
        word = re.sub(r"\n", "", word)
        word = re.sub(r",", "", word)
        word = re.sub(r"\-", " ", word)
        word = re.sub(r"\.{3}", " ", word)
        word = re.sub(r"a{2,}", "a", word)
        word = re.sub(r"é{2,}", "é", word)
        word = re.sub(r"i{2,}", "i", word)
        word = re.sub(r"ja{2,}", "ja", word)
        word = re.sub(r"á", "a", word)
        word = re.sub(r"é", "e", word)
        word = re.sub(r"í", "i", word)
        word = re.sub(r"ó", "o", word)
        word = re.sub(r"ú", "u", word)
        word = re.sub("[^a-zA-Z]", " ", word)
        list_word_clean = []
        for w1 in word.split(" "):
            if w1.lower() not in self.stopwords:
                list_word_clean.append(w1.lower())

        bigram_list = self.bigram[
            list_word_clean
        ]  # use of bigram anda lemmatization to check if there are word that work better together
        out_text = self.lemmatization(" ".join(bigram_list))
        stemmer = SnowballStemmer(
            "spanish"
        )  # use of stemmer to eliminate suffix in words, NLTK recommends SnowBall but, it can be used other stemmers.
        out_text = self.stemming(out_text, stemmer)
        return out_text

    def lemmatization(self, texts, allowed_postags=["NOUN", "ADJ"]):
        """Lemmatize text.
        Input:
            texts: a list with the words of a text
            allowed_postags: a list with Part of Speech used by spacy. Check out https://spacy.io/usage/linguistic-features for more options.
        Output:
            out_text: a list with lemmatize

        """
        texts_out = [
            token.text
            for token in self.nlp(texts)
            if token.pos_ in allowed_postags
            and token.text not in self.black_list
            and len(token.text) > 2
        ]
        return texts_out

    def stemming(self, text_list, stemmer):
        """Lemmatize text.
        Input:
            texts: a list with the words of a text
            stemmer: Stemmer used by NLTK module. Check out https://www.nltk.org/api/nltk.stem.html#module-nltk.stem for more options.
        Output:
            out_text: a list with stemmed text

        """
        review_clean = []
        for word in text_list:
            stem_word = stemmer.stem(word)  # stemming word
            review_clean.append(stem_word)

        return review_clean

    def build_freqs(self, tweets, ys):
        """Build frequencies.
        Input:
            tweets: a list of tweets
            ys: an m x 1 array with the sentiment label of each tweet
                (either 0 or 1)
        Output:
            freqs: a dictionary mapping each (word, sentiment) pair to its
            frequency
        """
        # Convert np array to list since zip needs an iterable.
        # The squeeze is necessary or the list ends up with one element.
        # Also note that this is just a NOP if ys is already a list.
        yslist = np.squeeze(ys).tolist()

        # Start with an empty dictionary and populate it by looping over all tweets
        # and over all processed words in each tweet.
        freqs = {}
        for y, tweet in zip(yslist, tweets):

            for word in tweet[0]:
                pair = (word, y)
                if pair in freqs:
                    freqs[pair] += 1
                else:
                    freqs[pair] = 1

        return freqs

    def normalize_bad_good_review(self, score):
        """Build numerical score.
        Input:
            score: a text showing if a movie is good or bad

        Output:
            num_score: a numerical represetation for a score

        """
        if score == "buena":
            return 1.0
        else:
            return 0.0

    def process_review(self, df_review_list, df_score):
        x_array = np.asarray(df_review_list)
        y_array = np.asarray(df_score)
        x_train, x_test, y_train, y_test = train_test_split(x_array, y_array)
        x_train = x_train[:, np.newaxis]
        x_test = x_test[:, np.newaxis]
        y_train = y_train[:, np.newaxis]
        y_test = y_test[:, np.newaxis]
        freqs = self.build_freqs(x_train, y_train)

        return x_train, x_test, y_train, y_test, freqs

    def process_one_review(self, review, score):
        x_array = np.asarray(review)
        y_array = np.asarray([score])
        x_array = x_array[:, np.newaxis]
        y_array = y_array[:, np.newaxis]

        return x_array, y_array


class ReviewAnalyzer(object):
    def extract_features(self, review, freqs):
        """
        Input:
            review: a list of words for one tweet
            freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        Output:
            x: a feature vector of dimension (1,3)
        """
        # process_tweet tokenizes, stems, and removes stopwords
        word_l = review

        # 3 elements in the form of a 1 x 3 vector
        x = np.zeros((1, 3))

        # bias term is set to 1
        x[0, 0] = 1

        # loop through each word in the list of words
        for word in word_l:

            # increment the word count for the positive label 1
            x[0, 1] += freqs.get((word, 1.0), 0)

            # increment the word count for the negative label 0
            x[0, 2] += freqs.get((word, 0.0), 0)

        assert x.shape == (1, 3)
        return x

    def extract_all_features(self, x_train, freqs):
        X = np.zeros((len(x_train), 3))
        print(X.shape)
        for i in range(len(x_train)):
            print(x_train[i, 0])
            X[i, :] = self.extract_features(x_train[i, 0], freqs)
            print(X[i, :])
        return X

    def sigmoid(self, z):
        """
        Input:
            z: is the input (can be a scalar or an array)
        Output:
            h: the sigmoid of z
        """

        # calculate the sigmoid of z
        h = 1 / (1 + np.exp(-z))

        return h

    def gradientDescent(self, x, y, theta, alpha, num_iters):
        """
        Input:
            x: matrix of features which is (m,n+1)
            y: corresponding labels of the input matrix x, dimensions (m,1)
            theta: weight vector of dimension (n+1,1)
            alpha: learning rate
            num_iters: number of iterations you want to train your model for
        Output:
            J: the final cost
            theta: your final weight vector
        Hint: you might want to print the cost to make sure that it is going down.
        """

        # get 'm', the number of rows in matrix x
        m = x.shape[0]

        for i in range(0, num_iters):

            # get z, the dot product of x and theta
            z = np.dot(x, theta)

            # get the sigmoid of z
            h = self.sigmoid(z)

            # calculate the cost function
            J = (
                -1.0
                / m
                * (
                    np.dot(y.transpose(), np.log(h))
                    + np.dot((1 - y).transpose(), np.log(1 - h))
                )
            )

            # update the weights theta
            theta = theta = theta - (alpha / m) * np.dot(x.transpose(), (h - y))

        J = float(J)
        return J, theta

    def test_logistic_regression(self, test_x, test_y, freqs, theta):
        """
        Input:
            test_x: a list of tweets
            test_y: (m, 1) vector with the corresponding labels for the list of tweets
            freqs: a dictionary with the frequency of each pair (or tuple)
            theta: weight vector of dimension (3, 1)
        Output:
            accuracy: (# of tweets classified correctly) / (total # of tweets)
        """

        # the list for storing predictions
        y_hat = []

        for tweet in test_x:
            # get the label prediction for the tweet
            y_pred = self.predict_tweet(tweet[0], freqs, theta)

            if y_pred > 0.5:
                # append 1.0 to the list
                y_hat.append(1)
            else:
                # append 0 to the list
                y_hat.append(0)

        # With the above implementation, y_hat is a list, but test_y is (m,1) array
        # convert both to one-dimensional arrays in order to compare them using the '==' operator
        accuracy = (y_hat == np.squeeze(test_y)).sum() / len(test_x)

        return accuracy

    def predict_review(self, review, freqs, theta):
        """
        Input:
            review: a list
            freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
            theta: (3,1) vector of weights
        Output:
            y_pred: the probability of a tweet being positive or negative
        """

        # extract the features of the tweet and store it into x
        x = self.extract_features(review, freqs)

        # make the prediction using x and theta
        y_pred = self.sigmoid(np.dot(x, theta))

        return y_pred
