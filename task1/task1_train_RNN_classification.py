import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

import re
import os
import math


import tensorflow as tf
import tensorflow_hub as hub


from laserembeddings import Laser

# class get data 

class GetData:

    def __init__(self):

        '''
        Constructor
        Init list review_text and rating
        '''    

        self.review_text = []
        self.rating = []

    def parse_data(self, path):

        '''
        parse data frome file .review
        input path of file
        Out add into liste rating and review text the data from file
        '''

        with open(path, encoding="ISO-8859-15") as fp:
            soup = BeautifulSoup(fp, 'html.parser')

        # get all tag review
        reviews = soup.find_all('review')

   
        # run through review to get data 
        for review in reviews:

            # append each review into review list, if doesn't exist append nan
            # I remove eache \n for basic cleaning data 
            try:
                review_text_temp = review.find('review_text').text.replace('\n', '')
                self.review_text.append(review_text_temp)

                rating_temp = review.find('rating').text
                self.rating.append(rating_temp)
            except:
                continue


def preprocessing(sentence):

    '''
    This fuction preprocess the review for embedding
    In data from extration from balise file
    Out data preprocessed 
    '''

    sentence = re.sub(r"<.*?>", "", sentence)

    sentence = sentence.replace("\n", "")

    sentence = sentence.lower()

    return sentence


# instance of object GetData
data = GetData()

folder_list =['books', 'apparel', 'baby', 'camera & photo', 'dvd', 'electronics']

for i in folder_list:
    try:
        data.parse_data('Sent1/sorted_data/'+i+'/positive.review')
        data.parse_data('Sent1/sorted_data/'+i+'/negative.review')
    except:
        continue     

# add list into DataFrame
df = pd.DataFrame({
    'review_text' : data.review_text,
    'rating' : data.rating,
    })

# data cleaning on DataFrame
df['review_text'] = df['review_text'].apply(lambda x: preprocessing(x))
df['rating'] = df['rating'].apply(lambda x: preprocessing(x))


def binary_rating(x):
    '''
    change number 4.0 and 5.o into 1
    2.0 and 1.0 into 0

    '''
    x = x.split('.')[0]
    if int(x) > 3.0:
        result = 1
    else:    
        result = 0
    return result

# change rating into 0 and 1
df['rating'] = df['rating'].apply(lambda x: binary_rating(x))

# mixed data
df = df.sample(frac=1).reset_index(drop=True)

# instance of object Laser
laser = Laser()

# Init feature and target 

# X Feature = embed sentences
X = laser.embed_sentences(df['review_text'], lang='en')

# y Target = convert rating series int np array
y = np.asarray(df['rating']).astype('float32')

# intit train and test

# get 80% of df 
len_08 = math.trunc(len(df)*0.8)

# split data
train_reviews = X[:len_08]
test_reviews = X[len_08:]
train_labels = y[:len_08]
test_labels = y[len_08:]


def get_compiled_model():
    '''
    Build Sequential model for RNN
    '''

    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=len(df),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

# call function get_compiled_model
model = get_compiled_model()

# fit model withe 
model.fit(train_reviews, train_labels, epochs=5, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(test_reviews, test_labels, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# save model
model.save('model_rnn_saved', include_optimizer=False)
