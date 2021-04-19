import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

import re
import os
import math


import tensorflow as tf
import tensorflow_hub as hub


from laserembeddings import Laser

'''
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
'''

class GetData:

    def __init__(self):

        # Init empty list

        self.review_text = []
        self.rating = []

    def parse_data(self, path):

        '''
        parse data frome file .review
        input path of file
        modifying liste rating and review text
        '''

        with open(path, encoding="ISO-8859-15") as fp:
            soup = BeautifulSoup(fp, 'html.parser')

        # get all tag review
        reviews = soup.find_all('review')

        # Init empty list


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

    """

    This fuction preprocess the review for embedding
    Imput data from extration from balise file

    """

    sentence = re.sub(r"<.*?>", "", sentence)

    sentence = sentence.replace("\n", "")

    sentence = sentence.lower()

    return sentence


# instance of object GetData
data = GetData()

# list of path for negative and positive review
#list_path = ['Sent1/sorted_data/apparel/positive.review', 'Sent1/sorted_data/apparel/negative.review']
#folder_list = [apparel, ]

folder_list =['books', 'apparel', 'baby', 'camera & photo', 'dvd', 'electronics'] #os.listdir('Sent1/sorted_data')

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

print('--------------------------------------')
print('\n')
print(len(df))
print('\n')
print('--------------------------------------')

# data cleaning 
df['review_text'] = df['review_text'].apply(lambda x: preprocessing(x))
df['rating'] = df['rating'].apply(lambda x: preprocessing(x))


def binary_rating(x):
    x = x.split('.')[0]
    if int(x) > 3.0:
        result = 1
    else:    
        result = 0
    return result

df['rating'] = df['rating'].apply(lambda x: binary_rating(x))

# mixed data
df = df.sample(frac=1).reset_index(drop=True)


laser = Laser()
X = laser.embed_sentences(df['review_text'], lang='en')

y = np.asarray(df['rating']).astype('float32')#.reshape((1,-1))

#print(y_train.shape)

# intit train and test

# split data
len_08 = math.trunc(len(df)*0.8)

# split data in train 80%, test 20%
train_reviews = X[:len_08]
test_reviews = X[len_08:]
train_labels = y[:len_08]
test_labels = y[len_08:]

#dataset = tf.data.Dataset.from_tensor_slices((list_review, list_rating))
#dataset = tf.data.Dataset.from_tensor_slices((list_review,y_train))
#train_dataset = dataset.shuffle(len(df)).batch(1)

#print(dataset[:3])

#print(train_dataset)

#test_dataset = dataset.shuffle(len(df)).batch(2)


#for feat, targ in dataset.take(5):
#  print ('Features: {}, Target: {}'.format(feat, targ))

#train_dataset = dataset.shuffle(len(df)).batch(1)

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='softmax'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = get_compiled_model()
model.fit(train_reviews, train_labels, epochs=5, verbose=1)

results = model.evaluate(test_reviews, test_labels)

print(results)

model.save('model_sequentiel_saved', include_optimizer=False)
