import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from laserembeddings import Laser

# class get data 

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
list_path = ['Sent1/sorted_data/apparel/positive.review', 'Sent1/sorted_data/apparel/negative.review']

# loop on list, and get data from files
for i in list_path:
    data.parse_data(i)

# add list into DataFrame
df = pd.DataFrame({
    'review_text' : data.review_text,
    'rating' : data.rating,
    })

# data cleaning 
df['review_text'] = df['review_text'].apply(lambda x: preprocessing(x))
df['rating'] = df['rating'].apply(lambda x: preprocessing(x))


# mixed data
df = df.sample(frac=1).reset_index(drop=True)


# instance laser
laser = Laser()

embed = laser.embed_sentences(df['review_text'], lang='en')

# intit train and test

# split data
X_train = embed[:1600]
X_test = embed[400:]
y_train = df['rating'][:1600]
y_test = df['rating'][400:]

# Fitting a random forest classifier to the training data

text_classifier = RandomForestClassifier(n_estimators = 50)


print("Fitting random forest to training data....")    

text_classifier = text_classifier.fit(X_train, y_train)


predictions = text_classifier.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))
