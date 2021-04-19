# Phedone

## Installation

$ pip install -r requirements.txt

## Project description

### Task 1:

#### Step 1: Retrieve data from the positive.review and negative.review files.

Open the files in ISO-8859-15
using the Beautifull soup library to parse the data tagged in the files. 
text recovery in <rating> and <review_text>.

#### Step 2: preprocessing, 

removal of tags present in the collected data
text in lower case to unify the words
removal of special characters.

Concerning the reviews that have more than one sentence, I made the choice to keep them as is so that LASER embeddings returns only one numerical representation of the review. Another way to study would be to split the reviews sentence by sentence in order to have a unique numerical representation of each sentence and to assign the rating that is associated to the review.

#### Step 3: Numerical representation of the reviews in (number of sentences*1024) 

#### Step 4: Classification of the reviews according to their numerical representation

Several approaches have been tested for the classification of the reviews.
Randome forest : 
This is a basic model that allows to have a first estimation.
accuracy : 90%.

Sequential classification model (TF Keras):
The neural network is adapted to 
accuracy : 79

RNN classification model (TF Keras):
The recurrent neural network is powerful for modeling sequence data and is suitable for natural language.
accuracy: 54%.


The algorithms were trained where there are as many positive and negative reviews, so as not to bias the algo.


To do: 
 Algorithm:
Improvement of the RNN scores
Create a script dedicated to the prediction of new non-labeled data

Apply a topic extraction to highlight the topics discussed in the positive and negative reviews.
For this extraction, it will be necessary to refine the processing and add stop words.


Major problem encountered during the realization of Task 1:
The main problem was to understand how the numerical representation of LASER embedded should be applied and how to use it.


The models are not accurate enough


What I learned:
Until now I had a quick overview of tensor flow.
This exercise allowed me to deepen my understanding of how to use data with TS and also how to design models.


### Task 2 : Machine translation

General idea for the RNN architecture: 

Seq2Seq and Encoder-Decoder:
We need to transmit input text to the encoder and output text to the decoder. The encoder passes context vectors to the decoder so that the decoder can do its job.


Not having the time to create a real model:
I followed these tracks: 
Tuto pytorch : 
https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/a60617788061539b5449701ae76aee56/seq2seq_translation_tutorial.ipynb

New York Times Multilingual Titles on the laser github : 
https://github.com/ceshine/LASER/blob/master/notebooks/New%20York%20Times%20Multilingual%20Titles.ipynb

I understand that this is not really an answer in itself, the files in question are not on the github.

### Task 3:
Design a dashboard that summarizes the main information of the prediction of Task 1:
Product list, overall rating, number of reviews, distribution of positive or negative feelings and some random example of review.
