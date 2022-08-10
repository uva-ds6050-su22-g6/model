# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="17ed0b92" tags=[]
#
# # DS6050 - Group 6
# * Andrej Erkelens <wsw3fa@virginia.edu>
# * Robert Knuuti <uqq5zz@virginia.edu>
# * Khoi Tran <kt2np@virginia.edu>
#
# ## Abstract
# English is a verbose language with over 69% redundancy in its construction, and as a result, individuals only need to identify important details to comprehend an intended message.
# While there are strong efforts to quantify the various elements of language, the average individual can still comprehend a written message that has errors, either in spelling or in grammar.
# The emulation of the effortless, yet obscure task of reading, writing, and understanding language is the perfect challenge for the biologically-inspired methods of deep learning.
# Most language and text related problems rely upon finding high-quality latent representations to understand the task at hand. Unfortunately, efforts to overcome such problems are limited to the data and computation power available to individuals; data availability often presents the largest problem, with small, specific domain tasks often proving to be limiting.
# Currently, these tasks are often aided or overcome by pre-trained large language models (LLMs), designed by large corporations and laboratories.
# Fine-tuning language models on domain-specific vocabulary with small data sizes still presents a challenge to the language community, but the growing availability of LLMs to augment such models alleviates the challenge.
# This paper explores different techniques to be applied on existing language models (LMs), built highly complex Deep Learning models, and investigates how to fine-tune these models, such that a pre-trained model is used to enrich a more domain-specific model that may be limited in textual data.
#
# ## Project Objective
#
# We are aiming on using several small domain specific language tasks, particularly classification tasks.
# We aim to take at least two models, probably BERT and distill-GPT2 as they seem readily available on HuggingFace and TensorFlow's model hub.
# We will iterate through different variants of layers we fine tune and compare these results with fully trained models, and ideally find benchmarks already in academic papers on all of the datasets.
#
# We aim to optimize compute efficiency and also effectiveness of the model on the given dataset. Our goal is to find a high performing and generalizable method for our fine tuning process and share this in our paper.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="f8971652" outputId="34c21793-ca21-4153-8fd9-474c66b16710"
# %autosave 0

# %% colab={"base_uri": "https://localhost:8080/"} id="M7fV1R567PdR" outputId="7e8bc1b5-5d77-45b4-86d8-001d53cd28ae"
# !pip install -q tensorflow-text tokenizers transformers

# %% id="CUp8f1yr760r"
import tensorflow as tf
import tensorflow_text as tf_text

# %% colab={"base_uri": "https://localhost:8080/"} id="vA3eBwuu-Dlf" outputId="1a5149ca-72ac-4d4c-8ba1-684dc3cd9877"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="j8jU9Dao-ZL5" outputId="c4af1e84-182a-4597-e850-ae3f3f944e8a"
# %cd /content/drive/MyDrive/ds6050/git/

# %% colab={"base_uri": "https://localhost:8080/"} id="ebdqcJVAqiyZ" outputId="9a10d30d-ae97-4c08-9c7c-433ffae6b436"
# !ls

# %% id="6eb2e492"
import os
from pathlib import Path

import numpy as np
import pandas as pd

import tokenizers
import transformers

from tensorflow import keras


np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_feather("data-extractor/data/dataset.feather")
df['topic'] = df['topic'].str.split('.').str[0]
df_train = df.sample(frac = 0.8)
df_test = df.drop(df_train.index)

# %% colab={"base_uri": "https://localhost:8080/"} id="-ATWAjiKzMIU" outputId="a7b2123b-0047-4f2c-c2ae-d083b3fbcdf5"
df.topic.unique()

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="trEGVefrzH50" outputId="50d957ad-d7f9-4207-810d-8e55a53df69b"
df=df[df['Topic'].isin(['biology','political-science'])]

# %% id="9wB9AJWdzimV"
pd.set_option('display.max_rows', None)

# %% id="D89ExmAezcJu"
df = df.drop(columns=['index'])

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="bTQlmx2f7jJF" outputId="84832632-5a15-4412-a7e4-4a5c9a9bcc6a"
df.head()

# %% id="ae8d1c35"
features = 'content' # feature for the future - add all the datasets ['categories', 'summary', 'content']
label = 'topic'

# %% id="aa1994aa"
# strategy = tf.distribute.MirroredStrategy()

# %% id="JkZm9g-J6Cge"
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

y_ = ohe.fit_transform(df['topic'].values.reshape(-1,1)).toarray()

# %% colab={"base_uri": "https://localhost:8080/"} id="k98DISb3X2Yr" outputId="ccab43a3-239f-4d4d-cd18-69fde94173d2"
max_len = 512
hf_bert_tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
hf_bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
# hf_bert_model = transformers.TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

# %% id="QukShKq2lnak"
train_encodings = hf_bert_tokenizer.batch_encode_plus(list(df_train.summary.values), 
                                                return_tensors='tf', 
                                                padding='max_length',
                                                max_length=None,
                                                truncation=True)

test_encodings = hf_bert_tokenizer.batch_encode_plus(list(df_test.summary.values), 
                                                return_tensors='tf', 
                                                padding='max_length',
                                                max_length=None,
                                                truncation=True)

# %% id="eLGCVgmFK1iz"
encodings = hf_bert_tokenizer.batch_encode_plus(list(df.summary.values), 
                                                return_tensors='tf', 
                                                padding='max_length',
                                                max_length=None,
                                                truncation=True)


# %% [markdown] id="QaGRZOWyWRuA"
# ## LightGBM Model Comparison

# %% colab={"base_uri": "https://localhost:8080/"} id="NHZoSTe4Z3Rf" outputId="4c0a6e68-a22c-4fda-b6ec-fd6f682094e4"
import re
import inflect
import string

from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer = PorterStemmer()

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()


# %% id="npQhkq6XZYol"
def text_lowercase(text):
    return text.lower()
p = inflect.engine()
 
# convert number into words
def convert_number(text):
    # split string into list of words
    temp_str = text.split()
    # initialise empty list
    new_string = []
 
    for word in temp_str:
        # if word is a digit, convert the digit
        # to numbers and append into the new_string list
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)
 
        # append the word as it is
        else:
            new_string.append(word)
 
    # join the words of new_string to form a string
    temp_str = ' '.join(new_string)
    return temp_str

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_whitespace(text):
    return  " ".join(text.split())

# remove stopwords function
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text

# stem words in the list of tokenized words
def stem_words(text):
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    return stems

# lemmatize string
def lemmatize_word(text):
    word_tokens = word_tokenize(text)
    # provide context i.e. part-of-speech
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens]
    return lemmas

def rejoin(text):
    return ' '.join(text)



# %% id="doSyDOwAZZhT"
values = df.summary.apply(text_lowercase).apply(convert_number).apply(remove_punctuation).apply(remove_whitespace).apply(remove_stopwords).apply(rejoin).apply(lemmatize_word)

# %% id="N0TBVPFIfF_n"
from sklearn.feature_extraction.text import TfidfVectorizer

# %% id="fNmt8aMXfG_g"
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(values.apply(rejoin))

# %% colab={"base_uri": "https://localhost:8080/"} id="6dC6rT05fZJ5" outputId="e6eb9d74-98f4-4eaa-b850-ab9388d3ee0d"
X.shape

# %% id="KvBD4_nSfgXh"
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# %% id="qgglLkGtf-0g"
le = preprocessing.LabelEncoder()
y = le.fit_transform(df['topic'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = lightgbm.Dataset(X_train, label=y_train)
test_data = lightgbm.Dataset(X_test, label=y_test)

# %% id="G8NHFYkOgOe6"
params = {'num_leaves': 31, 'objective': 'multiclass', 'seed' : 42, 'num_class': 7} 

# %% colab={"base_uri": "https://localhost:8080/"} id="9YaUkttGg-RB" outputId="49da3e87-6c05-460c-fb0e-5b802bea4cc0"
num_round = 10
bst = lightgbm.train(params, train_data, num_round, valid_sets=[test_data])

# %% id="sW_0FYSNhrGD"
y_pred = bst.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# %% colab={"base_uri": "https://localhost:8080/"} id="RTzZtI3BkxdZ" outputId="732609d9-2856-41b8-8dce-323f4dff7ede"
accuracy_score(y_test, y_pred)


# %% [markdown] id="aRnAZ2N8aTUY"
# ---

# %% id="rIYVgwNF1KSI"
def model_top(pretr_model):
  input_ids = tf.keras.Input(shape=(512,), dtype='int32')
  attention_masks = tf.keras.Input(shape=(512,), dtype='int32')

  output = pretr_model([input_ids, attention_masks])
  #pooler_output = output[1]
  #pooler_output = tf.keras.layers.AveragePooling1D(pool_size=512)(output[0])
  #flattened_output = tf.keras.layers.Flatten()(pooler_output)
  
  output = tf.keras.layers.Dense(512, activation='tanh')(output[1])
  output = tf.keras.layers.Dropout(0.2)(output)

  output = tf.keras.layers.Dense(7, activation='softmax')(output)
  model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model


# %% id="Ocu7YG7o2TCB"
model = model_top(hf_bert_model)

# %% colab={"base_uri": "https://localhost:8080/"} id="XaMC57JY4YRv" outputId="585d3637-e112-4dca-d583-7b6d5c6b2b25"
model.layers

# %% id="bc4MEFDK4b0b"
model.layers[2].trainable = False

# %% id="L6HcsyLKJUNn"
model.layers[3].trainable = True

# %% colab={"base_uri": "https://localhost:8080/"} id="zMC_xhvt4hfL" outputId="ec7517c5-67f0-403d-c0e2-4670dcaeed05"
model.summary()

# %% id="-5Rt2wQFYfuE"
checkpoint_filepath = './tmp/checkpoint'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='train_accuracy',
    mode='max',
    save_best_only=True)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="auto",
)



# %% colab={"base_uri": "https://localhost:8080/"} id="m11hJnqM4kc0" outputId="a1f0f764-62e0-4f62-c1dd-545fd893a522"
history = model.fit([encodings['input_ids'], 
                     encodings['attention_mask']], 
                    y_, 
                    validation_split=.2,
                    epochs=10,
                    batch_size=64,
                    shuffle=True,
                    callbacks=[model_checkpoint_callback, early_stopping_callback])

# %% id="hvCeKeivsE_5"
train_labels = df_train['topic']
test_labels = df_test['topic']

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings),
                                                         train_labels))

test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings),
                                                        test_labels))
