# ---
# jupyter:
#   jupytext:
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

# %% [markdown] pycharm={"name": "#%%\n"} colab={"base_uri": "https://localhost:8080/"} id="G9tHAjaRTKTb" outputId="3a59c19e-5e91-47aa-82ce-a01444f6fd2b" tags=[]
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
# ## Nodebook Objective
#
# With this notebook we seek to establish a baseline metric to compare our deep learning model performances.

# %%
import re
import string
import sys

import inflect

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


# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"} id="ywqGr9KRTKTc"
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



# %% colab={"base_uri": "https://localhost:8080/"} id="YIzC0n12To2M" outputId="7a691ce7-8d4d-4986-b4f7-8cb9d061ed78"
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    # %cd /content/drive/MyDrive/ds6050/git/

# %% colab={"base_uri": "https://localhost:8080/"} id="2hUkKWLpTV-3" outputId="59c87b18-aa2f-4142-b37c-6011ba92eb10"
import matplotlib.pyplot as plt, numpy as np, os, pandas as pd, seaborn as sns
df = pd.read_feather("data/dataset.feather")
df['topic'] = df['topic'].str.split('.').str[0]

values = df.summary.apply(text_lowercase).apply(convert_number).apply(remove_punctuation)\
           .apply(remove_whitespace).apply(remove_stopwords).apply(rejoin).apply(lemmatize_word)

values

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"} id="hQH4B_BoTKTd"
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"} colab={"base_uri": "https://localhost:8080/"} id="X3Q8Mui1TKTe" outputId="fb881213-6871-4074-f31d-8a5c23d6cf7b"
le = preprocessing.LabelEncoder()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(values.apply(rejoin))
y = le.fit_transform(df['topic'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = lightgbm.Dataset(X_train, label=y_train)
test_data = lightgbm.Dataset(X_test, label=y_test)

params = {'num_leaves': 31, 'objective': 'multiclass', 'seed' : 42, 'num_class': 7}

num_round = 10
bst = lightgbm.train(params, train_data, num_round, valid_sets = [test_data])

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"} colab={"base_uri": "https://localhost:8080/", "height": 585} id="WVkTmyh6TKTe" outputId="37aa6588-b968-44eb-ec84-a3a5e1a92ba3"
labels = list(df['topic'].unique())
labels.sort()
x_labs = labels
labels.sort(reverse = True)
y_labs = labels

def plot_cm(cm, title = 'Confusion Matrix'):
  fig = plt.figure(figsize = (14,8))
  ax = sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues');
  # labels, title and ticks
  ax.set_xlabel('Predicted category', fontsize = 12);
  ax.set_ylabel('Actual category', fontsize = 12); 
  ax.set_title(title, fontsize = 20); 
  ax.xaxis.set_ticklabels(x_labs, fontsize = 8); 
  ax.yaxis.set_ticklabels(y_labs, fontsize = 8);

  ax.set_facecolor('w')
  fig.set_facecolor('w')
  
  plt.tight_layout()
  plt.show()

y_pred = bst.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)
  
test_cm = confusion_matrix(y_test, y_pred)
plot_cm(test_cm, 'LightGBM Confusion Matrix, Testing Data')

# %% colab={"base_uri": "https://localhost:8080/", "height": 585} id="E17rvap9UyLT" outputId="1b3e7ca3-af43-45b2-ee8a-642256e3c24b"
train_cm = confusion_matrix(y_train, np.argmax(bst.predict(X_train), axis = 1))
plot_cm(train_cm, 'LightGBM Confusion Matrix, Training Data')

# %% colab={"base_uri": "https://localhost:8080/", "height": 585} id="J5ntOahAVeck" outputId="3e0172c8-abb1-4056-f06c-fb036b8d4645"
## plotting f1 scores
test_f1  = f1_score(y_test,  y_pred, average = None)
train_f1 = f1_score(y_train, np.argmax(bst.predict(X_train), axis = 1), average = None)

test_f1  = pd.Series(test_f1).reset_index()\
             .rename(columns = {'index': 'category', 0: 'f1'})
train_f1 = pd.Series(train_f1).reset_index()\
             .rename(columns = {'index': 'category', 0: 'f1'})
test_f1['type']  = 'test'
train_f1['type'] = 'train'
lightgbm_f1 = pd.concat([train_f1, test_f1]).reset_index(drop = True)\
                .replace({'category': {t: idx for idx, t in zip(sorted(df['topic'].unique()), range(7))}})\
                .sort_values(by = ['category', 'type'], ascending = False)

# plotting
fig = plt.figure(figsize = (14,8))
# can't get it to sort alphabetically for some reason
ax = sns.barplot(x = 'category', y = 'f1', hue = 'type', 
                 data = lightgbm_f1, order = list(set(lightgbm_f1.category)));
# labels, title and ticks
ax.set_xlabel('Category', fontsize = 12);
ax.set_ylabel('F1 Score'); 
ax.set_title('F1 Score in Training and Testing Data, LightGBM', fontsize = 20); 
ax.xaxis.set_ticklabels(labels); 

ax.set_facecolor('w')
fig.set_facecolor('w')

plt.tight_layout()
plt.show()

# %% id="RMwHpP2RX-RQ" outputId="764d601d-2c37-4751-924a-b1e38436cbeb" colab={"base_uri": "https://localhost:8080/"}
accuracy_score(y_test, y_pred)
