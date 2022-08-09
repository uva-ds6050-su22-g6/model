# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": true}
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


# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
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



# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
values = df.summary.apply(text_lowercase).apply(convert_number).apply(remove_punctuation).apply(remove_whitespace).apply(remove_stopwords).apply(rejoin).apply(lemmatize_word)

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
values

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(values.apply(rejoin))

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
le = preprocessing.LabelEncoder()
y = le.fit_transform(df['topic'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = lightgbm.Dataset(X_train, label=y_train)
test_data = lightgbm.Dataset(X_test, label=y_test)

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
params = {'num_leaves': 31, 'objective': 'multiclass', 'seed' : 42, 'num_class': 3}

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
num_round = 10
bst = lightgbm.train(params, train_data, num_round, valid_sets=[test_data])

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
y_pred = bst.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
accuracy_score(y_test, y_pred)
