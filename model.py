# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %% Disable autosave
# %autosave 0

# %% Data setup
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_text as tf_text

np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_feather("data/dataset.feather")
df_train = df.sample(frac = 0.8)
df_test = df.drop(df_train.index)

# %% Key definitions
features = 'content' # feature for the future - add all the datasets ['categories', 'summary', 'content']
label = 'topic'

# %% Parallel Strategy setup
strategy = tf.distribute.MirroredStrategy()

# %% Load data in
with strategy.scope():
    ds_train = tf.data.Dataset.from_tensor_slices(
        (
            df_train[features],
            df_train[label]
        )
    )

    ds_test = tf.data.Dataset.from_tensor_slices(
        (
            df_test[features],
            df_test[label]
        )
    )
del df_train, df_test

# %% Tokenize Data

# Need to look into UTF tokenization using wordpiece
#utf_tokenizer = tf_text.WordpieceTokenizer()

utf_tokenizer = tf_text.WhitespaceTokenizer()

@tf.function
def tokenize_ds(X, label):
    return utf_tokenizer.tokenize(tf_text.normalize_utf8(X)), label

ds_train_tok = ds_train.batch(256).map(tokenize_ds)
ds_test_tok = ds_test.batch(256).map(tokenize_ds)

