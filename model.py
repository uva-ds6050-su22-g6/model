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

# %% Code Heading [markdown]
#
# # DS6050 - Group 6
# * Andre Erkelens <wsw3fa@virginia.edu>
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


# %% Disable autosave
# %autosave 0

# %% Data setup
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text as tf_text
import tokenizers
import transformers

from tensorflow import keras


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

# %% Baseline Bert Initialization
## This is currently broken - Still tryign to get the TFBertModel to accept the token string in.
max_len = 384
hf_bert_tokenizer_bootstrapper = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
hf_bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")

save_path = Path("data") / "models"
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
hf_bert_tokenizer_bootstrapper.save_pretrained(save_path)
hf_bert_model.save_pretrained(save_path)

# Load the fast tokenizer from saved file
bert_tokenizer = tokenizers.BertWordPieceTokenizer(str(save_path/"vocab.txt"), lowercase=True)

def tf_hf_bertencode(features, label):
    x = bert_tokenizer.encode(tf.compat.as_str(features), add_special_tokens=True)
    y = bert_tokenizer.encode(tf.compat.as_str(label), add_special_tokens=True)
    return x, y

def tf_hf_bertencodeds(features, label):
    encode = tf.py_function(func=tf_hf_bertencode, inp=[features, label], Tout=[tf.int64, tf.int64])
    return encode

encoded_input = ds_train.batch(256).map(tf_hf_bertencodeds)
output = transformers.TFBertModel(config=transformers.PretrainedConfig.from_json_file(str(save_path/"config.json")))
hf_bert = output(encoded_input)


# %% Custom Tokenizer Mode

files = [] # Need to explode train_ds to sep files

tokenizer = tokenizers.BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=True,
    lowercase=True,
)

tokenizer.train(
    files,
    vocab_size=10000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    limit_alphabet=1000,
    wordpieces_prefix="##",
)

# Save the files
tokenizer.save_model(args.out, args.name)
