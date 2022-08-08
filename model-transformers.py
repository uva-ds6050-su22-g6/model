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

# %% [markdown] id="17ed0b92"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="f8971652" outputId="22d53531-3414-4096-de2d-83c2a8ed56a4"
# %autosave 0
import sys
import os
from pathlib import Path

# %% colab={"base_uri": "https://localhost:8080/"} id="M7fV1R567PdR" outputId="9285860d-4b3d-481e-c305-ab1ebd5d8ef2"
if 'google.colab' in sys.modules:
    # %tensorflow_version 2.0
    # %pip install -q tensorflow-text tokenizers transformers
    from google.colab import drive
    drive.mount('/content/drive')
    # %cd /content/drive/MyDrive/ds6050/
    pass # needed for py:percent script
# %% id="CUp8f1yr760r"
import tensorflow as tf
import tensorflow_text as tf_text

# %%
strategy = tf.distribute.MirroredStrategy()

# %%
#@title Hyperparameters

SEED=42
TRAIN_TEST_SPLIT=0.8
BATCH_SIZE=4
EPOCHS=10
LABEL='topic'
FEATURES='content'
PRETRAINED_WEIGHTS='bert-base-uncased'

# %%
features = FEATURES # feature for the future - add all the datasets ['categories', 'summary', 'content']
label = LABEL

# %% id="6eb2e492"
import numpy as np
import pandas as pd

import tokenizers
import transformers

from tensorflow import keras


np.random.seed(SEED)
tf.random.set_seed(SEED)

df = pd.read_feather("data/dataset.feather")
df[label] = df[label].str.split('.').str[0]

response_count = len(df[label].unique())

df_train = df.sample(frac = TRAIN_TEST_SPLIT)
df_test = df.drop(df_train.index)

# %% id="aa1994aa"
# strategy = tf.distribute.MirroredStrategy()

# %% id="JkZm9g-J6Cge"
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

y_ = ohe.fit_transform(df[label].values.reshape(-1,1)).toarray()

# %% colab={"base_uri": "https://localhost:8080/", "height": 254, "referenced_widgets": ["04aa9aa37f49402493bcf49ef2fcfc98", "87f9c3d4acc44d4d9e1383b3a7d807e5", "0bb5c399c8e24994bfd388e3869d810f", "aada1128811243d69840be5b217f7f20", "e45da87b38ac44ef948c3da7f84e18db", "ec072ac0b2c142fea6a5ff4798d55437", "695764774fc743b2b7c188fa97b242e7", "72f950028f7f46b7a07c561ac762abc6", "836056f24f174353a78c306e269d6c66", "5d4209f06afa4b70a7474f41a7736933", "654819517b6447b68d5d5a5ec68828d5", "a9136a5b122c4167a4ce62bd4648144d", "c68ca26f994b43bda84ee1b883e68e08", "be115c32c9f84fca9f65e0d1e64a09d3", "f4e5380ed2cd412d9b5c14679767eb6a", "b6ac2c6b17fe451daf67083d60be97d0", "b55031fb1376422090a7fa770498a6de", "6da855b690974635b805756bb737edb9", "cf4cb3ea9d404fb28b51c8da7cb67a3e", "79275c53e5154ee191602bdf34849101", "187205fe7fba43de889886ad84d9ffb9", "f2c1be6ffad34ff3a04688ff26fa99a9", "73c1daf1684846338fe30523c5c34305", "36f23d119f9e443496dbad65d9fdb866", "52de36b647bc48af96ec7fce4931dac0", "92be402421924bfd91482fbe9b1cf229", "cc8f336ef54847239865b4567581edb3", "96474c57912c45efb641af3f0affcdfe", "ffaad64b9f044be686a3144e63b377dd", "cd17d324507448efbc9202894d737e99", "afceda9bb12e43eeb6faa3de0938c131", "25bf161cb1ca4b528442f09a2246912c", "0e949019d27e4670b45fae1d56d13c1d", "ee13c34ffe634ea886bee3a6f4559331", "bf59ac2c8411402390d00839dcf81d07", "0c94a978c326498fb3448960a56efef7", "075314ac82834c6bba9a3686b4fa5e84", "86e39f47c7864565b36d01a9ee672400", "995d3d2cffd84a2b9f4b7e6a717999d0", "ee201bdb7c5a404bb06fb4be911ce845", "411224c49d5048a1a38944dcbac8a1fa", "811a039984184e04bbcc5441f009ab47", "4d577587518d4b4db004a35cfd795651", "ff1e7a44bc8143558982857431d8621e"]} id="k98DISb3X2Yr" outputId="447f7763-a53b-47d7-bcf9-a793758db613"
max_len = 512
hf_bert_tokenizer = transformers.BertTokenizerFast.from_pretrained(PRETRAINED_WEIGHTS)
with strategy.scope():
    hf_bert_model = transformers.TFBertModel.from_pretrained(PRETRAINED_WEIGHTS)
# hf_bert_model = transformers.TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

# %% id="eLGCVgmFK1iz"
encodings = hf_bert_tokenizer.batch_encode_plus(list(df.summary.values), 
                                                return_tensors='tf', 
                                                padding='max_length',
                                                max_length=None,
                                                truncation=True)


# %% id="rIYVgwNF1KSI"
def model_top(pretr_model):
  with strategy.scope():
      input_ids = tf.keras.Input(shape=(512,), dtype='int32')
      attention_masks = tf.keras.Input(shape=(512,), dtype='int32')

      output = pretr_model([input_ids, attention_masks])
      #pooler_output = output[1]
      pooler_output = tf.keras.layers.AveragePooling1D(pool_size=512)(output[0])
      flattened_output = tf.keras.layers.Flatten()(pooler_output)

      output = tf.keras.layers.Dense(32, activation='tanh')(flattened_output)
      output = tf.keras.layers.Dropout(0.2)(output)

      output = tf.keras.layers.Dense(response_count, activation='softmax')(output)
      model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model


# %% id="Ocu7YG7o2TCB"
with strategy.scope():
    model = model_top(hf_bert_model)

# %% colab={"base_uri": "https://localhost:8080/"} id="bHO0s2_z2e07" outputId="3568f909-7380-437d-ad73-28d9ddbe6eb1"
model.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="XaMC57JY4YRv" outputId="bba03a2d-af49-4663-c7c6-760a85b13bf8"
model.layers

# %% id="bc4MEFDK4b0b"
model.layers[2].trainable = False

# %% colab={"base_uri": "https://localhost:8080/"} id="zMC_xhvt4hfL" outputId="6e53259e-f64e-440a-8e4e-bccd2affcca4"
model.summary()

# %% colab={"base_uri": "https://localhost:8080/"} id="Ky67OTAeUjwf" outputId="7b6e5aac-ea72-44a4-b6cd-5cf6bc4e9319"
# !nvidia-smi

# %% colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="m11hJnqM4kc0" outputId="675e90b3-eb55-424e-c476-01425faa0e20"
with strategy.scope():
    checkpoint_filepath = './tmp/checkpoint'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="auto",
    )
    history = model.fit([encodings['input_ids'], 
                         encodings['attention_mask']], 
                        y_, 
                        validation_split=1-TRAIN_TEST_SPLIT,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=[model_checkpoint_callback, early_stopping_callback])

# %% id="hvCeKeivsE_5"
train_labels = df_train[LABEL]
test_labels = df_test[LABEL]

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings),
                                                         train_labels))

test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings),
                                                        test_labels))

# %% colab={"base_uri": "https://localhost:8080/"} id="RW2PaFMTuHww" outputId="f8ad3a6a-f6ba-40a3-ee42-976e6b1cd51e"
training_args.strategy.scope()

# %% colab={"base_uri": "https://localhost:8080/"} id="mHVXnOWZzVc7" outputId="e68e673f-58ea-49e7-e6b2-f0985c25bdc6"
train_encodings['input_ids']

# %% colab={"base_uri": "https://localhost:8080/", "height": 550} id="OIyPDA64zYjY" outputId="a113ac63-b015-4d6e-e8e9-6e74db0d61e7"
hf_bert_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['acc'])

hf_bert_model.fit(train_encodings['input_ids'])

# %% colab={"base_uri": "https://localhost:8080/", "height": 746} id="RLlGZYeruxnj" outputId="568a23c4-9da7-40f7-c1c3-b0bdc444cc0d"
hf_bert_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['acc'])

hf_bert_model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset)

# %% colab={"base_uri": "https://localhost:8080/"} id="Lm7HnV7VuEjD" outputId="793d41e8-7d39-4ab3-9c15-ea42da81f41f"
with training_args.strategy.scope():
  model = hf_bert_model

trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# %% [markdown] id="XPAdQkNTbnBF"
# ### Data Preview

# %% colab={"base_uri": "https://localhost:8080/"} id="YZIKcnCiYf-D" outputId="f051dc04-f214-418c-c97a-22a54fcdf7c7"
for text, label in ds_train.take(5):
  print('Text')
  print(text)
  print('Label')
  print(label)

# %% colab={"base_uri": "https://localhost:8080/", "height": 948} id="DmrzbcNs7E_a" outputId="de4e54c7-9ac3-4667-d15b-763052bd8584"
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

# %% id="_0PozHTddO9m"

# %% id="tbZFwexn7E_a"

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

# %% id="195d6bac"

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
