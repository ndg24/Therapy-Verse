import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import spacy
import tensorflow as tf
import tensorflow_datasets as tfds
tf.random.set_seed(1)
import os

pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
pd.set_option("max_colwidth", None)

chat_data = pd.read_csv("https://raw.githubusercontent.com/nbertagnolli/counsel-chat/master/data/20200325_counsel_chat.csv")

X = chat_data["questionText"]
y = chat_data["answerText"]

def preprocess_text(phrase):
  phrase = re.sub(r"\xa0", "", phrase)
  phrase = re.sub(r"\n", "", phrase)
  phrase = re.sub("[.]{1,}", ".", phrase)
  phrase = re.sub("[ ]{1,}", " ", phrase)
  return phrase

X = X.apply(preprocess_text)
y = y.apply(preprocess_text)

question_answer_pairs = []

for (question, answer) in zip(X, y):
  question = preprocess_text(question)
  answer = preprocess_text(answer)
  question_arr = question.split(".")
  answer_arr = answer.split(".")
  max_sentences = min(len(question_arr), len(answer_arr))

  for i in range(max_sentences):
    q_a_pair = []
    q_a_pair.append(question_arr[i])
    q_a_pair.append(answer_arr[i])
    question_answer_pairs.append(q_a_pair)

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    [arr[0] + arr[1] for arr in question_answer_pairs], target_vocab_size=2**13)

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2

MAX_LENGTH = 100

def tokenize_and_filter(inputs, outputs):
  tokenized_inputs, tokenized_outputs = [], []

  for (sentence1, sentence2) in zip(inputs, outputs):
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
    
    if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
      tokenized_inputs.append(sentence1)
      tokenized_outputs.append(sentence2)

  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding="post")
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding="post")
    
  return tokenized_inputs, tokenized_outputs

questions, answers = tokenize_and_filter([arr[0] for arr in question_answer_pairs], 
                                         [arr[1] for arr in question_answer_pairs])

BATCH_SIZE = 64
BUFFER_SIZE = 20000

dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

EPOCHS = 20
model.fit(dataset, epochs=EPOCHS)

model.load_weights("chatbot_transformer_v4.h5")

def evaluate(sentence):
  sentence = preprocess_text(sentence)
  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)


def predict(sentence):
  prediction = evaluate(sentence)
  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  print('Input: {}'.format(sentence))
  print('Output: {}'.format(predicted_sentence))

  return predicted_sentence

def simpleIntentChatbot(phrase):
  search_terms = ["find", "search", "check", "seek", "look", "figure"]
  phrase = phrase.lower()
  phrase_list = phrase.split(" ")

  is_asking_for_something = False

  for word in phrase_list:
    for term in search_terms:
      if term == word:
        is_asking_for_something = True

  if is_asking_for_something:
    return "Ok! I will find that information for you!"
  else:
    return "I don't know how to respond to that!"

phrase = "Can you look up that for me?"
print(simpleIntentChatbot(phrase))
