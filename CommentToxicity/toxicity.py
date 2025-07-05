# toxicity.py

import os
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding, TextVectorization
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
import gradio as gr

# Step 1: Load Data
data_path = os.path.join('jigsaw-toxic-comment-classification-challenge', 'train.csv')
df = pd.read_csv('C:\\Users\\Sowmi\\OneDrive\\Desktop\\toxic comment detector\\CommentToxicity\\jigsaw-toxic-comment-classification-challenge\\train.csv\\train.csv')

print(df.head())

# Step 2: Vectorization
X = df['comment_text']
y = df[df.columns[2:]].values

MAX_FEATURES = 200000
vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES,
    output_sequence_length=1800,
    output_mode='int'
)
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

# Step 3: Create Dataset
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache().shuffle(160000).batch(16).prefetch(8)

train = dataset.take(int(len(dataset) * .7))
val = dataset.skip(int(len(dataset) * .7)).take(int(len(dataset) * .2))
test = dataset.skip(int(len(dataset) * .9)).take(int(len(dataset) * .1))

# Step 4: Build Model
model = Sequential()
model.add(Embedding(MAX_FEATURES + 1, 32))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='BinaryCrossentropy', optimizer='Adam')
model.summary()

# Step 5: Train Model
history = model.fit(train, epochs=1, validation_data=val)

# Step 6: Plot Results
plt.figure(figsize=(8, 5))
pd.DataFrame(history.history).plot()
plt.show()

# Step 7: Evaluate
pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

for batch in test.as_numpy_iterator():
    X_true, y_true = batch
    yhat = model.predict(X_true)
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)

print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')

# Step 8: Save model
model.save('toxicity.h5')

# Step 9: Load model (if running again)
model = tf.keras.models.load_model('toxicity.h5')

# Step 10: Gradio Interface
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    labels = df.columns[2:]
    output = ''
    for idx, label in enumerate(labels):
        output += f'{label}: {"✅" if results[0][idx] > 0.5 else "❌"}\n'
    return output

interface = gr.Interface(
    fn=score_comment,
    inputs=gr.Textbox(lines=2, placeholder='Enter a comment...', label="Input Comment"),
    outputs=gr.Text(label="Toxicity Prediction"),
    title="Toxic Comment Classifier"
)

interface.launch(share=True)
