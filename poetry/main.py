import random
import numpy as np
import tensorflow as tf
import ssl
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Activation # type: ignore
from tensorflow.keras.optimizers import RMSprop # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

ssl._create_default_https_context = ssl._create_unverified_context # Bypass SSL certificate verification for downloading files

# opens the file and puts it into 'filepath'
# then reads the file and decodes it to lower case
#Change the file path to something else if I want to to generate text from a different source
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:1000000]  # Truncate to a smaller size for faster training

characters = sorted(set(text)) #set of all the characters that appear in the text (sorted)

char_to_index = dict((c, i) for i, c in enumerate(characters))  #dictionary that maps characters to indices {a:1, b:2, ...}
index_to_char = dict((i, c) for i, c in enumerate(characters)) #dictionary that maps indices to characters {1:a, 2:b, ...}

SEQ_LENGTH = 100  # Length of each input sequence
STEP_SIZE = 3  # Step size for creating sequences


sentences = []
next_chars = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])  # Extract sequences of length SEQ_LENGTH
    next_chars.append(text[i + SEQ_LENGTH])  # Extract the character following the sequence


# we now need to convert the sentences and next characters into numerical format
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)  
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

for i , sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1  # One-hot encoding of the characters
    y[i, char_to_index[next_chars[i]]] = 1  # One-hot encoding of the next character


'''
#This section trains the model. It is commented out to avoid re-training every time the script runs.

model = Sequential()
model.add(LSTM(512, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))  # Softmax activation for multi-class classification

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001))

early_stop = EarlyStopping(monitor='loss', patience=3)
model.fit(x, y, batch_size=256, epochs=15, callbacks=[early_stop])  # Train the model

model.save("textgenerator.keras")
'''



model = tf.keras.models.load_model('textgenerator.keras')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature  # Apply temperature to predictions
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)  # Normalize to sum to 1
    probas = np.random.multinomial(1, preds, 1)  # Sample from the distribution
    return np.argmax(probas)  # Return the index of the sampled character

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence

    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char, in enumerate(sentence):
            x[0, t, char_to_index[char]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char  # Update the sentence by removing the first character and adding the new one
    return generated


# Generate text with a specified length and temperature

print('0.5 Temp')
print(generate_text(1000, 0.5))  

