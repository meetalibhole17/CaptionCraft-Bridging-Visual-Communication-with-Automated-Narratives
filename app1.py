from flask import Flask, render_template, request
from keras.layers import Input, Dense, LSTM, Embedding, RepeatVector, TimeDistributed, concatenate
from keras.models import Model
from keras.applications import ResNet50
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import cv2

# Load ResNet50 model
resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling='avg')

print("="*50)
print("ResNet50 loaded")

# Load vocabulary
vocab = np.load('vocab2_100.npy', allow_pickle=True)
vocab = vocab.item()
inv_vocab = {v: k for k, v in vocab.items()}

# Model parameters
embedding_size = 128
max_len = 40
vocab_size = len(vocab) + 1

# Define input layers
image_input = Input(shape=(2048,))
language_input = Input(shape=(max_len,))

# Image model
image_features = Dense(embedding_size, activation='relu')(image_input)
image_features = RepeatVector(max_len)(image_features)

# Language model
language_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size)(language_input)
language_lstm = LSTM(256, return_sequences=True)(language_embedding)
language_features = TimeDistributed(Dense(embedding_size))(language_lstm)

# Concatenate image and language features
concatenated = concatenate([image_features, language_features])

# LSTM layers
x = LSTM(128, return_sequences=True)(concatenated)
x = LSTM(512, return_sequences=False)(x)

# Output layer
output = Dense(vocab_size, activation='softmax')(x)

# Define the model
model = Model(inputs=[image_input, language_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# Load weights
model.load_weights('mine_model_weights2_100.h5')

print("="*50)
print("Model loaded")

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['POST'])
def after():
    # Check if 'file1' is in request.files
    if 'file1' not in request.files:
        return render_template('error.html', message='No file part')

    file = request.files['file1']

    # Check if the file has a filename (i.e., if it was selected in the form)
    if file.filename == '':
        return render_template('error.html', message='No selected file')

    # Save the file with a secure filename in the 'static' folder
    file.save('static/file.jpg')

    img = cv2.imread('static/file.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224,224,))
    img = np.reshape(img, (1,224,224,3))

    features = resnet.predict(img).reshape(1,2048)

    text_in = ['startofseq']
    final = ''

    print("="*50)
    print("GETTING CAPTIONS")

    count = 0
    while count < 20:
        count += 1
        encoded = []
        for i in text_in:
            encoded.append(vocab[i])
        
        padded = pad_sequences([encoded], padding='post', truncating='post', maxlen=max_len)
        sampled_index = np.argmax(model.predict([features, padded]))
        sampled_word = inv_vocab[sampled_index]
        
        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word
        text_in.append(sampled_word)

    return render_template('predict.html', final=final)

if __name__ == "__main__":
    app.run(debug=True)
