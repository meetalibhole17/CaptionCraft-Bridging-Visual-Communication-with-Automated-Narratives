from flask import Flask, render_template, request
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import add
from keras.layers import Dense, Flatten, Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector, Concatenate
from keras.models import Sequential, Model
#from keras.utils import np_utils
import numpy as np
from keras.preprocessing import image, sequence
import cv2
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import load_model

from keras.applications import ResNet50
resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling='avg')

print("="*50)
print("resnet loaded")
vocab = np.load('vocab2_100.npy', allow_pickle = True)
vocab = vocab.item()
inv_vocab = {v:k for k,v in vocab.items()}

embedding_size = 128
max_len = 40
vocab_size = len(vocab)+1
image_model = Sequential()
image_model.add(Dense (embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))

language_model = Sequential()
language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len)) 
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense (embedding_size)))

conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True) (conca)
x = LSTM(512, return_sequences=False) (x)
x = Dense (vocab_size)(x)
out = Activation('softmax') (x)
model =Model(inputs=[image_model.input, language_model.input], outputs = out)
# model.load_weights("../input/model_weights.h5")
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.load_weights('mine_model_weights2_100.h5')

print("="*50)
print("model loaded")
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT']=1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['POST'])
def after():
    global model, vocab, inv_vocab, resnet
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
        sampled_word= inv_vocab[sampled_index]
        
        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word
        text_in.append(sampled_word)




    return render_template('predict.html', final = final)
if __name__ == "__main__":
    app.run(debug=True)
