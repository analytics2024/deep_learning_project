import streamlit as st
import keras
from keras_preprocessing.text import tokenizer_from_json
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
import tensorflow as tf
import pickle

# Function for feature extraction
def feature_extraction(image):
    model = tf.keras.applications.vgg16.VGG16()
    model = keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    
    # Convert image to RGB
    image = image.convert('RGB')
    image = image.resize((224, 224))  # Resize image to match VGG-16 input size
    arr = keras.preprocessing.image.img_to_array(image, dtype=np.float32)
    arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
    arr = keras.applications.vgg16.preprocess_input(arr)

    feature = model.predict(arr, verbose=0)
    return feature

# Function for generating caption
def generate_caption(model, tokenizer, max_length, vocab_size, feature):
    caption = "<startseq>"
    while True:
        encoded = tokenizer.texts_to_sequences([caption])[0]
        padded = keras.preprocessing.sequence.pad_sequences([encoded], maxlen=max_length, padding='pre')[0]
        padded = padded.reshape((1, max_length))

        pred_Y = model.predict([feature, padded])[0, -1, :]
        next_word = tokenizer.index_word[pred_Y.argmax()]

        caption = caption + ' ' + next_word

        if next_word == '<endseq>' or len(caption.split()) >= max_length:
            break

    caption = caption.replace('<startseq> ', '')
    caption = caption.replace(' <endseq>', '')
    return caption

# Load tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer_json = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_json)

# Load model
model = keras.models.load_model("VGG16_LSTM_model.h5")
vocab_size = tokenizer.num_words
max_length = 37


# Streamlit UI
st.title('Image Captioning App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Generate caption
    feature = feature_extraction(image)
    caption = generate_caption(model, tokenizer, max_length, vocab_size, feature)
    st.write("Predicted Caption:")
    st.write(caption)