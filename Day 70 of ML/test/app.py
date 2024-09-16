import streamlit as st
import numpy as np
import pickle
# import tensorflow as tf
# from tensorflow.keras.model import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

#load the LSTM model

model = load_model ('next_word_lstm.h5')

#open the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer  =pickle.load(handle)
    
#Function to predict the next word
# max_sequence_len =14
def predict_next_word(model,tokenizer ,text, max_sequence_len):
    token_list = tokenizer.text_to_sequences(text)[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]#Ensure the  sequence length match
    token_list = pad_sequences([token_list],maxlen = max_sequence_len-1, padding = 'pre')
    predicted  = model.predict(token_list,verbose = 0)
    predict_word_index = np.argmax(predicted,axis =1)
    for word,index in tokenizer.word_index.items():
        if index == predict_word_index:
            return word
    return None


#streamlit app

st.title("Next Word prediction with LSTM ")
input_text =st.text_input('Enter the sequenece of words',"To be or not to")
if st.button("predict Next Word"):
    
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f'Next word:{next_word}')