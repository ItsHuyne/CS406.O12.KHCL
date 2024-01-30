import numpy as np
import streamlit as st
from utils import *
from keras.models import load_model
import os

@st.cache_resource()
def load_remover(path):
    return load_model(path, compile=False)

st.header('Remove Background Demo')
model = load_remover('U2Net_AutoMattingData-0.6424-weights-10.h5')

file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
if file is not None:
    # image = Image.open(file)
    # image = image.resize((512, 512))
    # st.image(image, width=512, use_column_width=True)
    if st.button('Predict'):
        col1, col2 = st.columns(2)
        process_and_save_mask(file)
        with col1:
            image = Image.open(file)
            image = image.resize((512, 512))
            st.image(image)
        with col2:
            mask = get_mask(model, file)
            st.image(mask, clamp=True)

        origin_image = Image.open(file)
        matting_image = map(origin_image, mask)
        st.image(matting_image,use_column_width=True)

else:
    if os.path.exists('img.png'):
        os.remove('img.png')