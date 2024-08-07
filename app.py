import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import matplotlib.pyplot as plt
temp = pathlib.PosixPath
pathlib.PosixPath = temp
# Create title
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
  
    </style>
   
    <h1 class="centered-title">Classification of Transports</h1>
    """, 
    unsafe_allow_html=True
)


model=load_learner("transport_model_2.pkl")
file=st.file_uploader("Upload your image",['png',"jpeg","jpg","gif","svg"])
if file:
    img=PILImage.create(file)
    prediction,prediction_id,probability=model.predict(img)
    st.image(file)
    st.success(f'Prediction result: {prediction}')
    prob=f'Probability: {probability[prediction_id]*100:.1f}%'
    st.info(prob)
    fig=px.bar(x=probability*100,y=model.dls.vocab)
    st.plotly_chart(fig)
