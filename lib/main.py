import streamlit as sl
import joblib as jl

model = jl.load(open('./models/emotion_classifier_pipe_lr.pkl'))