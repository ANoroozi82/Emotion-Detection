import streamlit as st
import lib.model_operation as modelOperation
import numpy as np
import altair as alt
import pandas as pd

emoji = {
    "anger": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😁",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😟",
    "sadness": "😔",
    "shame": "😅",
    "surprise": "😲"
}

def main():
    st.title("Emotion Classifier App")
    st.subheader("Emotion Detection in Text")

    with st.form(key="text_form"):
        rawText = st.text_area("Type Here:")
        submit_button = st.form_submit_button(label='Submit')
    
    if submit_button:
        prediction = modelOperation.nowEmotion(rawText)
        probability = modelOperation.futureEmotion(rawText)

        st.success("Original Text")
        st.write(rawText)

        st.success("Prediction")
        emoji_icon = emoji[prediction]
        st.write(f"{prediction}: {emoji_icon}")
        st.write(f"Confidence: {np.max(probability):.2f}")

        st.success("Prediction Probability")
        proba_df = pd.DataFrame([probability], columns=modelOperation.model.classes_)
        proba_df_clean = proba_df.T.reset_index()
        proba_df_clean.columns = ["emotions", "probability"]

        fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
        st.altair_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
