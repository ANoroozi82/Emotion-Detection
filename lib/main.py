import streamlit
import lib.model_operation as modelOperation
import lib.database_operation as dbOps
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
    streamlit.title("Emotion Classifier App")
    streamlit.subheader("Emotion Detection in Text")

    dbOps.createEmotionclfTable()

    with streamlit.form(key="text_form"):
        rawText = streamlit.text_area("Type Here:"),
        submit_button = streamlit.form_submit_button(label='Submit')
    
    if submit_button:
        column = streamlit.columns(spec=1)

        prediction = modelOperation.nowEmotion(rawText)
        probability = modelOperation.futureEmotion(rawText)

        with column[0]:
            streamlit.success("Original Text")
            streamlit.write(rawText)

            streamlit.success("Prediction")
            emoji_icon = emoji[prediction]
            streamlit.write("{}:{}".format(prediction, emoji_icon))
            streamlit.write("Confidence:{}".format(np.max(probability)))

            streamlit.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=modelOperation.model.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]
            
            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            streamlit.altair_chart(fig, use_container_width=True)
