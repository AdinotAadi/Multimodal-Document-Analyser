import streamlit as st
from module import process_text

def format_score(score):
    score = max(0.0, min(1.0, score))
    return round(score * 100, 2)

def main():
    st.title("MultiModal Document Analyzer")
    text_input = st.text_area("Enter your text here:")

    if st.button("Analyze"):
        if text_input:
            sentiment, emotion, summary = process_text(text_input)
            with st.expander("Sentiment", expanded=True):
                st.markdown("**Label:** " + sentiment[0][0]['label'])
                st.markdown("**Confidence:** " + str(format_score(sentiment[0][0]['score'])) + "%")

            with st.expander("Emotion", expanded=False):
                st.markdown("**Label:** " + emotion[0]['label'])
                st.markdown("**Confidence:** " + str(format_score(emotion[0]['score'])) + "%")

            with st.expander("Summary", expanded=False):
                st.write(summary)
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
