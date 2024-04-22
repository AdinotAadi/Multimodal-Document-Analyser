import streamlit as st
from module import process_text
from extractor.pdf2txt import extract_text_from_pdf
from extractor.img2txt import extract_text_from_image

def format_score(score):
    score = max(0.0, min(1.0, score))
    return round(score * 100, 2)

def main():
    st.title("Multi-format File Analyzer")

    # File upload functionality
    uploaded_files = st.file_uploader("Upload one or more files", type=["pdf", "txt", "jpg", "png"], accept_multiple_files=True)

    if uploaded_files:
        # Analyze button outside the loop
        if st.button("Analyze"):
            for uploaded_file in uploaded_files:
                st.write(f"### Analysis for {uploaded_file.name}")

                file_contents = uploaded_file.read()

                # Check file type and extract text accordingly
                if uploaded_file.type == "application/pdf":
                    text_input = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    text_input = file_contents.decode("utf-8")
                elif uploaded_file.type.startswith('image'):
                    text_input = extract_text_from_image(uploaded_file)
                else:
                    st.error("Unsupported file type. Please upload a PDF, TXT, JPEG, or PNG file.")
                    continue

                # Perform analysis
                sentiment, emotion, summary = process_text(text_input)
                with st.expander("Sentiment", expanded=True):
                    st.markdown("**Label:** " + sentiment['label'])
                    st.markdown("**Confidence:** " + str(format_score(sentiment['score'])) + "%")

                with st.expander("Emotion", expanded=False):
                    st.markdown("**Label:** " + emotion['label'])
                    st.markdown("**Confidence:** " + str(format_score(emotion['score'])) + "%")

                with st.expander("Summary", expanded=False):
                    st.write(summary)
    else:
        st.warning("Please upload a file or files.")

if __name__ == "__main__":
    main()
