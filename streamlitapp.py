import streamlit as st
from FaqGenerator_v2 import AssignmentFAQGenerator
import tempfile, os

st.title("FAQ Generator")

uploaded_file = st.file_uploader("Upload assignment (.pdf or .txt)", type=["pdf", "txt"])
text_input = st.text_area("Or paste the assignment text here", height=200)

if st.button("Generate FAQs"):
    if uploaded_file:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        to_process = tmp_path
    else:
        to_process = text_input.strip()

    if not to_process:
        st.warning("Please provide input text or upload a file.")
    else:
        faq = AssignmentFAQGenerator()
        success = faq.process_document(to_process)
        if uploaded_file and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if not success:
            st.error("Failed to generate FAQs.")
        else:
            st.subheader("Generated FAQs")
            for i, pair in enumerate(faq.faq_pairs, 1):
                st.markdown(f"**Q{i}: {pair['question']}**")
                st.write(pair['answer'])
                st.write("---")
            st.subheader("Statistics")
            st.json(faq.get_statistics())
