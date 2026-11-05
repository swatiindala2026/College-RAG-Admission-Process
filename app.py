import streamlit as st

st.set_page_config(
    page_title="College Admission Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Indala College of Engineering")
st.subheader("AI-Based College Admission Assistant")

st.write(
    "Welcome! Ask your questions about college admissions, "
    "courses, eligibility, fees and admission procedures."
)

st.divider()

st.header("Upload College Documents")

uploaded_files = st.file_uploader(
    "Upload college prospectus or admission PDF",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success("Documents uploaded successfully!")

    for file in uploaded_files:
        st.write("📄", file.name)

st.divider()

st.header("Ask Your Admission Question")

question = st.text_input(
    "Enter your question here:"
)

if st.button("Submit Question"):
    if question.strip():
        st.info("Your question: " + question)
        st.success(
            "Your application interface is working! "
            "We will connect the RAG system next."
        )
    else:
        st.warning("Please enter a question.")