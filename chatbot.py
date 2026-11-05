
import streamlit as st
import chromadb
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq


# Center Logo
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image(
        "https://www.indala.ac.in/logo-white-bg.png",
        width=160
    )

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Indala College Admission Assistant",
    page_icon="🎓",
    layout="centered",  
    
)

# ---------------------------------
# CUSTOM CSS
# ---------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #ffffff;
}

.main-title {
    font-size: 34px;
    font-weight: bold;
    color: #17365D;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #555555;
    font-size: 16px;
    margin-bottom: 25px;
}

.stChatMessage {
    border-radius: 12px;
}

[data-testid="stChatInput"] {
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------
# LOAD API KEY
# ---------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


# ---------------------------------
# LOAD MODELS AND DATABASE
# ---------------------------------

@st.cache_resource
def load_resources():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    db_client = chromadb.PersistentClient(
        path="chroma_db"
    )

    collection = db_client.get_collection(
        name="college_admission"
    )

    return embedding_model, collection


try:
    model, collection = load_resources()

except Exception as e:
    st.error(f"Database loading error: {e}")
    st.stop()


# ---------------------------------
# GROQ CLIENT
# ---------------------------------

if not api_key:
    st.error("Groq API Key not found. Check your .env file.")
    st.stop()

groq_client = Groq(api_key=api_key)


# ---------------------------------
# HEADER
# ---------------------------------

st.markdown(
    '<div class="main-title">🎓 Indala College Admission Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Welcome! Ask anything about Indala College admissions.</div>',
    unsafe_allow_html=True
)


# ---------------------------------
# SESSION STATE
# ---------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------
# CLEAR CHAT
# ---------------------------------

with st.sidebar:

    st.title("🎓 Indala College")

    st.write("Admission Assistant")

    st.divider()

    st.write("Ask questions about:")

    st.write("📚 Engineering Courses")
    st.write("📝 Eligibility Criteria")
    st.write("📄 Required Documents")
    st.write("💰 Fee Structure")
    st.write("🎯 Admission Process")

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# ---------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------
# USER INPUT
# ---------------------------------

question = st.chat_input(
    "Ask anything about Indala College admissions..."
)


# ---------------------------------
# GENERATE ANSWER
# ---------------------------------

if question:

    # Display user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate AI response

    with st.chat_message("assistant"):

        with st.spinner("Finding information..."):

            try:

                # Create question embedding

                query_embedding = model.encode(
                    [question]
                ).tolist()

                # Search ChromaDB

                result_count = min(3, collection.count())

                if result_count == 0:
                    raise ValueError(
                        "College knowledge base is empty. Please add documents first."
                    )

                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=result_count
                )

                documents = results["documents"][0]

                context = "\n\n".join(documents)

                # Prepare system prompt

                system_prompt = f"""
You are Indala College of Engineering's official
Admission Assistant.

Your job is to answer student and parent questions
about college admissions.

Follow these rules:

1. Answer using only the supplied college information.
2. Give simple, clear and helpful answers.
3. Use bullet points wherever appropriate.
4. Answer in the language used by the student.
5. If the information is not available, clearly say
   that it is not available in the college knowledge base.
6. Never invent course names, fees, eligibility criteria,
   contact details or admission dates.
7. Be polite and professional.

COLLEGE INFORMATION:

{context}
"""

                # Include recent conversation history

                chat_history = []

                for msg in st.session_state.messages[-10:]:

                    chat_history.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

                # Generate response using Groq

                response = groq_client.chat.completions.create(

                    model="openai/gpt-oss-120b",

                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        *chat_history
                    ],

                    temperature=0.2
                )

                answer = response.choices[0].message.content

                # Display answer

                st.markdown(answer)

                # Save AI response

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(f"Error generating answer: {e}")
