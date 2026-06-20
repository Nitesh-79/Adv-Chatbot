import streamlit as st
import tempfile
from gtts import gTTS
from llm import get_llm
from streamlit_mic_recorder import mic_recorder
from vectorstore import create_vectorstore
from rag import get_rag_chain

# Page Config

st.set_page_config(page_title="AI Research Assistant")

st.title("AI Research Assistant")

# Sidebar

provider = st.sidebar.selectbox(
    "Choose Model",
    ["Groq", "Gemini"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# Load LLM

llm = get_llm(provider)

# Voice Recorder (UI only for now)

audio = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    key="recorder"
)

# Session State

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Create Vector DB only once

if uploaded_file and st.session_state.db is None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.read())

        st.session_state.db = create_vectorstore(
            tmp.name
        )

        st.session_state.rag_chain = get_rag_chain(llm)

# Display Chat History

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input

question = st.chat_input(
    "Ask Anything..."
)

# Handle Question

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    full_response = ""

    with st.chat_message("assistant"):

        placeholder = st.empty()

        try:

            # RAG Mode

            if st.session_state.db is not None:

                retriever = (
                    st.session_state.db
                    .as_retriever()
                )

                docs = retriever.invoke(
                    question
                )

                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )

                full_response = (
                    st.session_state.rag_chain
                    .invoke(
                        {
                            "context": context,
                            "question": question
                        }
                    )
                )

                placeholder.markdown(
                    full_response
                )

            # Normal Chat Mode

            else:

                for chunk in llm.stream(
                    question
                ):

                    if chunk.content:

                        full_response += (
                            chunk.content
                        )

                        placeholder.markdown(
                            full_response
                        )

        except Exception as e:

            full_response = str(e)

            placeholder.error(
                full_response
            )

    # Save assistant message

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )

    # Text To Speech

    try:

        tts = gTTS(
            text=full_response,
            lang="en"
        )

        tts.save("response.mp3")

        st.audio("response.mp3")

    except Exception as e:

        st.warning(
            f"TTS Error: {e}"
        )