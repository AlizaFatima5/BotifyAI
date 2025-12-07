import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

DB_FAISS_PATH = "vectorstore/db_faiss"

# Load vectorstore
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

# Custom Prompt
def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

def main():
    st.title("Ask ChatGpt!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    prompt = st.chat_input("Ask something...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use ONLY the following context to answer the question.
        If the answer is not found in the context, reply with "I don't know".

        Context:
        {context}

        Question:
        {question}

        Reply clearly and directly.
        """

        try:
            vectorstore = get_vectorstore()

            if vectorstore is None:
                st.error("Failed to load vector store")
                return

            # Your actual Groq API key
            GROQ_API_KEY = "API KEY"

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                    temperature=0.0,
                    groq_api_key=GROQ_API_KEY,
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
                },
                input_key="question"
            )

            # Ask model
            response = qa_chain.invoke({"question": prompt})
            result = response["result"]
            # source_docs = response["source_documents"]

            # output = f"{result}\n\nSource Docs:\n{source_docs}"
            output=result
            st.chat_message("assistant").markdown(output)
            st.session_state.messages.append(
                {"role": "assistant", "content": output}
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()




