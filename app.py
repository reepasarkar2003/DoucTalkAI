import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -----------------------------------
# PDF TEXT EXTRACTION
# -----------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text


# -----------------------------------
# TEXT CHUNKING
# -----------------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)


# -----------------------------------
# VECTOR STORE (FAISS)
# -----------------------------------
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_texts(text_chunks, embedding=embeddings)
    db.save_local("faiss_index")


# -----------------------------------
# LOAD LOCAL FLAN-T5 MODEL
# -----------------------------------
def load_llm():
    model_id = "google/flan-t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        temperature=0.3
    )

    return HuggingFacePipeline(pipeline=pipe)


# -----------------------------------
# QA CHAIN (NO API, NO ERRORS)
# -----------------------------------
def get_conversational_chain():

    prompt_template = """
    You are a helpful assistant. Answer ONLY from the given context.

    If the answer is not present, reply:
    "answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    llm = load_llm()

    chain = prompt | llm | StrOutputParser()
    return chain


# -----------------------------------
# HANDLE USER INPUT
# -----------------------------------
def user_input(user_question):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(user_question)
    context = "\n\n".join([d.page_content for d in docs])

    chain = get_conversational_chain()

    response = chain.invoke({
        "context": context,
        "question": user_question
    })

    st.write("Reply:", response)


# -----------------------------------
# STREAMLIT APP UI
# -----------------------------------
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF with my localpdfapp 🤗📄")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload PDF Files and Click Submit & Process",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing…"):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success("Done!")
            else:
                st.warning("Please upload at least one PDF!")


if __name__ == "__main__":
    main()
