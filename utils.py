from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import os
os.environ["OPENAI_API_KEY"] = 
os.environ["SERPAPI_API_KEY"] = 



def get_rag_chain(vectorstore):
    # llm = ChatOpenAI(openai_api_key = )
    llm = get_llama()
    rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    )
    return rag_pipeline

def get_conversation_chain(vectorstore):
    # llm = get_llama()
    llm = ChatOpenAI(openai_api_key = )
    # if you wanna use HF models, uncomment this and comment out ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain


import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


import streamlit as st
from htmlTemplates import bot_template, user_template


# print ai user interactions in streamlit 
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    
    # logger.info("All response ..... ")
    # logger.info(response)

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
