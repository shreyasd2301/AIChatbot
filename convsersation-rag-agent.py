import streamlit as st
import openai
from dotenv import load_dotenv

from utils import get_pdf_text
from utils import get_text_chunks, get_pdf_doc
from utils import get_vectorstore
from utils import get_conversation_chain
from utils import get_search_tool, get_rag_tool, get_google_tool
from utils import load_zero_shot_agent, load_conversational_agent


from htmlTemplates import css


def main():

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs, API KEY here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                st.text("Extracting Texts Done ✅")

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.text("Processing Chunks Done ✅")
                
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.text("Memory loaded Done ✅")
                
                # create conversation chain
                
                search_tool = get_search_tool()
                rag_tool = get_rag_tool(vectorstore)
                tools = [search_tool, rag_tool]
                st.session_state.agent = load_conversational_agent(tools)
                st.session_state.agent.agent.llm_chain.prompt.template = '''
                    Answer the following questions as best you can covering all the aspects in frame wise manner\
                    you get the answer.
                    You have access to the following tools:

                    Document Store: Use it to lookup information from document store. \
                                    Always used as first tool
                    Search: Use this to lookup information from search engine. \
                            Use it only after you have tried using the document store tool.

                    Use the following format:

                    Question: the input question you must answer
                    Thought: you should always think about what to do
                    Action: the action to take, should be one of [Document Store, Search]. \
                            Always look first in Document Store.
                    Action Input: the input to the action
                    Observation: the result of the action
                    ... (this Thought/Action/Action Input/Observation can repeat N times)
                    ... (do not perform same Thought/Action/Action Input repeatedly)
                    Thought: I now know the final answer
                    Final Answer: the final answer to the original input question

                    Begin!

                    Question: {input}
                    Thought:{agent_scratchpad}
                    '''
                st.text("Agent loaded Done ✅")
                st.write(st.session_state.agent.agent.llm_chain.prompt)

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Start chatting with your docs.")

    if not user_question:
        st.info("Please upload docs and openai key, and then proceed to chat")
    
    if user_question: 
        result = st.session_state.agent.invoke(user_question)
        st.header("Answer")
        # print(result)
        st.write(result)



if __name__ == '__main__':
    main()