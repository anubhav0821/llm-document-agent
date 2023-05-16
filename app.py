import os
from langchain.llms import OpenAI
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import ( 
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)



os.environ['OPENAI_API_KEY'] = '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'

llm = OpenAI(temperature=0.1)

loader = PyPDFLoader('annualreport.pdf')
pages = loader.load_and_split()
store = Chroma.from_documents(pages, collection_name='annualreport')

vectorestore_info = VectorStoreInfo(
    name='annual_report',
    description='a banking report as a pdf',
    vectorstore=store
)

toolkit = VectorStoreToolkit(vectorstore_info=vectorestore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

st.title('🦜🔗 GPT Investment Banker')
prompt = st.text_input("Input your prompt here")



if prompt:
    response = agent_executor.run(prompt)
    st.write(response)

    with st.expander('Document Similarity Search'):
        search = store.similarity_search_with_score(prompt)
        st.write(search[0][0].page_content)

