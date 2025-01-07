import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("Multiquery Retrievel Chatbot")

llm = Ollama(model='gpt-3.5-turbo')

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
Wiki_wrapper = WikipediaAPIWrapper(top_k_results=1 , doc_content_chars_max=1000)
wiki_query= WikipediaQueryRun(api_wrapper=Wiki_wrapper)


from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2 , doc_content_chars_max=1000)
arxiv_query = ArxivQueryRun(api_wrapper=arxiv_wrapper)

from langchain_community.document_loaders import WebBaseLoader
data_loader = WebBaseLoader("https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html")
web_data = data_loader.load()

split_data = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap = 200)
chunks = split_data.split_documents(web_data)

embedd = OllamaEmbeddings()



vectorstore = FAISS.from_documents(documents=chunks , embedding=embedd)
from langchain.retrievers import MultiQueryRetriever
Multi_query = MultiQueryRetriever.from_llm(
    retriever = vectorstore.as_retriever() , llm = llm
)

import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

from typing import List
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
prompts = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

class LineList(BaseModel):
    Lines : list[str] = Field(description="Lines of text")

class LineListOutputParser(PydanticOutputParser):
    def __init__(self) ->None:
        super().__init__(LineList)
    
    def parse(self, text: str) ->LineList:
        line  = text.strip().split("\n")
        return LineList(Lines=line)

outputparse = LineListOutputParser()

llm_chain = LLMChain(llm=llm , prompt=prompts , output_parser=outputparse)


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
context_q_prompt_system ="""Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
context_q_prompt = ChatPromptTemplate.from_messages(
    [
        ('system' ,context_q_prompt_system),
        MessagesPlaceholder('Chat_history'),
        ('human', "{input}"),
    ]
)
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
history_aware_history = create_history_aware_retriever(llm,Multi_query,context_q_prompt)



qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\ """
qa_prompt = ChatPromptTemplate.from_messages(
     ('system' ,qa_system_prompt),
        MessagesPlaceholder('Chat_history'),
        ('human', "{input}"),
)
from langchain.chains.combine_documents import create_stuff_documents_chain
stuff_documents = create_stuff_documents_chain(llm , qa_prompt)
rag_chain = create_retrieval_chain(history_aware_history,stuff_documents)

chat_history = []
def Answer_Multiquery(question : str, chat_history : list[str]):
    query_versaion = llm_chain.run(question =question)
    print(f"Multiple quer : {query_versaion}")

    answer = rag_chain.run(input =question , chat_history=chat_history)
    return answer

st.text_input("enter your Query")
if input:
    st.write(Answer_Multiquery(input , chat_history))
