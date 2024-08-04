#!/usr/bin/env python
from typing import List, Union

from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langserve import add_routes
from langserve import RemoteRunnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser



# Web loader
web_loader = WebBaseLoader("https://www.promtior.ai")
web_doc = web_loader.load()
# PDF loader
pdf_loader = PyPDFLoader("docs/AI_Engineer.pdf")
pdf_doc = pdf_loader.load()

docs = web_doc + pdf_doc
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "promtior_search",
    "Search for information about Promtior. For any questions about Promtior, you must use this tool!",
)
search = TavilySearchResults()
tools = [retriever_tool, search]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks that answers questions about Promtior."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input} {agent_scratchpad}"),
])

llm = ChatOpenAI( model="gpt-3.5-turbo", temperature=0)
agent = create_openai_tools_agent(llm, tools,prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

class Input(BaseModel):
    input: str
    chat_history: List[HumanMessage | AIMessage | SystemMessage] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )

class Output(BaseModel):
    output: str

playground_compatible_agent_executor = (
    agent_executor | (lambda x: x["output"])
).with_types(input_type=Input, output_type=Output)

add_routes(
    app,
    playground_compatible_agent_executor,
    path="/promtior_chatbot",
    playground_type="chat"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
