import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

groq_api_key = os.environ.get("GROQ_API_KEY")

chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192",max_tokens=1024)
prompt_template ="""### [INST]
Instruction:
you are a proficient human customer support agent named Alex for a company specializing in USTPO trademarks on a call with a customer. You possess comprehensive knowledge of trademark-related matters and be adept at addressing various customer inquiries and concerns promptly and accurately. You should provide assistance throughout the chat, offer guidance on trademark search strategies, explain legal terminology, and troubleshoot common issues faced by customers. You should maintain a friendly and professional tone, foster positive customer interactions. ((Do not provide false information))
Give all your answers such that you are on a call with the customer

Below you can find reference as to some example customer questions and their ideal answer, refer to this as context for your reply:
{context}

[/INST]
"""
with open('queries.txt','r') as f:
  data=f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=2
)

# Split the text into smaller chunks
texts = text_splitter.create_documents([data])
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# Converting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(texts, embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chat_history = ChatMessageHistory()
prompt = ChatPromptTemplate.from_messages(
    [
        prompt_template,
        MessagesPlaceholder(variable_name="messages"),
    ]
)
document_chain = create_stuff_documents_chain(chat, prompt)

@app.post("/start-task/")
async def responsee(query:str=Form(...)):
  chat_history.add_user_message(query)
  docs = retriever.invoke(query)
  response=document_chain.invoke(
      {
          "messages": chat_history.messages,
          "context": docs,
      }
  )
  chat_history.add_ai_message(response)
  return response
