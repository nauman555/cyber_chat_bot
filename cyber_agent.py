from dotenv import load_dotenv
import os

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


#load text

loader = PyPDFLoader("./data/NCCIA_Knowledge_Base.pdf")

documents =  loader.load()

#split text

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # smaller chunks = tighter, more precise matches
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "],  # keep structured entries together
)

splitted_data = text_splitter.split_documents(documents)

#embedding 

# embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)

#store data in Vector store

vector_store =  Chroma.from_documents(

    documents= splitted_data,
    embedding = embeddings
)




llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=2048,           
    reasoning_effort="low",    
)



def get_content(query:str):
    searched_data = vector_store.similarity_search(query=query , k = 8)
    context = ""

    for data in searched_data:
        context += data.page_content + "\n"

    return {
        "context" : context,
        "query" : query
    }

prompt = PromptTemplate.from_template("""

    YOU ARE A CYBER CRIME / NCCIA / LAWS EXPERT . You need to answer the query / questions asked by the user . the answers should be based from the context data.
    if you dont know the answer, just respond , i dont know the answer.
    Context: {context} 
    Question: {query}

""")
  
rag_chain = get_content | prompt | llm

response = rag_chain.invoke("where is cyber crime office gilgit")

print(response.content)