from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent


loader = PyPDFLoader("./data/NCCIA_Knowledge_Base.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # smaller chunks = tighter, more precise matches
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "],  # keep structured entries together
)

splitted_Docs = text_splitter.split_documents(docs)


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)


vector_Store = InMemoryVectorStore.from_documents(

    documents=splitted_Docs,
    embedding= embeddings
)

#create Agent for tool calling = #tools , # llm  , prompt

@tool
def retriever_tool(query:str):
    """
        This tool can help you to fetch the relevent data of the pdf documents and these pdf documents 
        have details related to cyber crime nccia
    """

    docs = vector_Store.similarity_search(query=query , k= 8)
    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"   # += appends
    return context

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=2048,           
    reasoning_effort="low",    
)
system_prompt = """
You are a Cyber Crime and NCCIA (National Cyber Crime Investigation Agency) expert assistant.

Your job is to answer user questions strictly using information retrieved from the knowledge base via the 'retriever_tool'.

RULES:
1. For ANY question about laws, sections, offices, procedures, penalties, or cyber crime topics, you MUST call 'retriever_tool' first — never answer from your own general knowledge.
2. If the user's question has multiple parts, call the tool separately for EACH distinct part, so you retrieve relevant context for all of them.
3. Base your answer ONLY on the retrieved context. Do not add outside knowledge, assumptions, or invented details.
4. If the retrieved context does not contain enough information to answer, respond exactly with: "I don't have enough information in the knowledge base to answer that."
5. Quote or closely paraphrase relevant legal text (section numbers, penalties, definitions) precisely — do not summarize away specific numbers, section references, or legal terms.
6. Keep answers clear and structured. Use bullet points or numbered lists when answering about multiple sections, offices, or steps.
7. Never fabricate section numbers, office locations, or legal citations that are not explicitly present in the retrieved context.

Always ground your response in retrieved context before answering.
"""


agent = create_agent(
    model= llm,
    tools= [retriever_tool],
    system_prompt= system_prompt
)

query = "someone has created a fake account on instagram using my pictures. what should i do and also NCCIA has blocked my account what should i do"

resp = agent.invoke({
    "messages": [{
        "role": "user",
        "content": query
    }]
})

result = resp["messages"][-1].content

print(result)