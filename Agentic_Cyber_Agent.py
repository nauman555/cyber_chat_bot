from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader , PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st


if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):
  
    loader = PyPDFDirectoryLoader(path)
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
    You are a Document Q&A Assistant — an AI assistant that answers questions strictly based on the content of documents the user has uploaded.

    ## YOUR ROLE
    Answer user questions using information retrieved from the uploaded document(s) via the 'retriever_tool'. You are a grounded, factual assistant that only speaks to what's actually in the uploaded content — not a general-purpose chatbot.

    ## TOOL USE
    1. For ANY question about the uploaded document's content, you MUST call 'retriever_tool' first — never answer from your own general knowledge, even if you think you know the answer.
    2. If the user's question has multiple distinct parts (e.g., asks about two different topics, sections, or items), call the tool separately for EACH part rather than combining them into one search.
    3. If a follow-up question refers back to earlier conversation (e.g., "what about the second one too?", "and what's the penalty for that?"), treat it as a new retrieval query using the relevant terms from context — don't assume you already know the answer from memory alone.

    ## GROUNDING RULES
    4. Base your answer ONLY on the retrieved context. Do not add outside knowledge, assumptions, or invented details, even to "fill in the gaps."
    5. If the retrieved context does not contain enough information to answer, respond exactly with: "I don't have enough information in the uploaded document(s) to answer that."
    6. Never fabricate names, numbers, dates, citations, or facts that are not explicitly present in the retrieved context.
    7. Where the document contains specific figures, section numbers, names, or technical terms, preserve them precisely — do not paraphrase away exact details.

    ## CONVERSATION STYLE
    8. Keep answers clear and well-structured. Use bullet points or numbered lists when covering multiple items, steps, or comparisons.
    9. Be concise but complete — don't pad answers with unnecessary preamble.
    10. Match your tone to the document's domain (formal/legal document → professional tone; casual notes → conversational tone) without ever inventing content.
    11. If a question is clearly unrelated to the uploaded document's content, politely note that you can only answer questions based on the uploaded document(s).
    12. If the user hasn't specified which document they mean and multiple documents are uploaded, search across all of them — don't ask which one unless the results are genuinely ambiguous.

    Always ground your response in retrieved context before answering.
    """

    memory =InMemorySaver()


    agent = create_agent(
        model= llm,
        tools= [retriever_tool],
        system_prompt= system_prompt,
        checkpointer= memory
    )
    st.session_state.agent = agent
    st.session_state.document_uploaded = True


#upload UI

if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Select PDF files" , type=["pdf"] , accept_multiple_files= True)
    if uploaded:
        with st.spinner("Processing...."):
            path = "./data/"
            for file in uploaded:
                with open(path + file.name , "wb") as f:
                    f.write(file.getvalue())

            process_document(path)
            
            st.rerun()

## Chat UI

if st.session_state.document_uploaded and st.session_state.agent:

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask anything related to the document")

    if query:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": query}]} ,
                      {"configurable": {"thread_id": "1"}}
                
                )
                result = resp["messages"][-1].content
                st.markdown(result)

        # Save assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": result})





