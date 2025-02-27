import streamlit as st
import os
import glob
import warnings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback

# Load environment variables
load_dotenv()

# HYPERPARAMETERS
n_search_kwargs = 3

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OPENAI_API_KEY is missing. Please set it in `.env` or Streamlit Secrets.")
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize session state for retrieved chunks
if "retrieved_chunks" not in st.session_state:
    st.session_state["retrieved_chunks"] = []

# Streamlit UI
st.title("🐯🔍🧞‍♀️ 고려대 의과대학 정보 지니")

# Define document storage folder
DOC_FOLDER = "temp_rec"
FAISS_INDEX_PATH = "faiss_index"

# Define supported file loaders
FILE_LOADERS = {
    "pdf": PyPDFLoader,
    "md": UnstructuredMarkdownLoader,
    "txt": TextLoader,
    "docx": UnstructuredWordDocumentLoader,
}

# Check if FAISS index exists
if os.path.exists(FAISS_INDEX_PATH):
    st.success("✅ 기존 데이터를 불러왔습니다. 바로 질문하세요!")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": n_search_kwargs})
else:
    st.warning("📂 문서를 처음 로딩 중입니다... 잠시만 기다려 주세요.")

    all_files = []
    for ext in FILE_LOADERS.keys():
        all_files.extend(glob.glob(os.path.join(DOC_FOLDER, f"*.{ext}")))

    if not all_files:
        st.error("❌ 지원되는 문서가 없습니다. `resources` 폴더에 파일을 추가해 주세요.")
    else:
        all_documents = []
        for file_path in all_files:
            ext = file_path.split(".")[-1].lower()
            if ext in FILE_LOADERS:
                loader = FILE_LOADERS[ext](file_path)
                documents = loader.load()

                # Store file source metadata for each document
                for doc in documents:
                    doc.metadata["source"] = os.path.basename(file_path)

                all_documents.extend(documents)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                       chunk_overlap=200,
                                                       separators=["\n# ", "\n## ", "\n### "])
        splits = text_splitter.split_documents(all_documents)

        # Embed and store in FAISS
        vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
        vectorstore.save_local(FAISS_INDEX_PATH)  
        retriever = vectorstore.as_retriever(search_kwargs={"k": n_search_kwargs})

        st.success("✅ 문서가 성공적으로 저장되었습니다! 질문하세요.")

# Custom RAG prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "질문을 할 학생은 고려대 의대생이고, 다음 질문에 대한 답변을 한국어로 정중하고 공식적인 톤으로 제공하세요. "
        """개인적인 의견이나 사실을 유추하는 등 추측하지 말고, 가능한 경우, 출처에서 직접 인용(\" \")하고, 관련 조항이 있으면 생략하지 말고 나열하세요.
        또한, 학점, 기간, 비용, 시간, 수업 번호, 웹페이지 링크, 조항 등 수치적 또는 구체적인 정보가 있으면 생략하지 말고 제공하세요. 
        질문에 대한 답변이 주어진 정보를 참고해도 지나치게 부정확하다고 판단되면 "질문을 더 구체적으로 해주세요!"라고 말하세요. 
        마지막에는 "아래 출처를 확인하여 더 정확한 정보를 얻거나 학생회, 또는 의과대학 행정실 (02-2286-1125)로 문의하세요."라는 문구를 적으세요. \n\n"""
        "**핫앵의 질문:** {question}\n"
        "**관련 정보:**\n{context}\n\n"
        "**답변:**"
    )
)

# Set up LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define function to retrieve and format context
def retrieve_and_format_docs(query):
    # with get_openai_callback() as cb:
    #     retrieved_docs = retriever.invoke(query)
    #     st.write(cb)

    retrieved_docs = retriever.invoke(query)
    retrieved_chunks_display = []

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown Source")
        text = f"{doc.page_content}\n\n📌 출처: {source}"
        retrieved_chunks_display.append(text)

    # Store retrieved chunks in session state
    st.session_state["retrieved_chunks"] = retrieved_chunks_display  

    return "\n\n---\n\n".join(retrieved_chunks_display)  # Separate retrieved chunks

# Define RAG chain
rag_chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | custom_prompt
    | llm
    | StrOutputParser()
)

# User input field
with st.form(key="question_form"):
    query = st.text_input("질문하세요:")
    submit_button = st.form_submit_button("Submit")

if submit_button:
    if query:
        # Retrieve context BEFORE invoking RAG
        formatted_context = retrieve_and_format_docs(query)

        # Invoke RAG with retrieved context
        response = rag_chain.invoke({"context": formatted_context, "question": query})

        # Show answer
        st.write("### 지니의 답변:")
        st.write(response)

        # **Add an expandable section for the retrieved chunks**
        with st.expander("📌 출처 표시"):
            if "retrieved_chunks" in st.session_state and st.session_state["retrieved_chunks"]:
                st.write("### 📌 참고한 문서 내용:")
                for chunk in st.session_state["retrieved_chunks"]:
                    st.write(chunk)  # Show first 500 characters for readability
                    st.write("\n")
            else:
                st.write("❌ 참고한 문서 조각이 없습니다.")

    else:
        st.warning("질문이 올바르지 않습니다. 다시 질문해 주세요.")