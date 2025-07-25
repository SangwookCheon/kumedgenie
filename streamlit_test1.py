import streamlit as st
import os
import glob
import warnings
import pandas as pd
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

# KUMEDGENIE 

#custom functions
from visit_counter import get_visit_count, update_visit_count, get_last_visit, update_last_visit  # Import visit tracking

# Load environment variables
load_dotenv()

# HYPERPARAMETERS
n_search_kwargs = 3

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
prompt_text = os.getenv("PROMPT_TEMPLATE")

if not openai_api_key:
    st.error("❌ OPENAI_API_KEY is missing. Please set it in `.env` or Streamlit Secrets.")
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize session state for retrieved chunks
if "retrieved_chunks" not in st.session_state:
    st.session_state["retrieved_chunks"] = []


# Visit Counter implementation
visit_count = update_visit_count()
last_visit = get_last_visit()
update_last_visit()

# Streamlit UI
st.title("🐯🔍 고려대 의과대학 정보 지니")
st.write("지니는 신입생 수강 신청 정보와 유용한 팁, 고려대학교 의과대학 학칙(휴·복학, 이중 전공, 장학생 기준 등)에 대한 궁금증을 해결해 드립니다!")

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
        st.error("❌ 지원되는 문서가 없습니다")
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
    template=prompt_text
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
    query = st.text_input("질문하세요 (질문이 구체적일수록 더욱 정확한 답변을 받을 수 있습니다):")
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

        # Add an expandable section for the retrieved chunks
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
    

st.markdown(
f"""
<hr style="margin-top: 50px;">
<p style="text-align: center; font-size: 14px;">
    Made with ❤️ by Wookie at &lt;/+/&gt;
</p>
""",
unsafe_allow_html=True
)

#    방문 횟수: {visit_count}회 | 마지막 방문: {last_visit} <br>