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
    UnstructuredMarkdownLoader,  # NOTE: This may require NLTK
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OPENAI_API_KEY is missing. Please set it in `.env` or Streamlit Secrets.")
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key

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
    retriever = vectorstore.as_retriever() #search_kwargs={"k": 5}
else:
    # If no FAISS index exists, process documents
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
                    doc.metadata["source"] = os.path.basename(file_path)  # Store only filename, not full path

                all_documents.extend(documents)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                       chunk_overlap=200,
                                                       separators=["\n# ", "\n## ", "\n### "])
        splits = text_splitter.split_documents(all_documents)

        # for i, doc in enumerate(splits[:3]):
        #     print(f"\nChunk {i+1}:\n{doc.page_content}\n")

        # Verify chunks
        # for i, doc in enumerate(splits[:3]):
        #     print(f"\nChunk {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}\n")

        # Embed and store in FAISS
        vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
        vectorstore.save_local(FAISS_INDEX_PATH)  # Save index
        retriever = vectorstore.as_retriever() #search_kwargs={"k": 5}

        st.success("✅ 문서가 성공적으로 저장되었습니다! 질문하세요.")

# Custom RAG prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "질문을 할 학생은 고려대 의대생이고, 다음 질문에 대한 답변을 제공하세요. "
        "모든 답변은 **한국어**로 작성되어야 하며, 정중하고 공식적인 톤을 유지하고, 개인적인 의견이 없어야 됩니다. 출처에서 제공되는 사실만 말하고 확대 해석하지 마세요."
        """가능한 경우, 출처에서 직접 인용(\" \")하고, 관련 조항이 있으면 생략하지 말고 나열하세요.
        또한, 학점, 기간, 비용, 시간, 수업 번호, 웹페이지 링크, 조항 등 수치적 또는 구체적인 정보가 있으면 생략하지 말고 제공하세요. 
        질문이 지나치게 정확하지 않은 것 같으면 질문이 정확한지 확인하거나 조금 더 구체적으로 질문해 달라고 부탁하세요. 
        반드시 출처를 유지하세요. 각 정보가 어느 문서에서 왔는지 명확히 밝히세요.
        출처에서 정보가 정확하지 않으면 관련 정보를 다시 확인하라고 하세요. 그렇지 않으면 마지막에 '이 정보는 정확하지 않을 수 있습니다.'라는 경고 문구를 포함하세요. \n\n"""
        "**질문:** {question}\n"
        "**출처:**\n{context}\n\n"
        "**답변 (출처 포함):**"
    )
)

# Set up LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define RAG chain
def format_docs(docs):
    formatted_texts = []
    
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")  # Get source filename
        formatted_texts.append(f"{doc.page_content}\n\n📌 출처: {source}")  # Append source after content
    
    return "\n\n---\n\n".join(formatted_texts)  # Separate retrieved chunks with "---"

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
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
        response = rag_chain.invoke(query)
        st.write("### 지니의 답변:")
        st.write(response)
    else:
        st.warning("질문이 올바르지 않습니다. 다시 질문해 주세요.")