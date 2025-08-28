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
import requests
from operator import itemgetter
import shutil
import time
from typing import Optional

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

# --------------------------
# GitHub Auto-pull Settings
# --------------------------
GITHUB_OWNER = "SangwookCheon"
GITHUB_REPO = "kumedgenie"
GITHUB_BRANCH = "main"
GITHUB_DOC_PATH = "temp_rec"  # folder in the repo that holds your docs
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # optional; required for private repos or high traffic

def _github_headers():
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h

def _list_github_dir(owner, repo, path, branch="main"):
    """List items in a GitHub folder via Contents API."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    r = requests.get(api_url, headers=_github_headers(), timeout=30)
    r.raise_for_status()
    return r.json()

def _raw_github_url(owner, repo, branch, full_path):
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{full_path}"

def _download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with requests.get(url, headers=_github_headers(), timeout=60, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def _sync_dir_recursive(owner, repo, branch, repo_base_path, local_base_path):
    """
    Recursively mirrors the repo folder (repo_base_path) into local_base_path,
    but only downloads files whose extensions are supported by FILE_LOADERS.
    """
    items = _list_github_dir(owner, repo, repo_base_path, branch)
    allowed_exts = set(FILE_LOADERS.keys())
    downloaded_any = False

    for it in items:
        it_type = it.get("type")
        it_name = it.get("name")
        it_path = it.get("path")  # full path within repo

        if it_type == "dir":
            # Recurse into subdir
            sub_downloaded = _sync_dir_recursive(owner, repo, branch, it_path, local_base_path)
            downloaded_any = downloaded_any or sub_downloaded

        elif it_type == "file":
            ext = (it_name.split(".")[-1].lower() if "." in it_name else "")
            if ext in allowed_exts or it_name == "version.txt":
                raw_url = _raw_github_url(owner, repo, branch, it_path)
                relative_local = os.path.relpath(it_path, GITHUB_DOC_PATH)
                dest_path = os.path.join(local_base_path, relative_local) if ext in allowed_exts else os.path.join(FAISS_INDEX_PATH, "version.txt")
                # For content files (allowed_exts), place under DOC_FOLDER; for version.txt we keep using FAISS_INDEX_PATH/version.txt (handled elsewhere too)
                if ext in allowed_exts:
                    # place under DOC_FOLDER mirroring relative path
                    dest_path = os.path.join(local_base_path, relative_local)
                try:
                    _download_file(raw_url, dest_path)
                    downloaded_any = True
                except Exception as e:
                    st.error(f"❌ 원격 파일 다운로드 실패: {it_name} ({e})")
    return downloaded_any

def sync_docs_from_github():
    """Clear DOC_FOLDER and repopulate with the latest files from GitHub (recursive)."""
    try:
        # Clean doc folder for a fresh mirror
        if os.path.exists(DOC_FOLDER):
            shutil.rmtree(DOC_FOLDER)
        os.makedirs(DOC_FOLDER, exist_ok=True)

        downloaded = _sync_dir_recursive(
            owner=GITHUB_OWNER,
            repo=GITHUB_REPO,
            branch=GITHUB_BRANCH,
            repo_base_path=GITHUB_DOC_PATH,
            local_base_path=DOC_FOLDER,
        )
        if not downloaded:
            st.warning("⚠️ 원격 저장소에서 다운로드할 지원 파일이 없습니다.")
        return downloaded
    except requests.HTTPError as e:
        st.error(f"❌ GitHub API 오류: {e}")
        return False
    except Exception as e:
        st.error(f"❌ 원격 문서 동기화 중 오류가 발생했습니다: {e}")
        return False

# --------------------------
# Versioning (remote/local)
# --------------------------

# Use GitHub raw for live; keep your local fake line commented for dev
REMOTE_VERSION_URL = "https://raw.githubusercontent.com/SangwookCheon/kumedgenie/main/temp_rec/version.txt"
# REMOTE_VERSION_URL = "file:///" + os.path.abspath("fake_remote_version.txt")

LOCAL_VERSION_PATH = os.path.join(FAISS_INDEX_PATH, "version.txt")

# ---- cached version fetch (prevents hammering GitHub) ----
@st.cache_data(ttl=60, show_spinner=False)
def cached_fetch_remote_version(url: str, headers: dict | None = None) -> Optional[str]:
    try:
        r = requests.get(url, headers=headers or {}, timeout=20)
        if r.status_code == 200:
            return r.text.strip()
    except Exception:
        return None
    return None

def get_remote_version():
    try:
        if REMOTE_VERSION_URL.startswith("file:///"):
            local_path = REMOTE_VERSION_URL.replace("file:///", "")
            with open(local_path, "r") as f:
                return f.read().strip()
        else:
            return cached_fetch_remote_version(REMOTE_VERSION_URL, _github_headers())
    except Exception as e:
        st.error(f"❌ 원격 버전 정보를 가져오는 데 실패했습니다: {e}")
    return None

def get_local_version():
    if os.path.exists(LOCAL_VERSION_PATH):
        with open(LOCAL_VERSION_PATH, "r") as f:
            return f.read().strip()
    return None

def update_local_version(version):
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    with open(LOCAL_VERSION_PATH, "w") as f:
        f.write(version)

# --------------------------
# Build / Load Index Helpers
# --------------------------
def build_index_from_local_docs():
    all_files = []
    for ext in FILE_LOADERS.keys():
        all_files.extend(glob.glob(os.path.join(DOC_FOLDER, f"**/*.{ext}"), recursive=True))
        all_files.extend(glob.glob(os.path.join(DOC_FOLDER, f"*.{ext}")))  # in case no subfolders

    if not all_files:
        st.error("❌ 지원되는 문서가 없습니다")
        return None, None

    all_documents = []
    for file_path in all_files:
        ext = file_path.split(".")[-1].lower()
        if ext in FILE_LOADERS:
            loader = FILE_LOADERS[ext](file_path)
            documents = loader.load()
            for doc in documents:
                # Keep only a relative, readable source name
                rel = os.path.relpath(file_path, DOC_FOLDER)
                doc.metadata["source"] = rel
            all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    splits = text_splitter.split_documents(all_documents)

    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    vectorstore.save_local(FAISS_INDEX_PATH)
    retriever_local = vectorstore.as_retriever(search_kwargs={"k": n_search_kwargs})
    return vectorstore, retriever_local

# --------------------------
# Rebuild lock (single-writer)
# --------------------------
LOCK_PATH = os.path.join(FAISS_INDEX_PATH, ".rebuild.lock")

class FileLock:
    def __init__(self, path, timeout=180):
        self.path = path
        self.timeout = timeout
    def __enter__(self):
        start = time.time()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if time.time() - start > self.timeout:
                    break
                time.sleep(0.2)
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception:
            pass

# --------------------------
# Version check + Sync + Index
# --------------------------
remote_version = get_remote_version()
local_version = get_local_version()

needs_rebuild = (not os.path.exists(FAISS_INDEX_PATH)) or (remote_version and remote_version != local_version)

if needs_rebuild:
    st.warning("📂 새로운 원격 버전을 감지했습니다. 문서를 가져오고 인덱스를 재생성합니다...")
    with FileLock(LOCK_PATH, timeout=180):
        # Re-check inside the lock in case another worker already rebuilt
        local_version = get_local_version()
        needs_rebuild_locked = (not os.path.exists(FAISS_INDEX_PATH)) or (remote_version and remote_version != local_version)

        if needs_rebuild_locked:
            # 1) Pull latest docs from GitHub into DOC_FOLDER
            synced = sync_docs_from_github()

            # 2) Build index from freshly synced docs
            if synced:
                vectorstore, retriever = build_index_from_local_docs()
                if vectorstore and retriever:
                    if remote_version:
                        update_local_version(remote_version)
                    st.success("✅ 문서를 최신으로 동기화하고 인덱스를 재생성했습니다. 질문하세요.")
                else:
                    st.error("❌ 인덱스 생성에 실패했습니다.")
            else:
                # Even if sync failed, try to use existing local docs (best effort)
                vectorstore, retriever = build_index_from_local_docs()
                if vectorstore and retriever:
                    st.warning("⚠️ 원격 동기화에 실패하여 로컬 문서로 인덱스를 재생성했습니다.")
                else:
                    st.error("❌ 문서 동기화 및 인덱스 생성에 실패했습니다.")
        else:
            # Someone else finished rebuild while we waited → just load existing
            vectorstore = FAISS.load_local(FAISS_INDEX_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever(search_kwargs={"k": n_search_kwargs})
            st.success("✅ 기존 데이터를 불러왔습니다. 질문하세요!")
else:
    # Load existing index
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": n_search_kwargs})
    st.success("✅ 기존 데이터를 불러왔습니다. 질문하세요!")

# --------------------------
# Prompt + LLM
# --------------------------
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_text
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# --------------------------
# Retrieval + UI
# --------------------------
def retrieve_and_format_docs(query):
    retrieved_docs = retriever.invoke(query)
    retrieved_chunks_display = []

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown Source")
        text = f"{doc.page_content}\n\n📌 출처: {source}"
        retrieved_chunks_display.append(text)

    # Store retrieved chunks in session state
    st.session_state["retrieved_chunks"] = retrieved_chunks_display
    return "\n\n---\n\n".join(retrieved_chunks_display)

# Define RAG chain
rag_chain = (
    {"context": itemgetter("context"), "question": itemgetter("question")}
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
                    st.write(chunk)
                    st.write("\n")
            else:
                st.write("❌ 참고한 문서 조각이 없습니다.")
    else:
        st.warning("질문이 올바르지 않습니다. 다시 질문해 주세요.")

version_display = remote_version if remote_version else "알 수 없음"

st.markdown(
f"""
<hr style="margin-top: 50px;">
<p style="text-align: center; font-size: 14px;">
    Made with ❤️ by Wookie at &lt;/+/&gt;<br>
    <span style="font-size:12px; color:gray;">버전: {version_display}</span>
</p>
""",
unsafe_allow_html=True
)

#    방문 횟수: {visit_count}회 | 마지막 방문: {last_visit} <br>