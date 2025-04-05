# 🩺 Medical Chatbot using LangChain, Qdrant & LLaMA 3

A powerful medical question-answering chatbot that can read and understand PDF documents. This application uses **LangChain**, **Hugging Face embeddings**, **Qdrant** for vector search, and **Meta's LLaMA 3 8B Instruct** model for answering queries in natural language.

---

## 🚀 Features

- ✅ Ingests medical PDFs and chunks them for processing
- 🧠 Uses BAAI's `bge-base-en-v1.5` embeddings
- 🗂️ Stores embeddings in **Qdrant** for fast retrieval
- 🤖 Answers questions using **Meta-LLaMA 3** via HuggingFaceHub
- 🔍 Custom prompt for accurate, honest answers

---

## 🛠️ Requirements

```bash
pip install langchain qdrant-client huggingface_hub pypdf
```
> Also make sure to export your Hugging Face API token:
```bash
export HUGGINGFACEHUB_API_TOKEN=your_token_here
```

---

## 📁 Load Medical PDFs

```python
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

def load_pdf(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents
```

---

## ✂️ Text Splitting

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks
```

```python
extracted_data = load_pdf("path_to_your_pdf_folder")
text_chunks = text_split(extracted_data)
print("Length of my chunks:", len(text_chunks))
```

---

## 📥 Embedding Model

```python
from langchain.embeddings import HuggingFaceEmbeddings

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    return embeddings
```

---

## 🧠 LLM Setup (Meta LLaMA 3)

```python
from langchain_community.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={
        'max_new_tokens': 512,
        'temperature': 0.2,
        'return_full_text': False
    }
)
```

---

## 🧾 Prompt Template

```python
prompt = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

from langchain.prompts import PromptTemplate

PROMPT = PromptTemplate(
    template=prompt,
    input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}
```

---

## 🔎 Retriever + QA Chain

Assuming you already created a retriever using Qdrant:

```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriver,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)
```

---

## 💬 User Interaction

```python
user_input = input("Input Prompt: ")
result = qa({"query": user_input})
print("Response:", result["result"])
```

---

## 📊 Pipeline Recap

1. Load PDFs → `load_pdf()`
2. Chunk text → `text_split()`
3. Embed text → `download_hugging_face_embeddings()`
4. Store in Qdrant (notebook-dependent)
5. Define prompt → `PromptTemplate`
6. Set up LLM → `HuggingFaceHub`
7. Create QA chain → `RetrievalQA`
8. Interact → input + `qa()`

---

## 📌 Notes

- For medical use, always include a disclaimer: **"This chatbot is not a substitute for professional medical advice."**
- Fine-tuning on domain-specific medical data can improve performance.

---

## 📚 License

MIT License © 2025 — Subhranil Paul

