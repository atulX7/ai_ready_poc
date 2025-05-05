from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pathlib

RAW = pathlib.Path("data/raw")
PROCESSED = pathlib.Path("data/processed")
PROCESSED.mkdir(exist_ok=True)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

for pdf in RAW.glob("*.pdf"):
    loader = PyPDFLoader(str(pdf))
    docs = loader.load()
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        fname = f"{pdf.stem}_{i}.txt"
        (PROCESSED / fname).write_text(chunk.page_content)
