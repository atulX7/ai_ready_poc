from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import pathlib

# Step 1: Load all processed .txt files into documents
PROCESSED = pathlib.Path("data/processed")
docs = []

for file in PROCESSED.glob("*.txt"):
    content = file.read_text()
    docs.append(Document(page_content=content, metadata={"source": file.name}))

# Step 2: Load embedding model and create FAISS vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# Step 3: Save FAISS index locally
vectorstore.save_local("faiss_index")

print(f"✅ FAISS index created with {len(docs)} documents embedded.")

# Step 4: Optional: test search to validate embeddings
query = "What are the side effects of this medication?"
results = vectorstore.similarity_search(query, k=3)

print("\n🔍 Top 3 similar chunks:")
for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(doc.page_content[:300])  # print first 300 characters
