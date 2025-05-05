import streamlit as st
import os, json
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from difflib import SequenceMatcher

# 🔐 Hugging Face Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# 🧠 Load Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 📊 Sidebar: Load Metrics
st.sidebar.title("📊 AI Trust Score Summary")
with open("data/processed/metrics.json") as f:
    metrics = json.load(f)
df_metrics = pd.DataFrame(metrics)

# 🧠 Split docs by AI Trust Score
all_docs = []
doc_dir = "data/processed"
for file in os.listdir(doc_dir):
    if file.endswith(".txt"):
        path = os.path.join(doc_dir, file)
        base_file = file.split("_")[0] + ".pdf"
        score = next((m["ai_trust_score"] for m in metrics if m["file"] == base_file), 0)
        content = open(path, encoding="utf-8").read()
        all_docs.append((Document(page_content=content, metadata={"source": file}), score))

# 🧠 Filter docs
ai_ready_docs = [doc for doc, score in all_docs if score >= 0.75]
non_ai_ready_docs = [doc for doc, score in all_docs if score < 0.75]

# 🤖 LLM
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    temperature=0.3,
    max_new_tokens=512
)

# 📦 Vectorstores & QA Chains
ai_qa = None
non_ai_qa = None
if ai_ready_docs:
    ai_ready_vs = FAISS.from_documents(ai_ready_docs, embedding_model)
    ai_retriever = ai_ready_vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    ai_qa = RetrievalQA.from_chain_type(llm=llm, retriever=ai_retriever, return_source_documents=True)

if non_ai_ready_docs:
    non_ai_ready_vs = FAISS.from_documents(non_ai_ready_docs, embedding_model)
    non_ai_retriever = non_ai_ready_vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    non_ai_qa = RetrievalQA.from_chain_type(llm=llm, retriever=non_ai_retriever, return_source_documents=True)

# 📊 Sidebar Summary
avg_score = df_metrics["ai_trust_score"].mean()
st.sidebar.markdown(f"### Avg Trust Score: **{avg_score:.2f}**")
if avg_score >= 0.75:
    st.sidebar.success("✅ High Trust — AI-Ready Data")
elif avg_score >= 0.5:
    st.sidebar.warning("⚠️ Medium Trust — Partially Usable")
else:
    st.sidebar.error("❌ Low Trust — Not AI-Ready")
st.sidebar.caption("🟢 0.75+ = High | 🟡 0.5–0.74 = Medium | 🔴 < 0.5 = Low")

st.sidebar.subheader("Trust Score per Document")
chart_data = df_metrics[["file", "ai_trust_score"]].set_index("file")
st.sidebar.bar_chart(chart_data)

# 💬 Chat Interface
st.title("🤖 AI-Ready Chatbot vs Non-AI-Ready Chatbot")
st.markdown("Enter your question and compare answers from AI-ready and non-AI-ready data.")
query = st.chat_input("Ask a question about the drug labels...")

if query:
    st.chat_message("user").write(query)

    if ai_qa:
        ai_response = ai_qa({"query": query})
        with st.chat_message("assistant"):
            st.markdown("🟢 **AI-Ready Bot Response**")
            st.write(ai_response["result"])
            st.markdown("**Sources:**")
            for doc in ai_response["source_documents"]:
                base_file = doc.metadata["source"].split("_")[0] + ".pdf"
                score = next((m["ai_trust_score"] for m in metrics if m["file"] == base_file), None)
                badge = "🟢" if score and score >= 0.75 else "🟡" if score and score >= 0.5 else "🔴"
                st.markdown(f"- `{doc.metadata['source']}` — {badge} Trust Score: **{score:.2f}**" if score else f"- `{doc.metadata['source']}`")
    else:
        st.chat_message("assistant").warning("No AI-ready documents available.")

    if non_ai_qa:
        non_response = non_ai_qa({"query": query})
        with st.chat_message("assistant"):
            st.markdown("🔴 **Non-AI-Ready Bot Response**")
            st.write(non_response["result"])
            st.markdown("**Sources:**")
            for doc in non_response["source_documents"]:
                base_file = doc.metadata["source"].split("_")[0] + ".pdf"
                score = next((m["ai_trust_score"] for m in metrics if m["file"] == base_file), None)
                badge = "🟢" if score and score >= 0.75 else "🟡" if score and score >= 0.5 else "🔴"
                st.markdown(f"- `{doc.metadata['source']}` — {badge} Trust Score: **{score:.2f}**" if score else f"- `{doc.metadata['source']}`")
    else:
        st.chat_message("assistant").warning("No non-AI-ready documents available.")

    # 🔍 Similarity Score
    if ai_qa and non_ai_qa:
        similarity = SequenceMatcher(None, ai_response["result"], non_response["result"]).ratio()
        with st.chat_message("assistant"):
            st.markdown("---")
            st.markdown(f"### 🔍 Response Similarity Score: `{similarity:.2f}`")
            if similarity < 0.6:
                st.error("Large difference — AI-Ready content provided significantly better guidance.")
            elif similarity < 0.85:
                st.warning("Moderate difference — noticeable variation in content.")
            else:
                st.success("Minimal difference — both data sources performed similarly.")

