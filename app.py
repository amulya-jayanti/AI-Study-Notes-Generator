import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

import umap  # used indirectly via nlp_pipeline
import hdbscan  # used indirectly via nlp_pipeline

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from nlp_pipeline import (
    summarize_text,
)  # you also have extract_glossary_terms, cluster_topics_from_text if needed
from app_extensions import (
    render_glossary_tab,
    render_topics_tab,
    render_eval_tab,
)

# -------------------------------------------------------------------
# ENV + PAGE CONFIG
# -------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI Study Generator", layout="wide")
st.title("🎓 Personalized Study Notes Generator")
st.markdown(
    "Generates **Summaries**, **Flashcards**, and **Quizzes** from your textbook."
)

# -------------------------------------------------------------------
# SIDEBAR – UPLOAD
# -------------------------------------------------------------------
st.sidebar.header("Upload Material")
uploaded_file = st.sidebar.file_uploader("Upload Chapter PDF", type="pdf")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

# Reset vector store when a new file is uploaded
if (
    uploaded_file is not None
    and st.session_state.last_uploaded_name != uploaded_file.name
):
    st.session_state.vector_store = None
    st.session_state.last_uploaded_name = uploaded_file.name

# If no file at all, just show info and stop
if uploaded_file is None:
    st.info("👈 Upload a PDF in the sidebar to start.")
    st.stop()

# -------------------------------------------------------------------
# PROCESS PDF → BUILD VECTOR STORE (ONLY ONCE PER FILE)
# -------------------------------------------------------------------
if st.session_state.vector_store is None:
    with st.spinner("Processing PDF and building vector store..."):
        try:
            # Save uploaded file to a temp path so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Chunking for retrieval & summarization
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2500,
                chunk_overlap=350,
            )
            splits = text_splitter.split_documents(docs)

            # Embeddings + FAISS
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = FAISS.from_documents(splits, embeddings)

            st.session_state.vector_store = vector_store
            st.success("PDF processed and indexed!")
        except Exception as e:
            st.error(f"Error while processing PDF: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# If still no vector store, abort
if st.session_state.vector_store is None:
    st.error("Vector store could not be created. Please re-upload the PDF.")
    st.stop()

# -------------------------------------------------------------------
# SHARED OBJECTS (LLMs, retriever)
# -------------------------------------------------------------------
vector_store = st.session_state.vector_store
retriever = vector_store.as_retriever()

# Base LLM for QA, flashcards, etc.
base_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Faster map LLM + higher-quality reduce LLM for summaries
map_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
reduce_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# All doc chunks in one list for summarization / flashcards
docs = list(vector_store.docstore._dict.values())
full_text = "\n".join(d.page_content for d in docs)

# -------------------------------------------------------------------
# MAIN TABS
# -------------------------------------------------------------------
tab_main, tab_qa, tab_glossary, tab_topics, tab_eval = st.tabs(
    ["Summaries", "Q&A / MCQs", "Glossary", "Topics", "Evaluation"]
)

# ===================================================================
# TAB 1: SUMMARIES  (with sub-tabs: Summaries / Flashcards)
# ===================================================================
with tab_main:
    sub_tab_summaries, sub_tab_flashcards = st.tabs(
        ["📝 Summaries", "📇 Flashcards"]
    )

    # ------------------------ SUMMARIES ------------------------ #
    with sub_tab_summaries:
        st.subheader("Choose Summary Type")

        summary_type = st.selectbox(
            "Select the summary style you want:",
            [
                "TL;DR Summary (Ultra Short)",
                "Medium Summary (1–2 Paragraphs)",
                "Detailed Study Notes (Structured Notes)",
                "Full Summary (Map–Reduce)",
            ],
        )

        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                # ---------- TL;DR ----------
                if summary_type == "TL;DR Summary (Ultra Short)":
                    result = reduce_llm.invoke(
                        f"""
Generate a **very short TL;DR summary** (5 bullet points max)
for the following document:

{full_text}
"""
                    ).content

                    st.markdown("### ⚡ TL;DR Summary")
                    st.write(result)

                # ---------- MEDIUM ----------
                elif summary_type == "Medium Summary (1–2 Paragraphs)":
                    result = reduce_llm.invoke(
                        f"""
Write a **medium-length summary (1–2 paragraphs)** that clearly explains the main ideas 
of the following document in simple language:

{full_text}
"""
                    ).content

                    st.markdown("### 📘 Medium Summary (1–2 Paragraphs)")
                    st.write(result)

                # ---------- DETAILED STUDY NOTES ----------
                elif summary_type == "Detailed Study Notes (Structured Notes)":
                    result = reduce_llm.invoke(
                        f"""
Convert the following document into **detailed structured study notes**.

Your output format MUST be:

### 1. Key Concepts
- bullet points

### 2. Important Terms
- bullet points

### 3. Explanation of Each Section
#### Section Title
- bullet points
- examples if available

### 4. Formulas / Methods
- list any formulas or steps mentioned

### 5. Important Insights
- bullet points

Document:
{full_text}
"""
                    ).content

                    st.markdown("### 🧾 Detailed Study Notes")
                    st.write(result)

                # ---------- FULL MAP–REDUCE ----------
                elif summary_type == "Full Summary (Map–Reduce)":
                    mini_summaries = []
                    for d in docs:
                        chunk_summary = map_llm.invoke(
                            f"Compress into 1–2 key bullets:\n\n{d.page_content}"
                        ).content
                        mini_summaries.append(chunk_summary)

                    combined = "\n".join(mini_summaries)

                    final_summary = reduce_llm.invoke(
                        f"""
Combine the following compressed summaries into a **full structured summary**.

Include:
- A 5-bullet TL;DR  
- A 1–2 paragraph explanation  
- Key concepts explained simply  
- Important insights  
- Any definitions or formulas  

Summaries:
{combined}
"""
                    ).content

                    st.markdown("### 📘 Full Summary (Map–Reduce)")
                    st.write(final_summary)

    # ------------------------ FLASHCARDS ------------------------ #
    with sub_tab_flashcards:
        from keybert import KeyBERT
        import spacy
        import re
        import numpy as np

        st.header("📇 Flashcards")

        subtab1, subtab2, subtab3 = st.tabs(
            ["Concept Flashcards", "Definition Flashcards", "Formula Flashcards"]
        )

        # spaCy + KeyBERT
        nlp = spacy.load("en_core_web_sm")
        kw_model = KeyBERT(model="all-MiniLM-L6-v2")

        def render_flashcard(card_text, card_type):
            card_html = f"""
            <div style="
                background-color: #1e1e1e;
                border-radius: 12px;
                padding: 18px;
                margin-bottom: 15px;
                border: 1px solid #444;
                box-shadow: 0 2px 8px rgba(0,0,0,0.25);
            ">
                <h4 style="color:#ffcccc;margin-top:0;">🧠 {card_type} Flashcard</h4>
                <p style="font-size:15px;line-height:1.5;color:#e6e6e6;">
                    {card_text.replace("\n","<br>")}
                </p>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

        def extract_keyphrases(text):
            phrases = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                top_n=15,
            )
            return [p[0] for p in phrases]

        def extract_definitions(text):
            pattern = r"([A-Z][A-Za-z0-9\- ]+?) (is|are|refers to|means) (.+?)(\.|\n)"
            matches = re.findall(pattern, text)
            defs = [m[0] + " " + m[1] + " " + m[2] for m in matches]
            return defs[:15]

        def extract_formulas(text):
            formula_patterns = [
                r"[A-Za-z]\s*=\s*[^,\n]+",
                r"\$.*?\$",
                r"\\frac\{.*?\}\{.*?\}",
                r"[A-Z][a-z]*\([^)]+\)",
                r"[A-Za-z0-9_\-]+\s*[\+\-\*/]\s*[A-Za-z0-9_\-]+",
            ]
            found = []
            for p in formula_patterns:
                matches = re.findall(p, text)
                found.extend(matches)
            return list(set(found))[:10]

        def generate_flashcards_llm(title, items, llm):
            prompt = f"""
Create high-quality flashcards for the topic type: {title}.

Rules:
- DO NOT hallucinate. ONLY use the provided items.
- Keep answers very short (1-2 sentences).
- If formula is available, include it as 'Formula: <formula>'.
- Use this EXACT format:

Q: <question>
A: <answer>
Formula: <formula or NONE>
Type: {title}

Items:
{items}
"""
            return llm.invoke(prompt).content

        # Concept Flashcards
        with subtab1:
            if st.button("Generate Concept Flashcards"):
                with st.spinner("Extracting concepts..."):
                    keyphrases = extract_keyphrases(full_text)
                    flashcards = generate_flashcards_llm("Concept", keyphrases, base_llm)
                    st.markdown("### 🧠 Concept Flashcards")
                    for block in flashcards.split("\n\n"):
                        if block.strip():
                            render_flashcard(block, "Concept")

        # Definition Flashcards
        with subtab2:
            if st.button("Generate Definition Flashcards"):
                with st.spinner("Extracting definitions..."):
                    definitions = extract_definitions(full_text)
                    flashcards = generate_flashcards_llm("Definition", definitions, base_llm)
                    st.markdown("### 📘 Definition Flashcards")
                    for block in flashcards.split("\n\n"):
                        if block.strip():
                            render_flashcard(block, "Definition")

        # Formula Flashcards
        with subtab3:
            if st.button("Generate Formula Flashcards"):
                with st.spinner("Extracting formulas..."):
                    formulas = extract_formulas(full_text)
                    flashcards = generate_flashcards_llm("Formula", formulas, base_llm)
                    st.markdown("### 📐 Formula Flashcards")
                    for block in flashcards.split("\n\n"):
                        if block.strip():
                            render_flashcard(block, "Formula")

# ===================================================================
# TAB 2: Q&A / MCQs
# ===================================================================
with tab_qa:
    qa_tab, quiz_tab = st.tabs(["Ask Questions", "Quizzes"])

    # ------------------------ FREE-FORM QA ------------------------ #
    with qa_tab:
        st.subheader("Ask a Question about this Chapter")

        user_q = st.text_input("Enter your question")
        if st.button("Get Answer"):
            if not user_q.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving answer..."):
                    chain = RetrievalQA.from_chain_type(
                        llm=base_llm, chain_type="stuff", retriever=retriever
                    )
                    res = chain.invoke({"query": user_q})
                    st.markdown("### Answer")
                    st.write(res["result"])

    # ------------------------ QUIZZES ------------------------ #
    with quiz_tab:
        st.subheader("Generate MCQ Quiz")

        topic = st.text_input("Quiz Topic (optional; leave blank for whole chapter)")

        if st.button("Generate MCQ"):
            with st.spinner("Creating quiz question..."):
                chain = RetrievalQA.from_chain_type(
                    llm=base_llm, chain_type="stuff", retriever=retriever
                )
                q = (
                    f"Create a challenging MCQ on {topic if topic else 'this chapter'}. "
                    "Format:\nQuestion\nA)\nB)\nC)\nD)\nCorrect Answer: X"
                )
                res = chain.invoke({"query": q})
                st.markdown("### Generated Question")
                st.write(res["result"])

# ===================================================================
# TAB 3: GLOSSARY  (reuses helper)
# ===================================================================
with tab_glossary:
    render_glossary_tab()

# ===================================================================
# TAB 4: TOPICS  (reuses helper)
# ===================================================================
with tab_topics:
    render_topics_tab()

# ===================================================================
# TAB 5: EVALUATION  (reuses helper)
# ===================================================================
with tab_eval:
    render_eval_tab()
