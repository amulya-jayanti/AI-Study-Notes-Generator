# app_extensions.py

"""
Helpers to plug into your existing Streamlit app.py:
- Glossary tab
- Topic clustering tab
- Evaluation tab (for ROUGE, etc.)
"""

import json
import os
from typing import List
import streamlit as st
from rouge_score import rouge_scorer
import re
import pandas as pd
from keybert import KeyBERT
import spacy
from langchain_openai import ChatOpenAI
from dataset_loader import load_evaluation_datasets, load_custom_glossary
from nlp_pipeline import (
    extract_glossary_terms,
    cluster_topics_from_text,
    summarize_cluster,
    summarize_text,
)

# Keyphrase extractor (important concepts)
_glossary_kw_model = KeyBERT("all-MiniLM-L6-v2")

# spaCy model (for optional debug NER)
_glossary_nlp = spacy.load("en_core_web_sm")

# LLM for generating definitions
_glossary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# --------------------------------------------------------------------
# Utility: to get concatenated text from your vector_store docs
# --------------------------------------------------------------------
def get_full_text_from_session_docs(max_chars: int = 15000) -> str:
    """Return concatenated text from docs in the current vector_store."""
    vs = st.session_state.get("vector_store")
    if vs is None:
        return ""
    try:
        docs = list(vs.docstore._dict.values())
    except Exception:
        return ""
    combined = "\n\n".join(d.page_content for d in docs)
    return combined[:max_chars]


# --------------------------------------------------------------------
# Glossary Tab
# --------------------------------------------------------------------

def _extract_glossary_keyphrases(text: str, top_n: int = 25) -> list[str]:
    """Use KeyBERT to get candidate glossary terms from the chapter."""
    if not text.strip():
        return []

    phrases = _glossary_kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=top_n,
    )
    candidates = []
    for term, score in phrases:
        term = term.strip()
        # Basic cleaning: no tiny strings, no digits/emails/urls
        if len(term) < 3:
            continue
        if re.search(r"\d", term):
            continue
        if "@" in term or "http" in term:
            continue
        candidates.append(term)

    # Deduplicate preserving order
    seen = set()
    clean = []
    for t in candidates:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            clean.append(t)
    return clean


def _define_terms_with_llm(terms: list[str], context: str) -> list[dict]:
    """
    Ask the LLM to create simple 1–2 sentence definitions for each term,
    using ONLY the given context.
    Returns: [{"term": ..., "definition": ...}, ...]
    """
    if not terms:
        return []

    prompt = f"""
You are creating a student-friendly glossary for a textbook chapter.

Chapter text:
{context[:8000]}

Create a glossary entry for each of the following terms, using ONLY the information
that can reasonably be inferred from the chapter. If the chapter does not clearly
define a term, give a short high-level description.

Terms:
{json.dumps(terms, ensure_ascii=False)}

Respond ONLY as a JSON list in this format:
[
  {{"term": "...", "definition": "..."}},
  ...
]
"""
    resp = _glossary_llm.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)

    try:
        data = json.loads(text)
        # Normalize a bit
        clean = []
        for item in data:
            term = item.get("term", "").strip()
            definition = item.get("definition", "").strip()
            if term and definition:
                clean.append({"term": term, "definition": definition})
        # Fallback if parsing produced nothing
        if clean:
            return clean
    except Exception:
        pass

    # Very defensive fallback: generic one-liners
    return [
        {"term": t, "definition": "Key concept mentioned in this chapter."}
        for t in terms
    ]


def render_glossary_tab():
    st.subheader("Glossary Extraction")

    from app_extensions import get_full_text_from_session_docs  # avoid circular import
    full_text = get_full_text_from_session_docs()
    if not full_text:
        st.info("Upload and process a PDF first (so a vector store is built).")
        return

    if st.button("Extract Glossary"):
        with st.spinner("Identifying key concepts and building glossary..."):
            # 1) Load custom glossary (if available)
            custom_gloss = load_custom_glossary()
            custom_terms_map = {}
            if custom_gloss and isinstance(custom_gloss, dict):
                for t in custom_gloss.get("terms", []):
                    key = t.get("term", "").lower().strip()
                    if key:
                        custom_terms_map[key] = t

            # 2) Use KeyBERT to find chapter-specific keyphrases
            keyphrases = _extract_glossary_keyphrases(full_text, top_n=25)

            # 3) Split into: (a) matches custom glossary, (b) new terms
            matched_custom = []
            new_terms = []
            for kp in keyphrases:
                key = kp.lower()
                if key in custom_terms_map:
                    entry = custom_terms_map[key]
                    matched_custom.append(
                        {
                            "term": entry.get("term", kp),
                            "subject": entry.get("subject", ""),
                            "category": entry.get("category", ""),
                            "definition": entry.get("definition", ""),
                            "source": "course_glossary",
                        }
                    )
                else:
                    new_terms.append(kp)

            # 4) Ask LLM to define the new terms using chapter context
            inferred_entries = []
            if new_terms:
                inferred_defs = _define_terms_with_llm(new_terms, full_text)
                for item in inferred_defs:
                    inferred_entries.append(
                        {
                            "term": item["term"],
                            "subject": "",
                            "category": "",
                            "definition": item["definition"],
                            "source": "chapter_inferred",
                        }
                    )

            # 5) Optional: NER debug entities
            raw_ents = extract_glossary_terms(full_text)

        # ---------------- DISPLAY ----------------
        if matched_custom:
            st.markdown(f"### 📚 Terms from course glossary ({len(matched_custom)})")
            df_custom = pd.DataFrame(matched_custom)
            st.dataframe(df_custom[["term", "subject", "category", "definition"]])

        if inferred_entries:
            st.markdown(
                f"### ✨ Additional concepts inferred from this chapter ({len(inferred_entries)})"
            )
            df_inferred = pd.DataFrame(inferred_entries)
            st.dataframe(df_inferred[["term", "definition"]])

        if not matched_custom and not inferred_entries:
            st.warning(
                "No meaningful glossary terms could be extracted. "
                "Try another chapter or check that the text is in English."
            )

        with st.expander("Show all detected named entities (debug)", expanded=False):
            st.caption(
                "These are the raw spaCy NER entities (PERSON, ORG, DATE, etc.) "
                "used only for debugging, not for the main glossary."
            )
            st.dataframe(raw_ents)


# --------------------------------------------------------------------
# Topic Clustering Tab
# --------------------------------------------------------------------

# Global keyphrase model for fallback topics
_topics_kw_model = KeyBERT("all-MiniLM-L6-v2")


def render_topics_tab():
    import streamlit as st
    from app_extensions import get_full_text_from_session_docs  # if helper is in same file, you can remove this line and just call it

    st.subheader("Key Topics & Themes")

    # Get up to ~20k chars of text from current document chunks
    full_text = get_full_text_from_session_docs(max_chars=20000)
    if not full_text:
        st.info("Upload and process a PDF first (so a vector store is built).")
        return

    min_cluster_size = st.slider("Minimum cluster size", 3, 15, 5)

    if st.button("Discover Topics"):
        with st.spinner("Clustering text into thematic groups..."):
            labels, emb_2d, chunks = cluster_topics_from_text(
                full_text, min_cluster_size=min_cluster_size
            )

        # If clustering completely failed or returned nothing
        if not labels or not chunks:
            st.warning("Not enough content for stable clusters. Showing key phrases instead.")
            return _render_keyword_topics(full_text)

        # Check if there is at least one real cluster (label != -1)
        unique_labels = sorted(set(labels) - {-1})
        if not unique_labels:
            st.warning("All chunks were considered noise by HDBSCAN. Showing key phrases instead.")
            return _render_keyword_topics(full_text)

        # ----------------- Show cluster-based topics -----------------
        import pandas as pd

        df = pd.DataFrame({"label": labels, "text": chunks})

        st.markdown("### Topics discovered in this chapter")

        for i, cid in enumerate(unique_labels, start=1):
            cluster_texts = df[df["label"] == cid]["text"].tolist()
            if not cluster_texts:
                continue

            topic_desc = summarize_cluster(cluster_texts)

            st.markdown(f"#### Topic {i}")
            st.code(topic_desc)

            st.markdown("**Example snippet:**")
            st.write(cluster_texts[0][:300] + "…")

        # Optional: show how many chunks were noise
        noise_count = sum(1 for l in labels if l == -1)
        if noise_count:
            st.caption(f"{noise_count} chunks were treated as noise by HDBSCAN.")


def _render_keyword_topics(full_text: str):
    """Fallback: show top keyphrases when clustering doesn't produce usable topics."""
    import streamlit as st

    st.markdown("### Top key phrases in this chapter")

    phrases = _topics_kw_model.extract_keywords(
        full_text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=15,
    )

    topics = [p[0] for p in phrases]
    df_topics = pd.DataFrame({"topic": topics})

    st.dataframe(df_topics)

    st.caption(
        "These key phrases are extracted using semantic similarity (KeyBERT). "
        "They give a quick sense of the main themes even when clustering is unstable."
    )


# --------------------------------------------------------------------
# Evaluation Tab (Summarization ROUGE on ML-ArXiv)
# --------------------------------------------------------------------

import os
from rouge_score import rouge_scorer
from langchain_openai import ChatOpenAI
from dataset_loader import load_evaluation_datasets
from nlp_pipeline import summarize_text  # we already import others above


def _simple_f1(pred: str, gold: str) -> float:
    """Very simple token-level F1 for QA answers."""
    import re

    def normalize(s):
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        tokens = s.split()
        return tokens

    pred_tokens = normalize(pred)
    gold_tokens = normalize(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def render_eval_tab():
    st.subheader("Model Evaluation")

    manager = load_evaluation_datasets()

    task = st.selectbox(
        "Task",
        [
            "Summarization (ML-ArXiv)",
            "QA Answering (AI/ML concepts – aiml_concepts_400)",
            "QA Answering (PubMedQA – qa_pubmed)",
        ],
    )

    n_samples = st.slider("Number of samples", 5, 30, 10)

    # ----------------- Summarization Evaluation ----------------- #
    if task.startswith("Summarization"):
        if st.button("Run ROUGE evaluation"):
            samples = manager.get_summarization_sample("ml_arxiv", n=n_samples)
            scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

            scores = []
            progress = st.progress(0.0)

            for i, ex in enumerate(samples):
                text = ex["document"]
                ref = ex["reference_summary"]

                # Use your summarizer
                pred = summarize_text(text, level="medium")

                s = scorer.score(ref, pred)
                scores.append(s)
                progress.progress((i + 1) / len(samples))

            avg_rouge1 = sum(x["rouge1"].fmeasure for x in scores) / len(scores)
            avg_rougeL = sum(x["rougeL"].fmeasure for x in scores) / len(scores)

            st.success("Summarization evaluation complete!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ROUGE-1 (F1)", f"{avg_rouge1*100:.1f}%")
            with col2:
                st.metric("ROUGE-L (F1)", f"{avg_rougeL*100:.1f}%")

            st.caption(
                f"Evaluated on {len(scores)} ML-ArXiv papers. "
                "ROUGE scores measure n-gram overlap between model and reference summaries; "
                "values around 10-30% are typical for long, technical texts with small models."
            )

    # ----------------- QA Evaluation (generic pattern) ----------------- #
    else:
        if st.button("Run QA evaluation"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OPENAI_API_KEY is not set. Please configure it first.")
                return

            qa_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                api_key=api_key,
            )

            if "AI/ML concepts" in task:
                qa_samples = manager.get_qa_sample("aiml_concepts_400", n=n_samples)
            else:
                qa_samples = manager.get_qa_sample("pubmed_qa", n=n_samples)

            f1_scores = []
            progress = st.progress(0.0)

            for i, ex in enumerate(qa_samples):
                question = ex.get("question", "")
                gold_answer = ex.get("answer", "")

                # Some of your QA datasets may have "context" field; use it if present
                context = ex.get("context", "")

                prompt = (
                    "You are an expert tutor. Answer the question based ONLY on the given context.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n"
                    "Answer:"
                )
                resp = qa_llm.invoke(prompt)
                pred_answer = resp.content if hasattr(resp, "content") else str(resp)

                f1 = _simple_f1(pred_answer, gold_answer)
                f1_scores.append(f1)

                progress.progress((i + 1) / len(qa_samples))

            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

            st.success("QA evaluation complete!")

            st.metric("Average token-level F1", f"{avg_f1*100:.1f}%")
            st.caption(
                f"Evaluated on {len(f1_scores)} Q&A pairs from the selected dataset. "
                "F1 measures overlap between the model's answer and the reference answer."
            )


