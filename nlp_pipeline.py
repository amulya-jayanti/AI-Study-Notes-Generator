# nlp_pipeline.py

"""
Core NLP pipeline utilities for the Study Notes app:
- Summarization wrapper
- Glossary / entity extraction
- Topic clustering with UMAP + HDBSCAN
"""

from typing import List, Dict, Tuple

import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI


# --------------------------------------------------------------------
# Models
# --------------------------------------------------------------------

summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY"))

# spaCy NER
_nlp = spacy.load("en_core_web_sm")

# Sentence embeddings for clustering
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------------------------
# Summarization
# --------------------------------------------------------------------

def summarize_text(text: str, level: str = "medium") -> str:
    """
    Call the LLM to summarize text at a given level.
    level: "short" | "medium" | "detailed"
    """
    if not text.strip():
        return ""

    if level == "short":
        style = "Give a 2–3 sentence TL;DR summary suitable for quick revision."
    elif level == "detailed":
        style = (
            "Create detailed, structured study notes with headings and bullet points. "
            "Focus on key concepts, definitions, and relationships."
        )
    else:
        style = (
            "Give a concise paragraph summary that captures the main ideas and key details."
        )

    prompt = (
        f"{style}\n\n"
        "Text:\n"
        f"{text}\n\n"
        "Summary:"
    )

    resp = summary_llm.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)


# --------------------------------------------------------------------
# Glossary / Entity Extraction
# --------------------------------------------------------------------

def extract_glossary_terms(text: str, min_len: int = 3) -> List[Dict]:
    """
    Extract named entities from the text and return as potential glossary terms.
    Does NOT require your custom glossary; this is raw extraction.
    """
    doc = _nlp(text)
    terms = []

    seen = set()
    for ent in doc.ents:
        term = ent.text.strip()
        if len(term) < min_len:
            continue

        key = (term.lower(), ent.label_)
        if key in seen:
            continue
        seen.add(key)

        terms.append(
            {
                "term": term,
                "category": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            }
        )

    return terms


# --------------------------------------------------------------------
# Topic clustering
# --------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    """
    Simple paragraph-based chunking used for topic clustering.
    Your app.py already has LangChain chunking for retrieval –
    this is just for visualization / topic discovery.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    # If paragraphs are too short/long, you can further merge/split here.
    return paras


def cluster_topics_from_text(
    text: str,
    min_cluster_size: int = 5,
) -> Tuple[List[int], np.ndarray, List[str]]:
    """
    Cluster paragraphs of the given text into topics.
    Returns:
      labels: cluster label for each chunk (-1 = noise)
      emb_2d: 2D embeddings from UMAP (for visualization)
      chunks: chunk texts
    """
    chunks = chunk_text(text)
    if len(chunks) < min_cluster_size + 1:
        return [], np.array([]), chunks

    embeddings = _embedder.encode(chunks, show_progress_bar=False)

    reducer = umap.UMAP(
        n_neighbors=15, n_components=2, min_dist=0.0, random_state=42
    )
    emb_2d = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(emb_2d)

    return labels.tolist(), emb_2d, chunks


def summarize_cluster(chunks: List[str]) -> str:
    """
    Summarize a cluster's chunks into a human-readable topic label/summary.
    """
    if not chunks:
        return ""
    text = "\n\n".join(chunks[:5])  
    prompt = (
        "You are labeling topics for study notes.\n"
        "Given the following snippets from the same topic, provide:\n"
        "1) A short topic name (3–6 words)\n"
        "2) A one-sentence description.\n\n"
        f"Snippets:\n{text}\n\n"
        "Respond in the format:\n"
        "Topic: <short name>\n"
        "Description: <one sentence>"
    )
    resp = summary_llm.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)
