# dataset_loader.py

"""
Dataset Loader Utility
Easy loading and access to all evaluation datasets.
"""

import json
import os
from typing import Dict, List
import pandas as pd  


class DatasetLoader:
    """Centralized dataset loader for all evaluation datasets"""

    def __init__(self, base_path: str = "datasets"):
        self.base_path = base_path
        self.datasets: Dict[str, Dict] = {}

    # ---------------------------------------------------------
    # Master loader
    # ---------------------------------------------------------
    def load_all(self) -> Dict:
        """Load all available datasets"""
        print("Loading evaluation datasets...")

        self.datasets = {
            "summarization": self.load_summarization_datasets(),
            "qa": self.load_qa_datasets(),
            "glossary": self.load_glossary_datasets(),
        }

        return self.datasets

    # ---------------------------------------------------------
    # Summarization datasets
    # ---------------------------------------------------------
    def load_summarization_datasets(self) -> Dict:
        """Load all summarization datasets"""
        datasets: Dict[str, List[dict]] = {}

        # ML-ArXiv summarization
        ml_arxiv_path = f"{self.base_path}/cleaned/summarization_ml_arxiv.json"
        if os.path.exists(ml_arxiv_path):
            with open(ml_arxiv_path, "r", encoding="utf-8") as f:
                datasets["ml_arxiv"] = json.load(f)
                print(f"Loaded {len(datasets['ml_arxiv'])} ML-ArXiv papers for summarization")

        return datasets

    # ---------------------------------------------------------
    # Q&A datasets
    # ---------------------------------------------------------
    def load_qa_datasets(self) -> Dict:
        """Load all Q&A datasets"""
        datasets: Dict[str, List[dict]] = {}

        # PubMedQA – scientific / AI-in-healthcare flavor
        pubmed_path = f"{self.base_path}/cleaned/qa_pubmed.json"
        if os.path.exists(pubmed_path):
            with open(pubmed_path, "r", encoding="utf-8") as f:
                datasets["pubmed_qa"] = json.load(f)
                print(f"Loaded {len(datasets['pubmed_qa'])} PubMed QA pairs")

        # Combined AI/ML conceptual + libraries Q&A (400 total)
        aiml_path = f"{self.base_path}/cleaned/qa_aiml_concepts_400.json"
        if os.path.exists(aiml_path):
            with open(aiml_path, "r", encoding="utf-8") as f:
                datasets["aiml_concepts_400"] = json.load(f)
                print(f"Loaded {len(datasets['aiml_concepts_400'])} AI/ML conceptual Q&A pairs")

        return datasets

    # ---------------------------------------------------------
    # Glossary datasets
    # ---------------------------------------------------------
    def load_glossary_datasets(self) -> Dict:
        """Load glossary datasets"""
        datasets: Dict[str, dict] = {}

        # Custom glossary (from build_glossary.py)
        custom_path = f"{self.base_path}/custom/custom_glossary.json"
        if os.path.exists(custom_path):
            with open(custom_path, "r", encoding="utf-8") as f:
                datasets["custom"] = json.load(f)
                meta = datasets["custom"].get("metadata", {})
                total_terms = meta.get("total_terms") or len(
                    datasets["custom"].get("terms", [])
                )
                print(f"Loaded {total_terms} custom glossary terms")

        return datasets

    # ---------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------
    def get_sample(self, dataset_type: str, dataset_name: str, n: int = 5) -> List[dict]:
        """Get n random samples from a dataset"""
        import random

        if dataset_type not in self.datasets:
            return []

        if dataset_name not in self.datasets[dataset_type]:
            return []

        data = self.datasets[dataset_type][dataset_name]

        # Handle glossary structure
        if isinstance(data, dict) and "terms" in data:
            data = data["terms"]

        if not data:
            return []

        return random.sample(data, min(n, len(data)))

    def get_stats(self) -> Dict:
        """Get statistics about loaded datasets"""
        stats = {
            "summarization": {},
            "qa": {},
            "glossary": {},
            "total_items": 0,
        }

        # Summarization stats
        for name, data in self.datasets.get("summarization", {}).items():
            count = len(data)
            stats["summarization"][name] = count
            stats["total_items"] += count

        # Q&A stats
        for name, data in self.datasets.get("qa", {}).items():
            count = len(data)
            stats["qa"][name] = count
            stats["total_items"] += count

        # Glossary stats
        for name, data in self.datasets.get("glossary", {}).items():
            if isinstance(data, dict) and "terms" in data:
                count = len(data["terms"])
            else:
                count = len(data)
            stats["glossary"][name] = count
            stats["total_items"] += count

        return stats


class EvaluationDatasetManager:
    """Manages datasets specifically for evaluation purposes"""

    def __init__(self):
        self.loader = DatasetLoader()
        self.datasets = self.loader.load_all()

    def get_summarization_sample(self, doc_type: str = "ml_arxiv", n: int = 10):
        """Get sample documents for summarization evaluation"""
        if doc_type not in self.datasets["summarization"]:
            return []
        return self.loader.get_sample("summarization", doc_type, n)

    def get_qa_sample(self, qa_type: str = "aiml_concepts_400", n: int = 10):
        """Get sample Q&A pairs for evaluation"""
        if qa_type not in self.datasets["qa"]:
            return []
        return self.loader.get_sample("qa", qa_type, n)

    def get_glossary_terms(self, subject: str | None = None):
        """Get glossary terms, optionally filtered by subject"""
        if "custom" not in self.datasets["glossary"]:
            return []

        terms = self.datasets["glossary"]["custom"].get("terms", [])

        if subject:
            terms = [
                t for t in terms
                if t.get("subject", "").lower() == subject.lower()
            ]

        return terms

    def export_dataset_info(self, filename: str = "datasets/dataset_info.json"):
        """Export comprehensive dataset information"""
        stats = self.loader.get_stats()

        info = {
            "statistics": stats,
            "available_datasets": {
                "summarization": list(self.datasets["summarization"].keys()),
                "qa": list(self.datasets["qa"].keys()),
                "glossary": list(self.datasets["glossary"].keys()),
            },
            "usage": {
                "summarization": "Use for evaluating TL;DR, medium, and detailed summaries.",
                "qa": "Use for evaluating question generation and answer quality.",
                "glossary": "Use for evaluating term extraction and definition quality.",
            },
        }

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        return info


# Quick helpers

def load_evaluation_datasets() -> EvaluationDatasetManager:
    return EvaluationDatasetManager()


def get_dataset_stats() -> Dict:
    loader = DatasetLoader()
    loader.load_all()
    return loader.get_stats()


def load_custom_glossary():
    path = "datasets/custom/custom_glossary.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    import json as _json

    print("=" * 60)
    print(" DATASET LOADER TEST")
    print("=" * 60)

    manager = EvaluationDatasetManager()
    stats = manager.loader.get_stats()
    print("\n Dataset Statistics:")
    print(_json.dumps(stats, indent=2))
    print("\n Dataset loader working correctly!")
