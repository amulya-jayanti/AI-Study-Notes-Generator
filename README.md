# AI Study Notes Generator for AI/ML/Data Science

An interactive **Streamlit-based NLP application** that generates:

* ✔ Chapter Summaries (short / medium / detailed)
* ✔ Flashcards & Q&A
* ✔ Glossary terms (LLM + KeyBERT + spaCy)
* ✔ Topic clusters using UMAP + HDBSCAN
* ✔ Evaluation metrics (ROUGE, glossary accuracy, etc.)

Users upload a **PDF chapter**, and the app automatically:

1. Reads and chunks the text
2. Builds embeddings with FAISS
3. Generates summaries and study material
4. Extracts glossary terms
5. Clusters topics using semantic embeddings
6. Enables evaluation against ground-truth datasets

---

## 📁 Project Structure

```
app.py                 → Streamlit application (entry point)
nlp_pipeline.py        → Summaries, glossary extraction, topic clustering
app_extensions.py      → Streamlit UI components (tabs)
dataset_loader.py      → Loads evaluation datasets
build_glossary.py      → Generates 115-term custom glossary
requirements.txt       → Python dependencies
dataset_preparation.ipynb → run this to prepare the datasets
```
---
Note: Evaluation datasets are not included in the repository. To recreate them, run dataset_preparation.ipynb, which automatically builds the full 1265-item evaluation corpus (ML-ArXiv papers, StackOverflow ML Q&A, PubMedQA, interview Q&A, and the 115-term glossary).

## 🛠 Prerequisites

* **Python 3.9+**
* **pip**
* **OpenAI API Key**

Recommended:

* A virtual environment (`venv`, `conda`, etc.)

---

## Installation & Setup

### **1. Clone the repository**

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### **2. Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
# OR
.\.venv\Scripts\activate      # Windows
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

### **4. Add your API key**

Create a file named `.env` in the project root:

```
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

The app uses:

* GPT-4o / GPT-4o-mini
* OpenAI embeddings for vector search
* SentenceTransformers (`MiniLM-L6-v2`) for clustering

---

## Run the Application

Run Streamlit from the project root:

```bash
streamlit run app.py
```
---

## How to Use the App

### **1. Upload a PDF chapter**

* Go to the *Home / Upload* tab in the UI
* The app will:

  * Load PDF with **PyPDFLoader**
  * Chunk text using **RecursiveCharacterTextSplitter**
  * Build vector store (OpenAI Embeddings + FAISS)

### **2. Generate Summaries**

Navigate to **Summaries** tab:

* Choose **Short / Medium / Detailed**
* Uses **map-reduce summarization** with GPT-4o / GPT-4o-mini

### **3. Glossary Extraction**

The glossary pipeline:

1. Extract candidate terms via **KeyBERT**
2. Add entities using **spaCy NER**
3. Generate definitions using LLM
4. Compare with **115-term ground-truth glossary**

### **4. Flashcards & Q&A**

* Auto-generated using GPT-4o-mini
* Includes conceptual Q&A and recall prompts

### **5. Topic Clustering**

Go to **Topics / Clusters** tab:

* Embeddings: SentenceTransformers (MiniLM-L6-v2)
* Dimensionality reduction: **UMAP**
* Clustering: **HDBSCAN**
* Cluster summaries: GPT-4o-mini

### **6. Evaluation**

In the **Evaluation** tab:

* ROUGE scoring for summarization
* Glossary accuracy
* Sample retrieval QA

---

## Tech Stack

| Component     | Tools Used                           |
| ------------- | ------------------------------------ |
| App Framework | Streamlit                            |
| LLMs          | GPT-4o / GPT-4o-mini                 |
| Embeddings    | OpenAIEmbeddings + FAISS             |
| NLP           | spaCy, KeyBERT, SentenceTransformers |
| Clustering    | UMAP + HDBSCAN                       |
| Evaluation    | ROUGE, glossary metrics              |

---

