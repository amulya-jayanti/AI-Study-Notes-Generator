# AI Study Notes Generator for AI/ML/Data Science

An interactive **Streamlit-based NLP application** that generates:
- ✔ Chapter Summaries (short/medium/detailed)
- ✔ Flashcards & Q&A
- ✔ Glossary terms (LLM + KeyBERT + spaCy)
- ✔ Topic clusters using UMAP + HDBSCAN
- ✔ Evaluation metrics (ROUGE, glossary accuracy, etc.)

Users upload a **PDF chapter**, and the app automatically:
1. Reads and chunks the text  
2. Builds embeddings + FAISS vector store  
3. Generates summaries and study material  
4. Extracts glossary terms  
5. Clusters topics using semantic embeddings  
6. Allows evaluation against ground-truth datasets  

---
app.py → Streamlit application
nlp_pipeline.py → Summaries, glossary extraction, topic clustering
app_extensions.py → Streamlit UI components (tabs)
dataset_loader.py → Loads evaluation datasets
build_glossary.py → Generates 115-term custom glossary

## Features

### **1. PDF Processing**
- Uses **PyPDFLoader** to read full PDF chapters
- Splits content using **RecursiveCharacterTextSplitter**
- Stores embeddings using **OpenAI Embeddings + FAISS**

### **2. Summarization**
- Multi-level summaries (short/medium/detailed)
- Uses **GPT-4o / GPT-4o-mini** for high-quality summarization
- Map-reduce summarization framework for large texts

### **3. Glossary Extraction**
Three-stage glossary pipeline:
1. **KeyBERT** → keyword/keyphrase candidates  
2. **spaCy NER** → named entities  
3. **LLM-generated definitions**  
4. Comparison with a **115-term ground-truth glossary**

### **4. Topic Clustering**
- Converts paragraphs into embeddings using **SentenceTransformers (MiniLM-L6-v2)**
- Reduces dimensions with **UMAP**
- Clusters using **HDBSCAN**
- Generates cluster summaries via GPT-4o-mini

### **5. Evaluation Module**
Includes:
- ROUGE scoring for summarization quality
- Glossary accuracy evaluation
- Sample retrieval QA evaluation

### **6. Modular Architecture**
Main modules:
