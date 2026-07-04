# AI Research Assistant (Local RAG Pipeline)

A specialized Retrieval-Augmented Generation (RAG) system built from scratch using Python. This application parses complex research documents (like quantum physics textbooks), converts semantic meaning into mathematical vector embeddings, indexes them in a local vector database, and queries them using advanced language models to provide grounded, citation-accurate responses.

## 🚀 Key Features
* **Document Parsing:** Drops heavy styling structures and extracts raw content from `.pdf` formats.
* **Contextual Chunking:** Slices massive texts into uniform 1,000-character windows with a 200-character overlapping buffer to maintain concept continuity.
* **Semantic Embedding Index:** Vectors are generated through Google's `gemini-embedding-2` model.
* **Local Persistent Storage:** Uses `ChromaDB` as an on-disk vector storage layer to preserve indexed documents without cloud-hosting overhead.
* **Grounded Queries:** Restricts `gemini-2.5-flash` synthesis to explicitly retrieved context blocks to eliminate model hallucinations.

## 🛠️ Tech Stack
* **Language:** Python
* **AI SDK:** Google GenAI SDK
* **Vector Database:** ChromaDB
* **Data Processing:** PyPDF