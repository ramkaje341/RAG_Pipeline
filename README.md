# 🎬 Movie Recommendation System – RAG Pipeline

A sophisticated **Retrieval-Augmented Generation (RAG)** solution designed to provide personalized movie recommendations and detailed explanations. By combining **FAISS** for high-speed similarity search and **Ollama** for local LLM generation, this system goes beyond basic filtering to offer context-aware cinematic insights.

---

## 📊 Project Overview

This project addresses the "analysis paralysis" of choosing a movie by:
* **Semantic Search:** Understanding user intent (e.g., "movies about space travel and isolation") rather than just matching keywords.
* **Contextual Explanations:** Using an LLM to explain *why* a specific movie was recommended based on its plot and metadata.
* **Local Privacy:** Running the entire inference pipeline locally using Ollama and FAISS.

---

## 🏗️ System Architecture

The pipeline follows a modular RAG workflow:

1.  **Data Preprocessing:** Cleaning the TMDB 5000 dataset using **Pandas**.
2.  **Embedding Generation:** Converting movie overviews and genres into high-dimensional vectors using **SentenceTransformers**.
3.  **Vector Indexing:** Storing embeddings in **FAISS** for optimized, low-latency similarity retrieval.
4.  **Augmented Generation:** Injecting retrieved movie data into a structured prompt for **Ollama (Phi/Mistral)** to generate a conversational response.

---

## 🛠️ Tech Stack

### ⚙️ Backend / AI Layer
* **Python:** Core programming language.
* **LangChain:** Framework for prompt chaining, retrieval management, and LLM integration.
* **FAISS (Facebook AI Similarity Search):** High-performance vector database for similarity search.
* **SentenceTransformers:** Generates state-of-the-art text embeddings.

### 🤖 LLM Layer
* **Ollama:** Facilitates running large language models locally.
* **Phi / Mistral:** Local models used for generating natural language recommendations and summaries.

### 🌐 Frontend / UI
* **Streamlit:** Interactive web interface for user queries and displaying results.

### 🗄️ Data Layer
* **Dataset:** TMDB 5000 Movies (CSV).
* **Pandas:** Used for data loading, cleaning, and feature engineering.

---

## 📂 Repository Structure

```text
├── app.py                # Streamlit UI and main logic
├── vector_store.py       # Script for FAISS indexing and embeddings
├── preprocess.py         # Data cleaning and Pandas logic
├── requirements.txt      # Project dependencies
├── tmdb_5000_movies.csv  # Raw movie dataset
└── faiss_index/          # Local directory for stored vector indices

---

## 🚀 Getting Started

### 1. Prerequisites
* **Python 3.9+**
* **Ollama** installed and running locally.
* Pull your preferred model:  
  ```bash
  ollama pull phi
---

## Clone the repository
git clone [https://github.com/ramkaje341/RAG_Pipline.git](https://github.com/ramkaje341/RAG_Pipline.git)
cd RAG_Pipline
---

## Install dependencies
pip install -r requirements.txt

---
# First, generate the vector index from the dataset
python vector_store.py

# Launch the Streamlit dashboard
streamlit run app.py



