

# 🧑‍⚖️ Legal Document Understanding AI Assistant (RAG-Based)

## 📌 Overview

The **Legal Document Understanding AI Assistant** is an intelligent, retrieval-augmented system designed to **analyze legal situations described by users** and provide **clear, structured, and context-aware legal insights**.

This project is built to help individuals—especially **victims and non-legal professionals**—understand complex legal documents by mapping real-world situations to **relevant legal sections and punishments** using **LLMs + RAG**.

The system leverages:

* **FAISS Vector Search**
* **Ollama-powered LLMs**
* **Advanced Prompt Engineering**
* **Streamlit UI**

---

## ✨ Key Features

### 🔍 1. Legal Document Understanding

* Interprets complex legal sections
* Identifies applicable laws for a given situation
* Explains **why** a section applies
* Clearly states **punishments and legal consequences**

---

### 📚 2. Section-Based Chunking & Vector Storage

* Legal documents are split into **meaningful sections**
* Each section is embedded and stored in a **FAISS vector database**
* This improves **retrieval accuracy** and avoids irrelevant context

---

### 🧠 3. Retrieval-Augmented Generation (RAG)

When a user submits a query:

1. Relevant legal sections are retrieved from FAISS
2. Retrieved sections are injected into the LLM prompt
3. The model generates **grounded and explainable outputs**

---

### 🧪 4. Advanced Prompt Engineering

The system uses:

* **Meta prompting** → controls tone, structure, and role
* **One-shot prompting** → ensures consistent legal-style responses

This results in **highly structured and reliable legal answers**.

---

### 🧾 5. Structured Output

The AI produces:

* Applicable legal sections
* Explanation for each section
* Punishment details
* Contextual justification

This makes the system suitable for:

* Legal dashboards
* Advisory tools
* AI-powered assistants

---

## 🏗️ Architecture Overview

```text
User Query
   ↓
Situation Analysis
   ↓
FAISS Vector Search
   ↓
Relevant Legal Sections Retrieved
   ↓
LLM Reasoning (Meta + One-Shot Prompting)
   ↓
Structured Legal Output
```

---

## 🗂️ Project Structure

```text
law-rag-assistant/
│
├── app.py                         # Streamlit application
├── requirements.txt               # Dependencies
├── bns.json                       # BNS legal sections
├── bns_to_ipc.json                # BNS → IPC mapping
├── bns_vector_db_up_oct11/        # FAISS vector database
│   ├── index.faiss
│   └── index.pkl
└── README.md
```

---

## ⚙️ How to Run the Project (Step-by-Step)

### ✅ 1. Clone the Repository

```bash
git clone <your-repo-url>
cd law-rag-assistant
```

---

### ✅ 2. Create & Activate Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### ✅ 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> If FAISS fails on Windows:

```bash
pip install faiss-cpu --no-cache-dir
```

---

### ✅ 4. Install Ollama (Required)

Ollama is used to run local LLMs and embeddings.

🔗 Download: [https://ollama.com/download](https://ollama.com/download)

After installation, pull required models:

```bash
ollama pull nomic-embed-text
ollama pull deepseek-r1:1.5b
```

Start the Ollama server:

```bash
ollama serve
```

Ollama must be running at:

```
http://localhost:11434
```

---

### ✅ 5. Run the Streamlit App

```bash
streamlit run app.py
```

You will see:

```text
Local URL: http://localhost:8501
```

Open it in your browser 🎉

---

## 🧪 How the App Works (User Flow)

1. User enters a **legal situation** in plain language
2. System retrieves **relevant legal sections** from FAISS
3. LLM analyzes retrieved sections
4. AI generates:

   * Applicable sections
   * Explanation
   * Punishment details
5. Output is displayed in a clean, readable format

---

## 🎯 Problem This Solves

* Legal documents are complex and inaccessible
* Victims often don’t know which laws apply
* Legal consultation is expensive and slow

✅ This system:

* Simplifies legal language
* Provides instant insights
* Improves legal awareness
* Makes legal knowledge accessible

---

## 🚀 Why This Project Stands Out

* Complete **end-to-end RAG pipeline**
* Professional **section-based chunking**
* Advanced **prompt engineering experimentation**
* Real-world legal applicability
* Fully local & privacy-preserving (Ollama)

---

## ⚠️ Disclaimer

This project is for **educational and informational purposes only**.
It does **not replace professional legal advice**.

---

## 📌 Future Enhancements

* Multi-language legal support
* Court-case citation retrieval
* PDF upload & document parsing
* User feedback loop
* Cloud deployment

---
https://github.com/user-attachments/assets/cecd2b30-94e8-4556-ac82-660f3d1d055e



