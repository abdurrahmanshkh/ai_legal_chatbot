# iLegalBot

**A conversational AI assistant for legal services and resources**

## 📋 Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Architecture](#architecture)
* [Dataset & Training](#dataset--training)
* [Setup & Installation](#setup--installation)
* [Usage](#usage)
* [Customization](#customization)
* [Deployment](#deployment)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

---

## Overview

iLegalBot is a smart, AI-driven chatbot designed to streamline legal service access for users. It assists with lawyer registration, provides answers to legal documentation queries, explains fines, and more. Law students and practicing lawyers can also leverage iLegalBot to retrieve structured legal information on statutes, case law, and regulatory frameworks.

---

## Features

* **User-Focused Interactions**: Ask general legal questions, check documentation requirements, and learn about fines or penalties.
* **Lawyer Registration**: Simplified workflow for lawyers to register and manage profiles.
* **Dual-Model Support**: Choose between Google Gemini 2.0 Flash and OpenAI GPT-4o Mini for responses.
* **Fast Retrieval**: Embedding-based search powered by FAISS index for sub-second retrieval of relevant legal texts.
* **Extensive Knowledge Base**:

  * Over 100,000 FAQ entries on Indian law
  * Core legal texts: BNS, BNSS, BSA, IPC, CrPC, IEA, Constitution of India, and more.
* **Streamlit UI**: User-friendly, interactive frontend with real-time chat interface.

---

## Architecture

1. **Embedding & Indexing**

   * Embedded legal documents and FAQs using Hugging Face embedding models.
   * Built a FAISS index for vector similarity search.
2. **Retriever**

   * Created a retriever object to fetch top-k relevant passages based on user queries.
3. **Prompt Template**

   * Defined templated prompts to provide context to the LLMs before generating responses.
4. **LLM Pipeline**

   * Integrated two backend models:

     * Google Gemini 2.0 Flash
     * OpenAI GPT-4o Mini
   * Users can select their preferred model at runtime.
5. **Backend Server**

   * Python-based API server handling chat sessions, retrieval, and model calls.
6. **Frontend UI**

   * Streamlit app for chat interface, model selection, and conversation history.

---

## Dataset & Training

* **FAQ Collection**: Curated and cleaned over 100,000 question-answer pairs on Indian legal topics.
* **Legal Documents**: Parsed and embedded statutes and regulations, including:

  * Bare Acts: BNS, BNSS, BSA
  * Indian Penal Code (IPC)
  * Criminal Procedure Code (CrPC)
  * Information Technology Act (IEA)
  * Constitution of India
* **Training**:

  * Generated embeddings for all texts.
  * Indexed vectors in FAISS for efficient similarity search.

---

## Setup & Installation

### Prerequisites

* Python 3.8+
* `pip` package manager
* (Optional) API keys for OpenAI and Google Cloud

### Clone the Repository

```bash
git clone https://github.com/abdurrahmanshkh/legal_chatbot_backend.git
cd legal_chatbot_backend
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configuration

1. Copy `.env.example` to `.env` and set the following variables:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_PROJECT_ID=your_google_project_id
   GEMINI_MODEL_NAME=gemini-2.0-flash
   ```
2. Ensure you have access credentials for any cloud services.

---

## Usage

### Launch Backend Server

```bash
python server.py
```

The server will start on `http://localhost:8000` by default.

### Start Streamlit Frontend

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to interact with iLegalBot.

---

## Customization

* **Adding New Documents**:

  1. Place new `.txt` or `.pdf` files into the `data/` directory.
  2. Run the indexing script:

     ```bash
     python scripts/build_index.py
     ```
* **Adjusting Prompt Templates**:

  * Edit `templates/chat_prompt.txt` to tweak the system instructions or conversation flow.
* **Switching Models**:

  * In the UI, use the model selector dropdown to switch between Gemini 2.0 and GPT-4o Mini.

---

## Deployment

We recommend deploying the backend on a cloud VM (e.g., Google Compute Engine) and the Streamlit app via Streamlit Cloud or Docker.

### Docker (Optional)

1. Build the image:

   ```bash
   docker build -t ilegalbot .
   ```
2. Run the container:

   ```bash
   docker run -p 8000:8000 -p 8501:8501 --env-file .env ilegalbot
   ```

---

## Contributing

Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request on GitHub.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

**Project Maintainer**: Abdur Rehman Shaikh ([as31@somaiya.edu](mailto:as31@somaiya.edu))

Feel free to reach out for any questions or partnership opportunities.")
