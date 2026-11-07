# Multimodal RAG

An AI-powered Multimodal Retrieval-Augmented Generation (RAG) application that answers questions about PDF documents and images using **NVIDIA Multimodal Embeddings, ChromaDB, LangChain and Vision LLMs**.

## Live Demo

[Try NVIDIA Multimodal RAG](https://YOUR-APP-NAME.streamlit.app/)

## Features

- Upload PDF documents and images (PNG, JPG, JPEG, WEBP).
- Converts PDF pages into images using PyMuPDF.
- Generates image embeddings using NVIDIA Nemotron.
- Stores and retrieves embeddings using ChromaDB.
- Performs similarity search to retrieve the most relevant document pages.
- Uses a Vision LLM to understand text, charts, tables and diagrams.
- Generates context-aware answers using retrieved page images.
- Simple and interactive interface built with Streamlit.

## Tech Stack

- **Python** – Core programming language
- **Streamlit** – Web application interface
- **LangChain** – LLM and embedding integration
- **NVIDIA API** – Multimodal embeddings and Vision LLM inference
- **ChromaDB** – Vector database
- **PyMuPDF** – PDF processing
- **Pillow** – Image processing
- **OpenAI SDK** – NVIDIA API integration

## Local Setup

**1. Clone the repository**

```bash
git clone https://github.com/bupendra-d/Multimodal-RAG.git
cd Multimodal-RAG
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file in the project directory:

```env
NVIDIA_API_KEY=your_nvidia_api_key
```

Get your API key from:

- [NVIDIA API Catalog](https://build.nvidia.com/)

**5. Run the application**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Deployment

The application is deployed using Streamlit Community Cloud.

Configure the following secret under your Streamlit application's **Settings → Secrets**:

```toml
NVIDIA_API_KEY = "your_nvidia_api_key"
```

The API key is securely managed through environment variables and Streamlit Secrets and is not exposed in the application interface.

## Note

- PDF documents are processed page by page, and each page is converted into an image for visual embedding.
- Retrieval uses text-to-image similarity search to identify relevant document pages.
- Large documents may require additional processing time and NVIDIA API usage.
- ChromaDB uses an in-memory collection in this implementation. Uploaded documents need to be processed again when the application session or runtime is reset.