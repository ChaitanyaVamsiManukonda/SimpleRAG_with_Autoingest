# SimpleRAG_with_Autoingest (Document Assistant)

A powerful, local Retrieval-Augmented Generation (RAG) system that uses Anthropic's Claude and local embeddings to answer questions about your documents.

## Features

- **Auto-Ingestion**: Automatically processes new PDF and TXT files placed in the `documents/` folder.
- **Smart Retrieval**: Uses semantic search with `sentence-transformers` and FAISS to find the most relevant context.
- **Powered by Claude**: Generates accurate, context-aware answers using Anthropic's Claude 4.5 Haiku model.
- **Persistent Storage**: Saves processed embeddings to a local vector database so you don't have to re-index everything.

## Setup

### 1. Prerequisites
- Python 3.8 or higher
- An Anthropic API Key (Get one [here](https://console.anthropic.com/))

### 2. Installation

1.  **Clone or open the project.**

2.  **Create and activate a virtual environment:**
    ```powershell
    # Windows
    python -m venv venv
    .\venv\Scripts\Activate
    ```

3.  **Install dependencies:**
    ```powershell
    pip install -r requirements.txt
    ```

4.  **Configure API Key:**
    - Create a file named `.env` in the project root.
    - Add your API key:
      ```
      ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
      ```

## Usage

### 1. Add Documents
Simply drop your `.pdf` or `.txt` files into the `documents/` folder. The system will automatically ingest them the next time you run a query.

### 2. Ask Questions
Run the `query` command to ask questions about your documents:

```powershell
python main.py query --query "What are the key points in the monthly report?"
```

### 3. Manual Ingestion (Optional)
If you want to ingest a specific file from a different location:

```powershell
python main.py ingest --document "C:\path\to\your\file.pdf"
```

## Advanced Configuration

You can tune the RAG performance with additional flags:

- `--top-k 5`: Number of chunks to send to Claude (default: 5).
- `--fetch-k 20`: Number of candidates to fetch before re-ranking (default: 20).
- `--index-type ivf`: Use 'ivf' index for faster search on very large datasets (default: 'flat').
- `--model`: Specify a different Claude model version.

## Project Structure

- `main.py`: Entry point and CLI interface.
- `src/`: Core logic (Document processing, Vector store, Pipeline).
- `documents/`: Default folder for input files.
- `vector_db/`: Storage for the vector embeddings.
