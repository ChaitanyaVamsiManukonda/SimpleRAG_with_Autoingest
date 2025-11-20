# main.py
import argparse
import os
import json
from pathlib import Path
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.query_processor import QueryProcessor
from src.response_generator import ResponseGenerator
from src.pipeline import RAGPipeline


def load_env_file(filepath=".env"):
    """Simple .env loader to avoid adding dependencies"""
    if os.path.exists(filepath):
        print(f"Loading environment from {filepath}")
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip().strip("'").strip('"')
                    os.environ[key.strip()] = value

# Load .env if it exists
load_env_file()


class DocumentTracker:
    """Tracks which documents have been ingested to avoid re-processing"""

    def __init__(self, tracker_file: str = ".ingested_files.json"):
        self.tracker_file = tracker_file
        self.ingested_files = self._load_tracker()

    def _load_tracker(self) -> dict:
        """Load the list of previously ingested files"""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_tracker(self):
        """Save the ingested files list"""
        with open(self.tracker_file, "w") as f:
            json.dump(self.ingested_files, f, indent=2)

    def is_ingested(self, file_path: str) -> bool:
        """Check if file has already been ingested"""
        abs_path = os.path.abspath(file_path)
        if abs_path in self.ingested_files:
            # Check if file still exists and hasn't been modified
            if os.path.exists(abs_path):
                current_mtime = os.path.getmtime(abs_path)
                stored_mtime = self.ingested_files[abs_path].get("mtime", 0)
                return current_mtime == stored_mtime
        return False

    def mark_as_ingested(self, file_path: str):
        """Mark a file as ingested"""
        abs_path = os.path.abspath(file_path)
        mtime = os.path.getmtime(abs_path)
        self.ingested_files[abs_path] = {
            "mtime": mtime,
            "size": os.path.getsize(abs_path),
            "name": os.path.basename(abs_path),
        }
        self._save_tracker()
        print(f"✓ Marked as ingested: {os.path.basename(abs_path)}")

    def get_new_files(self, folder_path: str, extensions: list = None) -> list:
        """Get list of new files in folder that haven't been ingested"""
        if extensions is None:
            extensions = [".pdf", ".txt"]

        new_files = []
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist")
            return new_files

        for file in Path(folder_path).glob("*"):
            if file.is_file() and file.suffix.lower() in extensions:
                if not self.is_ingested(str(file)):
                    new_files.append(str(file))

        return sorted(new_files)


def auto_ingest_documents(pipeline, tracker, documents_folder: str = "./documents"):
    """Automatically ingest new documents from folder"""
    new_files = tracker.get_new_files(documents_folder)

    if not new_files:
        print(f"✓ No new documents to ingest (checking folder: {documents_folder})")
        return 0

    print(f"\n{'='*70}")
    print(f"Found {len(new_files)} new document(s) to ingest")
    print(f"{'='*70}\n")

    ingested_count = 0
    for file_path in new_files:
        try:
            print(f"Ingesting: {os.path.basename(file_path)}")
            pipeline.ingest_document(file_path)
            tracker.mark_as_ingested(file_path)
            ingested_count += 1
            print()
        except Exception as e:
            print(f"✗ Error ingesting {file_path}: {str(e)}\n")

    if ingested_count > 0:
        print(f"{'='*70}")
        print(f"Successfully ingested {ingested_count} document(s)")
        print(f"{'='*70}\n")

    return ingested_count


def main():
    parser = argparse.ArgumentParser(
        description="World-Class RAG System with Auto-Ingestion"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("--query", "-q", required=True, help="Query to process")

    # Ingest command (manual)
    ingest_parser = subparsers.add_parser("ingest", help="Manually ingest a document")
    ingest_parser.add_argument(
        "--document", "-d", required=True, help="Path to document"
    )
    ingest_parser.add_argument("--metadata", "-m", help="JSON metadata for document")

    # Common arguments
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Claude model to use (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model for document processing",
    )
    parser.add_argument(
        "--vector-db", default="./vector_db", help="Path to vector database"
    )
    parser.add_argument(
        "--documents-folder",
        default="./documents",
        help="Folder to auto-ingest documents from",
    )
    parser.add_argument(
        "--api-key", help="Claude API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--skip-auto-ingest",
        action="store_true",
        help="Skip auto-ingestion on startup",
    )

    # Advanced RAG tuning parameters
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final chunks to pass to Claude (default: 5)",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=20,
        help="Number of candidates to fetch before re-ranking (default: 20)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional absolute minimum similarity score (default: dynamic threshold)",
    )
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf"],
        default="flat",
        help="FAISS index type: flat (exact) or ivf (approximate, faster for large datasets)",
    )

    args = parser.parse_args()

    # Create documents folder if it doesn't exist
    if not os.path.exists(args.documents_folder):
        os.makedirs(args.documents_folder)
        print(f"✓ Created documents folder: {args.documents_folder}")

    # Initialize tracker and pipeline
    tracker = DocumentTracker()
    pipeline = RAGPipeline(
        model_name=args.embedding_model,
        vector_db_path=args.vector_db,
        api_key=args.api_key,
        claude_model=args.model,
        index_type=args.index_type,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        min_score=args.min_score,
    )

    # Auto-ingest new documents (unless skipped)
    if not args.skip_auto_ingest:
        auto_ingest_documents(pipeline, tracker, args.documents_folder)

    # Handle commands
    if args.command == "query":
        result = pipeline.process_query(args.query)
        print("\n" + "=" * 70)
        print("=== Response ===")
        print("=" * 70)
        print(result["response"])
        print("\n" + "=" * 70)
        print("=== Retrieved Chunks ===")
        print("=" * 70)
        for i, chunk in enumerate(result["retrieved_chunks"]):
            print(f"\nChunk {i+1} (Score: {chunk['score']:.4f}):")
            print(f"Source: {chunk['metadata'].get('filename', 'Unknown')}")
            print("-" * 70)
            text_preview = (
                chunk["text"][:300] + "..."
                if len(chunk["text"]) > 300
                else chunk["text"]
            )
            print(text_preview)

    elif args.command == "ingest":
        # Manual ingest
        metadata = None
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
            except:
                print(f"Error parsing metadata JSON: {args.metadata}")
                return

        pipeline.ingest_document(args.document, metadata)
        tracker.mark_as_ingested(args.document)

    else:
        print("\n" + "=" * 70)
        print("World-Class RAG System with Auto-Ingestion")
        print("=" * 70)
        print("\nUSAGE:")
        print('  Query: python main.py query --query "your question"')
        print("  Manual ingest: python main.py ingest --document path/to/file.pdf")
        print("\nAUTO-INGEST FEATURE:")
        print(f"  Place PDF or TXT files in: {args.documents_folder}/")
        print("  They will be automatically ingested when main.py is run")
        print("  Only new/modified files will be processed")
        print("\nOPTIONS:")
        print("  --skip-auto-ingest: Skip auto-ingestion on startup")
        print("  --documents-folder: Change documents folder (default: ./documents)")
        print("\nADVANCED RAG TUNING:")
        print("  --top-k N: Final chunks to use for generation (default: 5)")
        print("  --fetch-k N: Candidates to retrieve before re-ranking (default: 20)")
        print("  --min-score X: Minimum similarity score (default: dynamic)")
        print("  --index-type: 'flat' (exact) or 'ivf' (fast, for large datasets)")
        print("\n" + "=" * 70 + "\n")
        parser.print_help()


if __name__ == "__main__":
    main()
