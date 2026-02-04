"""
MemFlow Vault Indexer
Sub-Task 2.2: Create initial indexing of vault with local embeddings

Reads all .md files from the Obsidian vault, creates vector embeddings,
and persists the index for fast retrieval.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import frontmatter
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Document,
    Settings,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
VAULT_PATH = os.environ.get(
    "VAULT_PATH",
    "/mnt/c/Users/gurba/Documents/Second Brain"
)
INDEX_STORAGE = os.environ.get(
    "INDEX_STORAGE",
    os.path.join(os.path.dirname(__file__), ".memflow_index")
)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def extract_wikilinks(content: str) -> List[str]:
    """Extract Obsidian-style wikilinks [[Note Name]] from content."""
    pattern = r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]'
    return re.findall(pattern, content)


def extract_metadata(filepath: Path, content: str) -> Dict[str, Any]:
    """Extract metadata from a markdown file including frontmatter and backlinks."""
    metadata = {
        "file_path": str(filepath),
        "file_name": filepath.name,
        "folder": str(filepath.parent.relative_to(VAULT_PATH)) if filepath.is_relative_to(VAULT_PATH) else str(filepath.parent),
        "modified_time": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
        "backlinks": [],
        "wikilinks": extract_wikilinks(content),
    }

    # Parse frontmatter if present
    try:
        post = frontmatter.loads(content)
        if post.metadata:
            metadata["frontmatter"] = post.metadata
            # Extract tags if present
            if "tags" in post.metadata:
                metadata["tags"] = post.metadata["tags"]
    except Exception as e:
        logger.debug(f"No frontmatter in {filepath}: {e}")

    return metadata


def load_documents_with_metadata() -> List[Document]:
    """Load all markdown documents from vault with custom metadata extraction."""
    documents = []
    vault_path = Path(VAULT_PATH)

    if not vault_path.exists():
        raise FileNotFoundError(f"Vault path does not exist: {VAULT_PATH}")

    md_files = list(vault_path.rglob("*.md"))
    logger.info(f"Found {len(md_files)} markdown files in vault")

    for filepath in md_files:
        try:
            content = filepath.read_text(encoding='utf-8')
            metadata = extract_metadata(filepath, content)

            # Create document with content and metadata
            doc = Document(
                text=content,
                metadata=metadata,
                doc_id=str(filepath.relative_to(vault_path)),
            )
            documents.append(doc)

        except Exception as e:
            logger.warning(f"Error reading {filepath}: {e}")
            continue

    # Build backlink index
    backlink_map: Dict[str, List[str]] = {}
    for doc in documents:
        for link in doc.metadata.get("wikilinks", []):
            if link not in backlink_map:
                backlink_map[link] = []
            backlink_map[link].append(doc.metadata["file_name"])

    # Update documents with backlink counts
    for doc in documents:
        file_name_no_ext = Path(doc.metadata["file_name"]).stem
        doc.metadata["backlink_count"] = len(backlink_map.get(file_name_no_ext, []))
        doc.metadata["backlinks"] = backlink_map.get(file_name_no_ext, [])

    return documents


def create_index(documents: List[Document]) -> VectorStoreIndex:
    """Create vector store index from documents using local embeddings."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")

    # Configure embedding model
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        trust_remote_code=True,
    )

    # Set global settings
    Settings.embed_model = embed_model
    Settings.llm = None  # We don't need LLM for indexing
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    logger.info(f"Creating index from {len(documents)} documents...")

    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    return index


def persist_index(index: VectorStoreIndex, storage_path: str = INDEX_STORAGE) -> None:
    """Persist index to disk for later retrieval."""
    logger.info(f"Persisting index to: {storage_path}")
    os.makedirs(storage_path, exist_ok=True)
    index.storage_context.persist(persist_dir=storage_path)
    logger.info("Index persisted successfully")


def load_index(storage_path: str = INDEX_STORAGE) -> Optional[VectorStoreIndex]:
    """Load persisted index from disk."""
    if not os.path.exists(storage_path):
        logger.warning(f"No existing index found at {storage_path}")
        return None

    try:
        logger.info(f"Loading index from: {storage_path}")

        # Configure embedding model for loading
        embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            trust_remote_code=True,
        )
        Settings.embed_model = embed_model
        Settings.llm = None

        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context)
        logger.info("Index loaded successfully")
        return index

    except Exception as e:
        logger.error(f"Error loading index: {e}")
        return None


def index_vault(force_rebuild: bool = False) -> VectorStoreIndex:
    """
    Main function to index the vault.

    Args:
        force_rebuild: If True, rebuild index even if one exists

    Returns:
        VectorStoreIndex ready for querying
    """
    # Try to load existing index first
    if not force_rebuild:
        existing_index = load_index()
        if existing_index:
            logger.info("Using existing index")
            return existing_index

    # Build new index
    logger.info("Building new index...")
    documents = load_documents_with_metadata()
    index = create_index(documents)
    persist_index(index)

    logger.info(f"✅ Indexed {len(documents)} documents")
    return index


def update_documents(filepaths: List[str], index: VectorStoreIndex) -> VectorStoreIndex:
    """
    Incrementally update specific documents in the index.

    Args:
        filepaths: List of file paths that were modified
        index: Existing index to update

    Returns:
        Updated index
    """
    vault_path = Path(VAULT_PATH)

    for filepath_str in filepaths:
        filepath = Path(filepath_str)

        if not filepath.exists():
            # File was deleted - remove from index
            try:
                doc_id = str(filepath.relative_to(vault_path))
                index.delete_ref_doc(doc_id)
                logger.info(f"Removed deleted document: {doc_id}")
            except Exception as e:
                logger.warning(f"Could not remove {filepath}: {e}")
            continue

        try:
            # Read updated content
            content = filepath.read_text(encoding='utf-8')
            metadata = extract_metadata(filepath, content)
            doc_id = str(filepath.relative_to(vault_path))

            # Create new document
            doc = Document(
                text=content,
                metadata=metadata,
                doc_id=doc_id,
            )

            # Update in index (delete old, insert new)
            try:
                index.delete_ref_doc(doc_id)
            except Exception:
                pass  # Document might not exist yet

            index.insert(doc)
            logger.info(f"Updated document: {doc_id}")

        except Exception as e:
            logger.error(f"Error updating {filepath}: {e}")

    # Persist updated index
    persist_index(index)
    return index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index Obsidian vault for MemFlow")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild index even if one exists"
    )
    parser.add_argument(
        "--vault",
        type=str,
        default=VAULT_PATH,
        help="Path to Obsidian vault"
    )
    args = parser.parse_args()

    if args.vault:
        VAULT_PATH = args.vault

    index = index_vault(force_rebuild=args.rebuild)

    # Quick test query
    query_engine = index.as_query_engine()
    print("\n🔍 Test query: 'job search priorities'")
    response = query_engine.query("What are my job search priorities?")
    print(f"Response: {response}")
