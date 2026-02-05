"""
MemFlow Query Engine
Sub-Task 2.3: Context retrieval using Relationships.md for better context

Implements hybrid retrieval combining:
1. Vector similarity search (semantic)
2. Relationship-guided retrieval (explicit connections from Relationships.md)
3. Metadata boosting (recent files, high backlinks)
"""

import os

# Ensure HuggingFace runs offline in restricted environments
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from index_vault import (
    load_index,
    VAULT_PATH,
    INDEX_STORAGE,
    EMBEDDING_MODEL,
    build_exclude_paths,
    get_memflow_config,
    is_excluded_path,
    is_included_path,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Relationship retrieval limits (tune via env vars)
RELATIONSHIP_DIR_LIMIT = int(os.environ.get("RELATIONSHIP_DIR_LIMIT", "0"))
RELATIONSHIP_TOTAL_LIMIT = int(os.environ.get("RELATIONSHIP_TOTAL_LIMIT", "0"))
MEMFLOW_PROFILE = os.environ.get("MEMFLOW_PROFILE", "false").lower() == "true"

# Path to Relationships.md
DEFAULT_RELATIONSHIPS_PATH = os.environ.get(
    "RELATIONSHIPS_PATH",
    os.path.join(VAULT_PATH, "Relationships.md"),
)
LEGACY_RELATIONSHIPS_PATH = os.path.join(
    VAULT_PATH, "Projects", "Personal", "MemFlow", "Relationships.md"
)
RELATIONSHIPS_PATH = DEFAULT_RELATIONSHIPS_PATH


@dataclass
class UseCase:
    """Represents a use case from Relationships.md"""
    name: str
    triggers: List[str]
    relevant_files: List[str]
    keywords: List[str]
    answer_structure: List[str]


@dataclass
class ContextResult:
    """Result from context retrieval"""
    context: str
    sources: List[str]
    confidence: float
    use_case: Optional[str] = None


class RelationshipParser:
    """Parses Relationships.md to extract use cases and their mappings."""

    def __init__(self, relationships_path: str = RELATIONSHIPS_PATH):
        self.relationships_path = relationships_path
        self.use_cases: List[UseCase] = []
        self._resolve_relationships_path()
        self._parse()

    def _resolve_relationships_path(self) -> None:
        """Resolve Relationships.md path with a legacy fallback."""
        if os.path.exists(self.relationships_path):
            return

        if (
            self.relationships_path == DEFAULT_RELATIONSHIPS_PATH
            and os.path.exists(LEGACY_RELATIONSHIPS_PATH)
        ):
            logger.info(
                f"Relationships.md not found at vault root. "
                f"Falling back to legacy path: {LEGACY_RELATIONSHIPS_PATH}"
            )
            self.relationships_path = LEGACY_RELATIONSHIPS_PATH

    def _parse(self) -> None:
        """Parse the Relationships.md file to extract use cases."""
        if not os.path.exists(self.relationships_path):
            logger.warning(f"Relationships.md not found at {self.relationships_path}")
            return

        try:
            content = Path(self.relationships_path).read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Error reading Relationships.md: {e}")
            return

        # Split by USE CASE sections
        use_case_pattern = r'## USE CASE \d+: (.+?)(?=## USE CASE|\Z|## CRITICAL)'
        matches = re.findall(use_case_pattern, content, re.DOTALL)

        for i, match in enumerate(matches):
            # Extract use case name from the section
            lines = content.split(f"## USE CASE {i+1}:")[1].split("## USE CASE")[0] if f"## USE CASE {i+1}:" in content else ""

            use_case = self._parse_use_case(lines if lines else match)
            if use_case:
                self.use_cases.append(use_case)

        logger.info(f"Parsed {len(self.use_cases)} use cases from Relationships.md")

    def _parse_use_case(self, section: str) -> Optional[UseCase]:
        """Parse a single use case section."""
        # Extract name (first line after the header)
        name_match = re.search(r'^(.+?)$', section.strip(), re.MULTILINE)
        name = name_match.group(1).strip() if name_match else "Unknown"

        # Extract triggers from TRIGGER section
        trigger_match = re.search(r'\*\*TRIGGER:\*\*\s*(.+?)(?=\*\*RELEVANT|\Z)', section, re.DOTALL)
        triggers = []
        if trigger_match:
            trigger_text = trigger_match.group(1)
            # Extract quoted phrases
            triggers = re.findall(r'"([^"]+)"', trigger_text)
            # Also extract patterns like "asks about X"
            triggers.extend(re.findall(r'asks about ([^,\n"]+)', trigger_text))

        # Extract relevant files from RELEVANT CONTEXT section
        relevant_files = []
        files_match = re.search(r'\*\*RELEVANT CONTEXT:\*\*(.+?)(?=\*\*RELATIONSHIP|\Z)', section, re.DOTALL)
        if files_match:
            # Extract paths in backticks
            relevant_files = re.findall(r'`([^`]+\.md)`|`([^`]+/)`', files_match.group(1))
            relevant_files = [f[0] or f[1] for f in relevant_files]

        # Extract keywords from the section
        keywords = []
        # Look for key terms in the section
        keyword_patterns = [
            r'March 2026',
            r'Solutions Engineer',
            r'H1B',
            r'positioning',
            r'Enterprise SaaS',
            r'AI security',
            r'LinkedIn',
            r'portfolio',
            r'technical credibility',
        ]
        for pattern in keyword_patterns:
            if re.search(pattern, section, re.IGNORECASE):
                keywords.append(pattern)

        # Extract answer structure
        answer_structure = []
        answer_match = re.search(r'\*\*ANSWER STRUCTURE:\*\*(.+?)(?=---|## |\Z)', section, re.DOTALL)
        if answer_match:
            answer_structure = re.findall(r'\d+\.\s+\*\*([^*]+)\*\*', answer_match.group(1))

        return UseCase(
            name=name,
            triggers=triggers,
            relevant_files=relevant_files,
            keywords=keywords,
            answer_structure=answer_structure,
        )

    def match_use_case(self, query: str) -> Optional[UseCase]:
        """Find the best matching use case for a query."""
        query_lower = query.lower()

        # Define trigger patterns for each use case type
        linkedin_triggers = ['linkedin', 'post', 'content', 'write a', 'publish']
        job_triggers = ['apply', 'job', 'company', 'resume', 'application', 'fit', 'role', 'sponsor', 'visa']
        project_triggers = ['build', 'project', 'what should i', 'prioritize', 'next project', 'work on']
        learning_triggers = ['learn', 'learning', 'skill', 'technology', 'framework', 'course', 'cert']

        for use_case in self.use_cases:
            # Check explicit triggers
            for trigger in use_case.triggers:
                if trigger.lower() in query_lower:
                    return use_case

        # Check category-based matching (order matters)
        if any(t in query_lower for t in linkedin_triggers):
            for uc in self.use_cases:
                if 'LinkedIn' in uc.name or 'Content' in uc.name:
                    return uc

        if any(t in query_lower for t in job_triggers):
            for uc in self.use_cases:
                if 'Job' in uc.name or 'Application' in uc.name:
                    return uc

        if any(t in query_lower for t in project_triggers):
            for uc in self.use_cases:
                if 'Project' in uc.name:
                    return uc

        if any(t in query_lower for t in learning_triggers):
            for uc in self.use_cases:
                if 'Learning' in uc.name or 'Skill' in uc.name:
                    return uc

        return None


class MemFlowQueryEngine:
    """
    Main query engine combining vector search with relationship-guided retrieval.
    """

    def __init__(self, index: Optional[VectorStoreIndex] = None):
        """Initialize the query engine."""
        self.index = index or load_index()
        if self.index is None:
            raise ValueError("No index available. Run index_vault.py first.")

        self.relationship_parser = RelationshipParser()
        self._file_cache: Dict[str, Dict[str, Any]] = {}
        self._relationship_cache: Dict[str, List[Tuple[str, float, Dict]]] = {}
        self._config = get_memflow_config()
        self._exclude_paths = build_exclude_paths(Path(VAULT_PATH))

        # Configure embedding model
        embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            trust_remote_code=True,
        )
        Settings.embed_model = embed_model
        Settings.llm = None

        # Warm up embedding model and retriever to reduce first-query latency
        self._warmup()

    def _warmup(self) -> None:
        """Warm up the retriever to avoid cold-start latency."""
        try:
            retriever = self.index.as_retriever(similarity_top_k=1)
            _ = retriever.retrieve("warmup")
        except Exception as e:
            logger.debug(f"Warmup failed: {e}")

    def _read_file_cached(self, file_path: Path) -> Optional[str]:
        """Read a file with simple mtime-based caching."""
        key = str(file_path)
        try:
            mtime = file_path.stat().st_mtime
        except Exception:
            return None

        cached = self._file_cache.get(key)
        if cached and cached.get("mtime") == mtime:
            return cached.get("content")

        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.debug(f"Error reading {file_path}: {e}")
            return None

        self._file_cache[key] = {"content": content, "mtime": mtime}
        return content

    def _iter_markdown_files(self, root: Path) -> List[Path]:
        """Iterate markdown files under root, pruning excluded paths early."""
        md_files: List[Path] = []
        vault_path = Path(VAULT_PATH)

        for dirpath, dirnames, filenames in os.walk(root):
            dir_path = Path(dirpath)

            # Prune excluded directories in-place
            kept_dirs = []
            for d in dirnames:
                candidate = dir_path / d
                if is_excluded_path(candidate, vault_path, self._exclude_paths):
                    continue
                kept_dirs.append(d)
            dirnames[:] = kept_dirs

            for filename in filenames:
                if not filename.endswith(".md"):
                    continue
                file_path = dir_path / filename
                if is_excluded_path(file_path, vault_path, self._exclude_paths):
                    continue
                if not is_included_path(file_path, vault_path, self._config):
                    continue
                md_files.append(file_path)

        return md_files

    def _vector_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Perform vector similarity search.

        Returns:
            List of (content, score, metadata) tuples
        """
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            results.append((
                node.text,
                node.score if hasattr(node, 'score') else 0.5,
                node.metadata if hasattr(node, 'metadata') else {},
            ))

        return results

    def _relationship_search(self, use_case: UseCase) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve documents based on relationship mappings.

        Returns:
            List of (content, score, metadata) tuples
        """
        cache_key = use_case.name
        cached = self._relationship_cache.get(cache_key)
        if cached is not None:
            return cached

        results = []
        total_added = 0
        vault_path = Path(VAULT_PATH)

        for file_pattern in use_case.relevant_files:
            # Handle directory patterns (ending with /)
            if file_pattern.endswith('/'):
                search_path = vault_path / file_pattern.rstrip('/')
                if search_path.exists() and search_path.is_dir():
                    md_files = self._iter_markdown_files(search_path)
                    md_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    if RELATIONSHIP_DIR_LIMIT > 0:
                        md_files = md_files[:RELATIONSHIP_DIR_LIMIT]
                    for md_file in md_files:
                        content = self._read_file_cached(md_file)
                        if not content:
                            continue
                        metadata = {
                            "file_path": str(md_file),
                            "file_name": md_file.name,
                            "source": "relationship",
                        }
                        # Higher score for relationship-based matches
                        results.append((content, 0.8, metadata))
                        total_added += 1
                        if RELATIONSHIP_TOTAL_LIMIT > 0 and total_added >= RELATIONSHIP_TOTAL_LIMIT:
                            break
            else:
                # Direct file reference
                file_path = vault_path / file_pattern
                if file_path.exists():
                    if is_excluded_path(file_path, vault_path, self._exclude_paths):
                        continue
                    if not is_included_path(file_path, vault_path, self._config):
                        continue
                    content = self._read_file_cached(file_path)
                    if not content:
                        continue
                    metadata = {
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "source": "relationship",
                    }
                    results.append((content, 0.9, metadata))
                    total_added += 1

            if RELATIONSHIP_TOTAL_LIMIT > 0 and total_added >= RELATIONSHIP_TOTAL_LIMIT:
                break

        self._relationship_cache[cache_key] = results
        return results

    def _boost_by_metadata(
        self,
        results: List[Tuple[str, float, Dict]],
    ) -> List[Tuple[str, float, Dict]]:
        """
        Boost scores based on metadata (recency, backlinks).
        """
        boosted = []
        now = datetime.now()

        for content, score, metadata in results:
            boost = 1.0

            # Boost recent files
            if 'modified_time' in metadata:
                try:
                    mod_time = datetime.fromisoformat(metadata['modified_time'])
                    days_old = (now - mod_time).days
                    if days_old < 7:
                        boost += 0.2
                    elif days_old < 30:
                        boost += 0.1
                except Exception:
                    pass

            # Boost highly linked files
            backlink_count = metadata.get('backlink_count', 0)
            if backlink_count > 5:
                boost += 0.15
            elif backlink_count > 2:
                boost += 0.1

            boosted.append((content, score * boost, metadata))

        return boosted

    def _format_context(
        self,
        results: List[Tuple[str, float, Dict]],
        use_case: Optional[UseCase],
        max_tokens: int,
    ) -> str:
        """
        Format retrieved content into structured context.
        """
        def extract_key_lines(text: str, max_lines: int = 4) -> List[str]:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines:
                return []

            # Prefer structured lines: bullets, numbered items, or key-value style
            preferred = []
            for ln in lines[:200]:
                if ln.startswith(("-", "*")):
                    preferred.append(ln)
                elif ln[:2].isdigit() and (")" in ln[:4] or "." in ln[:4]):
                    preferred.append(ln)
                elif ":" in ln and len(ln) < 140:
                    preferred.append(ln)

            # Boost known critical keywords if present
            keyword_hits = []
            keywords = [
                "March 2026",
                "H1B",
                "Solutions Engineer",
                "Implementation",
                "TAM",
                "Enterprise SaaS",
                "AI security",
                "positioning",
                "visa",
                "deadline",
            ]
            for ln in lines[:400]:
                if any(k.lower() in ln.lower() for k in keywords):
                    keyword_hits.append(ln)

            # Merge with priority: keyword hits, then preferred, then first lines
            merged = []
            for ln in keyword_hits + preferred + lines[:10]:
                if ln not in merged:
                    merged.append(ln)
                if len(merged) >= max_lines:
                    break

            return merged[:max_lines]

        # Deduplicate by file path
        seen_files = set()
        unique_results = []
        for content, score, metadata in results:
            file_path = metadata.get('file_path', '')
            if file_path not in seen_files:
                seen_files.add(file_path)
                unique_results.append((content, score, metadata))

        # Sort by score
        unique_results.sort(key=lambda x: x[1], reverse=True)

        # Build context sections
        context_parts = []

        if use_case:
            context_parts.append(f"[MemFlow Context - {use_case.name}]")
            context_parts.append("")

        # Group by category
        job_context = []
        project_context = []
        identity_context = []
        other_context = []

        for content, score, metadata in unique_results[:5]:  # Top 5 results
            file_path = metadata.get('file_path', '')
            file_name = metadata.get('file_name', 'Unknown')

            key_lines = extract_key_lines(content, max_lines=4)
            if key_lines:
                entry = f"- {file_name}: " + " | ".join(key_lines[:4])
            else:
                summary = content[:500].split('\n\n')[0]
                entry = f"- {file_name}: {summary[:200]}..."

            if 'Job/' in file_path or 'Career' in file_path:
                job_context.append(entry)
            elif 'Project' in file_path:
                project_context.append(entry)
            elif 'Identity' in file_path:
                identity_context.append(entry)
            else:
                other_context.append(entry)

        if job_context:
            context_parts.append("JOB/CAREER CONTEXT:")
            context_parts.extend(job_context[:2])
            context_parts.append("")

        if identity_context:
            context_parts.append("IDENTITY/POSITIONING:")
            context_parts.extend(identity_context[:2])
            context_parts.append("")

        if project_context:
            context_parts.append("CURRENT PROJECTS:")
            context_parts.extend(project_context[:2])
            context_parts.append("")

        if other_context and len(context_parts) < 10:
            context_parts.append("RELATED NOTES:")
            context_parts.extend(other_context[:2])
            context_parts.append("")

        context_parts.append("[End MemFlow Context]")

        # Truncate if too long (rough token estimate: 4 chars per token)
        full_context = '\n'.join(context_parts)
        max_chars = max_tokens * 4
        if len(full_context) > max_chars:
            full_context = full_context[:max_chars] + "\n...[truncated]"

        return full_context

    def get_context(self, query: str, max_tokens: int = 1000) -> ContextResult:
        """
        Main retrieval function combining vector search with relationships.

        Args:
            query: User's question
            max_tokens: Maximum context tokens to return

        Returns:
            ContextResult with context, sources, and confidence
        """
        import time
        start_time = time.time()

        # 1. Check if query matches a use case
        t0 = time.time()
        use_case = self.relationship_parser.match_use_case(query)
        t1 = time.time()

        # 2. Perform vector similarity search
        vector_results = self._vector_search(query, top_k=5)
        t2 = time.time()

        # 3. If use case matched, add relationship-guided results
        if use_case:
            logger.info(f"Matched use case: {use_case.name}")
            relationship_results = self._relationship_search(use_case)
            all_results = vector_results + relationship_results
        else:
            all_results = vector_results
        t3 = time.time()

        # 4. Boost by metadata
        boosted_results = self._boost_by_metadata(all_results)
        t4 = time.time()

        # 5. Format context
        context = self._format_context(boosted_results, use_case, max_tokens)
        t5 = time.time()

        # 6. Extract sources
        sources = list(set(
            r[2].get('file_name', 'Unknown')
            for r in boosted_results[:5]
        ))

        # 7. Calculate confidence
        if boosted_results:
            avg_score = sum(r[1] for r in boosted_results[:3]) / min(3, len(boosted_results))
            confidence = min(avg_score, 1.0)
        else:
            confidence = 0.0

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Query completed in {elapsed_ms:.1f}ms, confidence: {confidence:.2f}")
        if MEMFLOW_PROFILE:
            logger.info(
                "Timing breakdown (ms) - use_case: %.1f, vector: %.1f, relationship: %.1f, boost: %.1f, format: %.1f",
                (t1 - t0) * 1000,
                (t2 - t1) * 1000,
                (t3 - t2) * 1000,
                (t4 - t3) * 1000,
                (t5 - t4) * 1000,
            )

        return ContextResult(
            context=context,
            sources=sources,
            confidence=confidence,
            use_case=use_case.name if use_case else None,
        )


# Singleton instance for easy import
_engine: Optional[MemFlowQueryEngine] = None


def get_context(query: str, max_tokens: int = 1000) -> Dict[str, Any]:
    """
    Convenience function for getting context.

    Args:
        query: User's question
        max_tokens: Maximum context tokens

    Returns:
        Dictionary with context, sources, and confidence
    """
    global _engine
    if _engine is None:
        _engine = MemFlowQueryEngine()

    result = _engine.get_context(query, max_tokens)
    return {
        "context": result.context,
        "sources": result.sources,
        "confidence": result.confidence,
        "use_case": result.use_case,
    }


if __name__ == "__main__":
    # Test queries
    test_queries = [
        "Should I learn LangChain?",
        "Help me write a LinkedIn post about OAuth",
        "Should I apply to this company?",
        "What project should I work on next?",
    ]

    print("MemFlow Query Engine Test\n" + "=" * 50)

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 40)

        try:
            result = get_context(query)
            print(f"Use Case: {result['use_case']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Sources: {', '.join(result['sources'])}")
            print(f"\nContext Preview:\n{result['context'][:500]}...")
        except Exception as e:
            print(f"Error: {e}")

        print()
