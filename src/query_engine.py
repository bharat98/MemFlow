"""
MemFlow Query Engine
Sub-Task 2.3: Context retrieval using Relationships.md for better context

Implements hybrid retrieval combining:
1. Vector similarity search (semantic)
2. Relationship-guided retrieval (explicit connections from Relationships.md)
3. Metadata boosting (recent files, high backlinks)
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from index_vault import load_index, VAULT_PATH, INDEX_STORAGE, EMBEDDING_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to Relationships.md
RELATIONSHIPS_PATH = os.path.join(VAULT_PATH, "Projects", "Personal", "MemFlow", "Relationships.md")


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
        self._parse()

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
        learning_triggers = ['learn', 'should i', 'worth it', 'skill', 'technology', 'framework', 'course']
        linkedin_triggers = ['linkedin', 'post', 'content', 'write a', 'publish']
        project_triggers = ['build', 'project', 'what should i', 'prioritize', 'next project']
        job_triggers = ['apply', 'job', 'company', 'resume', 'application', 'fit', 'role']

        for use_case in self.use_cases:
            # Check explicit triggers
            for trigger in use_case.triggers:
                if trigger.lower() in query_lower:
                    return use_case

        # Check category-based matching
        if any(t in query_lower for t in learning_triggers):
            for uc in self.use_cases:
                if 'Learning' in uc.name or 'Skill' in uc.name:
                    return uc

        if any(t in query_lower for t in linkedin_triggers):
            for uc in self.use_cases:
                if 'LinkedIn' in uc.name or 'Content' in uc.name:
                    return uc

        if any(t in query_lower for t in project_triggers):
            for uc in self.use_cases:
                if 'Project' in uc.name:
                    return uc

        if any(t in query_lower for t in job_triggers):
            for uc in self.use_cases:
                if 'Job' in uc.name or 'Application' in uc.name:
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

        # Configure embedding model
        embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            trust_remote_code=True,
        )
        Settings.embed_model = embed_model
        Settings.llm = None

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
        results = []
        vault_path = Path(VAULT_PATH)

        for file_pattern in use_case.relevant_files:
            # Handle directory patterns (ending with /)
            if file_pattern.endswith('/'):
                search_path = vault_path / file_pattern.rstrip('/')
                if search_path.exists() and search_path.is_dir():
                    for md_file in search_path.rglob('*.md'):
                        try:
                            content = md_file.read_text(encoding='utf-8')
                            metadata = {
                                "file_path": str(md_file),
                                "file_name": md_file.name,
                                "source": "relationship",
                            }
                            # Higher score for relationship-based matches
                            results.append((content, 0.8, metadata))
                        except Exception as e:
                            logger.debug(f"Error reading {md_file}: {e}")
            else:
                # Direct file reference
                file_path = vault_path / file_pattern
                if file_path.exists():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        metadata = {
                            "file_path": str(file_path),
                            "file_name": file_path.name,
                            "source": "relationship",
                        }
                        results.append((content, 0.9, metadata))
                    except Exception as e:
                        logger.debug(f"Error reading {file_path}: {e}")

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

            # Extract key information (first 500 chars or until double newline)
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
        use_case = self.relationship_parser.match_use_case(query)

        # 2. Perform vector similarity search
        vector_results = self._vector_search(query, top_k=5)

        # 3. If use case matched, add relationship-guided results
        if use_case:
            logger.info(f"Matched use case: {use_case.name}")
            relationship_results = self._relationship_search(use_case)
            all_results = vector_results + relationship_results
        else:
            all_results = vector_results

        # 4. Boost by metadata
        boosted_results = self._boost_by_metadata(all_results)

        # 5. Format context
        context = self._format_context(boosted_results, use_case, max_tokens)

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
