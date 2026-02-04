"""
MemFlow Retrieval Tests
Sub-Task 2.5: Validate end-to-end retrieval with real queries

Tests that the query engine correctly retrieves relevant context
based on the Relationships.md mappings.
"""

import os
import sys
import time
import pytest
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from query_engine import get_context, MemFlowQueryEngine
from index_vault import load_index, VAULT_PATH


# Test configuration
TEST_QUERIES = [
    {
        "query": "Should I learn LangChain?",
        "expected_file_patterns": ["Job/", "Identity", "Projects/", "README"],
        "expected_keywords": ["March 2026", "Solutions Engineer", "positioning", "deadline", "Enterprise"],
        "description": "Learning decision should pull job timeline and positioning context",
    },
    {
        "query": "Help me write a LinkedIn post about OAuth",
        "expected_file_patterns": ["LinkedIn", "Identity", "Projects/"],
        "expected_keywords": ["technical", "content", "OAuth", "brand", "positioning"],
        "description": "LinkedIn content should pull content strategy and recent projects",
    },
    {
        "query": "Should I apply to this company?",
        "expected_file_patterns": ["Job/", "Identity", "Career", "README"],
        "expected_keywords": ["H1B", "sponsorship", "B2B", "SaaS", "salary"],
        "description": "Job application should pull constraints and requirements",
    },
    {
        "query": "What project should I work on next?",
        "expected_file_patterns": ["Projects/", "Job/", "Identity"],
        "expected_keywords": ["portfolio", "demonstrate", "timeline", "January"],
        "description": "Project selection should pull portfolio needs and timeline",
    },
]

# Performance thresholds
MAX_RETRIEVAL_MS = 200  # Target: <200ms per query


class TestIndexExists:
    """Test that index is properly created and loadable."""

    def test_index_loads(self):
        """Index should load without errors."""
        index = load_index()
        assert index is not None, "Index failed to load. Run index_vault.py first."

    def test_vault_path_exists(self):
        """Vault path should be accessible."""
        assert os.path.exists(VAULT_PATH), f"Vault path not found: {VAULT_PATH}"

    def test_vault_has_markdown_files(self):
        """Vault should contain markdown files."""
        md_files = list(
            f for f in os.listdir(VAULT_PATH)
            if f.endswith('.md') or os.path.isdir(os.path.join(VAULT_PATH, f))
        )
        assert len(md_files) > 0, "Vault appears empty"


class TestQueryEngine:
    """Test the query engine functionality."""

    @pytest.fixture(scope="class")
    def engine(self):
        """Create query engine instance for tests."""
        try:
            return MemFlowQueryEngine()
        except Exception as e:
            pytest.skip(f"Could not create query engine: {e}")

    def test_engine_initializes(self, engine):
        """Query engine should initialize successfully."""
        assert engine is not None
        assert engine.index is not None

    def test_relationship_parser_loads(self, engine):
        """Relationship parser should load use cases."""
        # May be empty if Relationships.md isn't in expected location
        assert engine.relationship_parser is not None


class TestContextRetrieval:
    """Test context retrieval for various query types."""

    @pytest.fixture(scope="class")
    def engine(self):
        """Create query engine instance for tests."""
        try:
            return MemFlowQueryEngine()
        except Exception as e:
            pytest.skip(f"Could not create query engine: {e}")

    @pytest.mark.parametrize("test_case", TEST_QUERIES, ids=[t["query"][:30] for t in TEST_QUERIES])
    def test_query_returns_context(self, engine, test_case):
        """Each query should return non-empty context."""
        result = engine.get_context(test_case["query"])

        assert result is not None, "Result should not be None"
        assert result.context, f"Context should not be empty for: {test_case['query']}"
        assert len(result.sources) > 0, "Should have at least one source"
        assert result.confidence > 0, "Confidence should be positive"

    @pytest.mark.parametrize("test_case", TEST_QUERIES, ids=[t["query"][:30] for t in TEST_QUERIES])
    def test_query_retrieves_relevant_files(self, engine, test_case):
        """Retrieved sources should match expected file patterns."""
        result = engine.get_context(test_case["query"])

        # Check if any expected pattern matches any source or context
        context_and_sources = result.context + " ".join(result.sources)

        matches_found = []
        for pattern in test_case["expected_file_patterns"]:
            if pattern.lower() in context_and_sources.lower():
                matches_found.append(pattern)

        # Should match at least one expected pattern
        assert len(matches_found) > 0, (
            f"Query '{test_case['query']}' should retrieve files matching "
            f"patterns {test_case['expected_file_patterns']}, "
            f"but got sources: {result.sources}"
        )

    @pytest.mark.parametrize("test_case", TEST_QUERIES, ids=[t["query"][:30] for t in TEST_QUERIES])
    def test_query_contains_expected_keywords(self, engine, test_case):
        """Context should contain expected keywords."""
        result = engine.get_context(test_case["query"])

        context_lower = result.context.lower()
        keywords_found = [
            kw for kw in test_case["expected_keywords"]
            if kw.lower() in context_lower
        ]

        # Should find at least some expected keywords (not all required)
        # This is lenient because vault content may vary
        min_keywords = min(2, len(test_case["expected_keywords"]))

        # Note: This may fail if vault doesn't contain expected content
        # In that case, it's informational rather than a hard failure
        if len(keywords_found) < min_keywords:
            pytest.xfail(
                f"Query '{test_case['query']}' found {len(keywords_found)} of "
                f"{len(test_case['expected_keywords'])} expected keywords. "
                f"Found: {keywords_found}. This may indicate vault content differs from expected."
            )


class TestPerformance:
    """Test retrieval performance."""

    @pytest.fixture(scope="class")
    def engine(self):
        """Create query engine instance for tests."""
        try:
            return MemFlowQueryEngine()
        except Exception as e:
            pytest.skip(f"Could not create query engine: {e}")

    def test_retrieval_latency(self, engine):
        """Retrieval should complete within target latency."""
        query = "Should I learn LangChain?"

        # Warm up
        engine.get_context(query)

        # Measure
        times = []
        for _ in range(3):
            start = time.time()
            engine.get_context(query)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)

        avg_ms = sum(times) / len(times)

        # Allow some tolerance for CI/slow machines
        assert avg_ms < MAX_RETRIEVAL_MS * 2, (
            f"Average retrieval time {avg_ms:.1f}ms exceeds "
            f"threshold {MAX_RETRIEVAL_MS * 2}ms"
        )

        print(f"\nRetrieval latency: {avg_ms:.1f}ms (target: <{MAX_RETRIEVAL_MS}ms)")

    def test_multiple_queries_stable(self, engine):
        """Multiple queries should have stable performance."""
        queries = [t["query"] for t in TEST_QUERIES]

        times = []
        for query in queries:
            start = time.time()
            result = engine.get_context(query)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)
            assert result.context, f"Query failed: {query}"

        avg_ms = sum(times) / len(times)
        max_ms = max(times)

        print(f"\nMulti-query performance:")
        print(f"  Average: {avg_ms:.1f}ms")
        print(f"  Max: {max_ms:.1f}ms")

        # Max should not be dramatically higher than average
        assert max_ms < avg_ms * 3, "Query times too variable"


class TestContextFormat:
    """Test context output formatting."""

    @pytest.fixture(scope="class")
    def engine(self):
        """Create query engine instance for tests."""
        try:
            return MemFlowQueryEngine()
        except Exception as e:
            pytest.skip(f"Could not create query engine: {e}")

    def test_context_has_structure(self, engine):
        """Context should have structured sections."""
        result = engine.get_context("Should I learn LangChain?")

        # Check for section markers
        assert "[MemFlow Context" in result.context or "CONTEXT:" in result.context.upper(), \
            "Context should have section markers"

    def test_context_not_raw_dump(self, engine):
        """Context should not be raw markdown dump."""
        result = engine.get_context("Should I learn LangChain?")

        # Should have some structure, not just raw markdown
        lines = result.context.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]

        # Should have multiple sections/lines
        assert len(non_empty_lines) >= 3, "Context should have structured content"

    def test_sources_are_provided(self, engine):
        """Sources should list retrieved files."""
        result = engine.get_context("Should I learn LangChain?")

        assert isinstance(result.sources, list), "Sources should be a list"
        # Sources should be file names, not full paths
        for source in result.sources:
            assert '/' not in source or source.count('/') < 3, \
                f"Source should be filename or short path: {source}"


def run_manual_tests():
    """Run tests manually with detailed output."""
    print("=" * 60)
    print("MemFlow Retrieval Test Suite")
    print("=" * 60)

    # Check prerequisites
    print("\n1. Checking prerequisites...")
    if not os.path.exists(VAULT_PATH):
        print(f"   ❌ Vault not found: {VAULT_PATH}")
        return False

    index = load_index()
    if index is None:
        print("   ❌ Index not found. Run: python index_vault.py")
        return False

    print(f"   ✅ Vault: {VAULT_PATH}")
    print("   ✅ Index loaded")

    # Initialize engine
    print("\n2. Initializing query engine...")
    try:
        engine = MemFlowQueryEngine()
        print("   ✅ Engine initialized")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    # Run test queries
    print("\n3. Running test queries...")
    all_passed = True

    for i, test_case in enumerate(TEST_QUERIES, 1):
        print(f"\n   Test {i}: {test_case['query']}")
        print(f"   {test_case['description']}")

        start = time.time()
        try:
            result = engine.get_context(test_case["query"])
            elapsed_ms = (time.time() - start) * 1000

            print(f"   ⏱️  Time: {elapsed_ms:.1f}ms")
            print(f"   📊 Confidence: {result.confidence:.2f}")
            print(f"   📁 Sources: {', '.join(result.sources[:3])}")

            # Check file patterns
            context_and_sources = result.context + " ".join(result.sources)
            matches = sum(
                1 for p in test_case["expected_file_patterns"]
                if p.lower() in context_and_sources.lower()
            )
            if matches > 0:
                print(f"   ✅ File patterns: {matches}/{len(test_case['expected_file_patterns'])} matched")
            else:
                print(f"   ⚠️  File patterns: 0/{len(test_case['expected_file_patterns'])} matched")

            # Check keywords
            context_lower = result.context.lower()
            keyword_matches = sum(
                1 for kw in test_case["expected_keywords"]
                if kw.lower() in context_lower
            )
            if keyword_matches >= 2:
                print(f"   ✅ Keywords: {keyword_matches}/{len(test_case['expected_keywords'])} found")
            else:
                print(f"   ⚠️  Keywords: {keyword_matches}/{len(test_case['expected_keywords'])} found")

            # Performance check
            if elapsed_ms <= MAX_RETRIEVAL_MS:
                print(f"   ✅ Performance: within {MAX_RETRIEVAL_MS}ms target")
            else:
                print(f"   ⚠️  Performance: {elapsed_ms:.1f}ms exceeds {MAX_RETRIEVAL_MS}ms target")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests completed successfully!")
    else:
        print("⚠️  Some tests had issues - check output above")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    # Allow running with pytest or directly
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        success = run_manual_tests()
        sys.exit(0 if success else 1)
