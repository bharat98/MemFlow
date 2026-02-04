"""
MemFlow MCP Server
Sub-Task 3.1: FastAPI server implementing MCP protocol

Exposes the MemFlow retrieval engine to Claude Desktop and other AI tools
via the Model Context Protocol (MCP).
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from query_engine import get_context, MemFlowQueryEngine
from file_watcher import start_background_watcher, stop_background_watcher, get_watcher
from index_vault import VAULT_PATH, INDEX_STORAGE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
HOST = os.environ.get("MCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("MCP_PORT", "8080"))
ENABLE_FILE_WATCHER = os.environ.get("ENABLE_FILE_WATCHER", "true").lower() == "true"


# Pydantic models for MCP protocol
class ToolInputSchema(BaseModel):
    type: str = "object"
    properties: Dict[str, Any] = {}
    required: List[str] = []


class Tool(BaseModel):
    name: str
    description: str
    inputSchema: ToolInputSchema


class ToolsListResponse(BaseModel):
    tools: List[Tool]


class ToolCallArguments(BaseModel):
    query: str = Field(..., description="The user's question or request")
    max_tokens: Optional[int] = Field(1000, description="Maximum context tokens to return")


class ToolCallParams(BaseModel):
    name: str
    arguments: ToolCallArguments


class ToolCallRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: str = "tools/call"
    params: ToolCallParams


class TextContent(BaseModel):
    type: str = "text"
    text: str


class ToolCallResponse(BaseModel):
    content: List[TextContent]
    isError: bool = False


class ServerInfo(BaseModel):
    name: str = "memflow"
    version: str = "0.1.0"
    vault_path: str
    index_path: str
    watcher_enabled: bool
    watcher_status: Optional[Dict[str, Any]] = None


# Global engine instance
_engine: Optional[MemFlowQueryEngine] = None


def get_engine() -> MemFlowQueryEngine:
    """Get or create the query engine instance."""
    global _engine
    if _engine is None:
        logger.info("Initializing MemFlow query engine...")
        _engine = MemFlowQueryEngine()
        logger.info("Query engine initialized")
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("=" * 50)
    logger.info("MemFlow MCP Server Starting")
    logger.info("=" * 50)

    # Initialize query engine
    try:
        get_engine()
        logger.info(f"✅ Query engine ready")
    except Exception as e:
        logger.error(f"❌ Failed to initialize query engine: {e}")
        logger.error("Run 'python index_vault.py' first to create the index")

    # Start file watcher if enabled
    if ENABLE_FILE_WATCHER:
        try:
            start_background_watcher(VAULT_PATH)
            logger.info(f"✅ File watcher started for: {VAULT_PATH}")
        except Exception as e:
            logger.warning(f"⚠️ Could not start file watcher: {e}")

    logger.info(f"✅ Server running at http://{HOST}:{PORT}")
    logger.info("=" * 50)

    yield

    # Shutdown
    logger.info("Shutting down...")
    if ENABLE_FILE_WATCHER:
        stop_background_watcher()
    logger.info("Server stopped")


# Create FastAPI app
app = FastAPI(
    title="MemFlow MCP Server",
    description="Model Context Protocol server for Obsidian vault context retrieval",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for browser extension access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# MCP Protocol Endpoints

@app.get("/")
async def root():
    """Root endpoint with server info."""
    watcher = get_watcher()
    return ServerInfo(
        vault_path=VAULT_PATH,
        index_path=INDEX_STORAGE,
        watcher_enabled=ENABLE_FILE_WATCHER,
        watcher_status=watcher.get_stats() if watcher else None,
    )


@app.get("/tools/list", response_model=ToolsListResponse)
async def list_tools():
    """
    List available MCP tools.

    This endpoint is called by Claude Desktop to discover available tools.
    """
    return ToolsListResponse(
        tools=[
            Tool(
                name="get_vault_context",
                description=(
                    "Retrieve relevant context from the user's Obsidian vault based on their query. "
                    "Returns personalized information about job search, projects, skills, and goals. "
                    "Use this tool when the user asks questions that might benefit from their personal context."
                ),
                inputSchema=ToolInputSchema(
                    type="object",
                    properties={
                        "query": {
                            "type": "string",
                            "description": "The user's question or topic to find context for"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens of context to return (default: 1000)",
                            "default": 1000
                        }
                    },
                    required=["query"]
                )
            )
        ]
    )


@app.post("/tools/call", response_model=ToolCallResponse)
async def call_tool(request: ToolCallRequest):
    """
    Execute an MCP tool call.

    This endpoint is called by Claude Desktop when it wants to use a tool.
    """
    logger.info(f"Tool call received: {request.params.name}")
    logger.info(f"Query: {request.params.arguments.query}")

    if request.params.name != "get_vault_context":
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tool: {request.params.name}"
        )

    try:
        # Get context from vault
        result = get_context(
            query=request.params.arguments.query,
            max_tokens=request.params.arguments.max_tokens or 1000,
        )

        logger.info(f"Context retrieved: {len(result['context'])} chars, confidence: {result['confidence']:.2f}")

        return ToolCallResponse(
            content=[
                TextContent(
                    type="text",
                    text=result["context"]
                )
            ],
            isError=False,
        )

    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return ToolCallResponse(
            content=[
                TextContent(
                    type="text",
                    text=f"Error retrieving context: {str(e)}"
                )
            ],
            isError=True,
        )


# Additional utility endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        engine = get_engine()
        return {
            "status": "healthy",
            "engine": "ready" if engine else "not initialized",
            "vault_path": VAULT_PATH,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@app.get("/stats")
async def get_stats():
    """Get server statistics."""
    watcher = get_watcher()
    return {
        "vault_path": VAULT_PATH,
        "index_path": INDEX_STORAGE,
        "file_watcher": watcher.get_stats() if watcher else None,
    }


@app.post("/query")
async def direct_query(query: str, max_tokens: int = 1000):
    """
    Direct query endpoint for testing (not part of MCP protocol).

    Args:
        query: The search query
        max_tokens: Maximum context tokens
    """
    try:
        result = get_context(query, max_tokens)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex")
async def trigger_reindex():
    """
    Trigger a full re-index of the vault.

    This is useful if the index gets out of sync.
    """
    from index_vault import index_vault

    try:
        global _engine
        logger.info("Re-indexing vault...")
        index_vault(force_rebuild=True)
        _engine = None  # Reset engine to pick up new index
        get_engine()
        logger.info("Re-index complete")
        return {"status": "success", "message": "Vault re-indexed successfully"}
    except Exception as e:
        logger.error(f"Re-index failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="MemFlow MCP Server")
    parser.add_argument("--host", default=HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=PORT, help="Port to listen on")
    parser.add_argument("--no-watcher", action="store_true", help="Disable file watcher")
    parser.add_argument("--vault", default=VAULT_PATH, help="Path to Obsidian vault")

    args = parser.parse_args()

    global ENABLE_FILE_WATCHER, VAULT_PATH
    if args.no_watcher:
        ENABLE_FILE_WATCHER = False
    if args.vault:
        os.environ["VAULT_PATH"] = args.vault

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
