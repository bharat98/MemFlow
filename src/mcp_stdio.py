"""
MemFlow MCP StdIO Server

Implements MCP over stdio for Claude Desktop.
All non-JSON output is redirected to stderr to avoid protocol errors.
"""

import json
import sys
import traceback
from typing import Any, Dict

# Redirect all default stdout to stderr so only JSON goes to the real stdout.
_json_stdout = sys.stdout
sys.stdout = sys.stderr

# Import after stdout redirect to avoid stray prints on stdout.
from query_engine import get_context  # noqa: E402


def _write_json(message: Dict[str, Any]) -> None:
    _json_stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    _json_stdout.flush()


def _handle_initialize(msg: Dict[str, Any]) -> None:
    _write_json(
        {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "memflow", "version": "0.1.0"},
            },
        }
    )


def _handle_tools_list(msg: Dict[str, Any]) -> None:
    _write_json(
        {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": {
                "tools": [
                    {
                        "name": "get_vault_context",
                        "description": (
                            "Retrieve relevant context from the user's Obsidian vault based on their query. "
                            "Returns personalized information about job search, projects, skills, and goals."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The user's question or topic to find context for",
                                },
                                "max_tokens": {
                                    "type": "integer",
                                    "description": "Maximum tokens of context to return (default: 1000)",
                                    "default": 1000,
                                },
                            },
                            "required": ["query"],
                        },
                    }
                ]
            },
        }
    )


def _handle_tools_call(msg: Dict[str, Any]) -> None:
    params = msg.get("params", {}) or {}
    arguments = params.get("arguments", {}) or {}
    query = arguments.get("query", "")
    max_tokens = arguments.get("max_tokens", 1000)

    result = get_context(query=query, max_tokens=max_tokens)
    _write_json(
        {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": {
                "content": [{"type": "text", "text": result.get("context", "")}],
                "isError": False,
            },
        }
    )


def _handle_error(msg_id: Any, message: str) -> None:
    # Only respond with errors when an id is provided.
    if msg_id is None:
        print(f"mcp_stdio error (no id): {message}", file=sys.stderr)
        return
    _write_json(
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32000, "message": message},
        }
    )


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except Exception:
            _handle_error(None, "Invalid JSON")
            continue

        method = msg.get("method")

        try:
            if method == "initialize":
                _handle_initialize(msg)
            elif method == "tools/list":
                _handle_tools_list(msg)
            elif method == "tools/call":
                _handle_tools_call(msg)
            elif method in ("notifications/cancelled", "notifications/initialized"):
                # No response required
                continue
            else:
                _handle_error(msg.get("id"), f"Unknown method: {method}")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _handle_error(msg.get("id"), str(exc))


if __name__ == "__main__":
    main()
