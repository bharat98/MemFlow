# MemFlow Setup Guide

This guide walks you through setting up MemFlow for automatic context injection from your Obsidian vault into Claude Desktop conversations.

## Prerequisites

- Python 3.9 or higher
- Obsidian vault (your "Second Brain")
- Claude Desktop (for MCP integration)

## Quick Start

### 1. Create Virtual Environment

```bash
# Navigate to the src directory
cd "/mnt/c/Users/gurba/Documents/Second Brain/Projects/Personal/MemFlow/src"

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac/WSL)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** First run will download the embedding model (~130MB). This only happens once.

### 3. Index Your Vault

```bash
python index_vault.py --vault "/mnt/c/Users/gurba/Documents/Second Brain"
```

Expected output:
```
Loading embedding model: BAAI/bge-small-en-v1.5
Creating index from XXX documents...
Persisting index to: .memflow_index
✅ Indexed XXX documents
```

### 4. Test Retrieval

```bash
python test_retrieval.py
```

This runs test queries and validates the retrieval is working.

### 5. Start MCP Server

```bash
python mcp_server.py
```

The server runs at `http://localhost:8080`. Test it:

```bash
curl http://localhost:8080/tools/list
curl http://localhost:8080/health
```

### 6. Configure Claude Desktop

**Windows:** Copy config to `%APPDATA%\Claude\claude_desktop_config.json`

```bash
# From PowerShell
copy claude_desktop_config.json "$env:APPDATA\Claude\claude_desktop_config.json"
```

**Mac:** Copy config to `~/Library/Application Support/Claude/claude_desktop_config.json`

```bash
cp claude_desktop_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Important:**
- Use absolute paths in the config
- Restart Claude Desktop after adding the config

### 7. Verify Integration

1. Start the MCP server: `python mcp_server.py`
2. Open Claude Desktop
3. Start a new conversation
4. Ask: "Should I learn LangChain?"
5. Claude should automatically use the `get_vault_context` tool

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VAULT_PATH` | `/mnt/c/Users/gurba/Documents/Second Brain` | Path to Obsidian vault |
| `INDEX_STORAGE` | `./src/.memflow_index` | Where to store the index |
| `MCP_HOST` | `0.0.0.0` | Server bind address |
| `MCP_PORT` | `8080` | Server port |
| `ENABLE_FILE_WATCHER` | `true` | Auto-update index on file changes |
| `DEBOUNCE_SECONDS` | `30` | Wait time before re-indexing |

### Paths for Your System

Update these paths in `claude_desktop_config.json`:

**Windows (native):**
```json
{
  "args": ["C:/Users/gurba/Documents/Second Brain/Projects/Personal/MemFlow/src/mcp_server.py"],
  "env": {
    "VAULT_PATH": "C:/Users/gurba/Documents/Second Brain"
  }
}
```

**WSL:**
```json
{
  "args": ["/mnt/c/Users/gurba/Documents/Second Brain/Projects/Personal/MemFlow/src/mcp_server.py"],
  "env": {
    "VAULT_PATH": "/mnt/c/Users/gurba/Documents/Second Brain"
  }
}
```

## Troubleshooting

### Index not found
```bash
# Rebuild the index
python index_vault.py --rebuild
```

### Embedding model download fails
```bash
# Pre-download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
```

### Claude Desktop doesn't see the MCP server
1. Check the config file location
2. Ensure paths are absolute
3. Restart Claude Desktop completely (quit and reopen)
4. Check server is running: `curl http://localhost:8080/health`

### Slow retrieval (>500ms)
1. Ensure embedding model is cached (not re-downloading)
2. Check vault size (>1000 notes may need optimization)
3. First query is slower due to model loading

### File watcher not updating
1. Check watcher is enabled: `curl http://localhost:8080/stats`
2. Wait for debounce period (30 seconds)
3. Check logs for update messages

## Files Overview

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `index_vault.py` | Creates/updates the vector index |
| `query_engine.py` | Retrieves context from index |
| `file_watcher.py` | Auto-updates index on file changes |
| `test_retrieval.py` | Validation tests |
| `mcp_server.py` | MCP protocol server for Claude |
| `claude_desktop_config.json` | Claude Desktop configuration |
| `.memflow_index/` | Persisted vector index (generated) |

## Running as a Service (Optional)

To run the MCP server automatically on startup:

**Windows (Task Scheduler):**
1. Open Task Scheduler
2. Create Basic Task
3. Trigger: At startup
4. Action: Start a program
5. Program: `python`
6. Arguments: `"C:\Users\gurba\Documents\Second Brain\Projects\Personal\MemFlow\src\mcp_server.py"`

**Linux/Mac (systemd):**
```ini
# /etc/systemd/user/memflow.service
[Unit]
Description=MemFlow MCP Server

[Service]
ExecStart=/path/to/venv/bin/python /path/to/mcp_server.py
Restart=always

[Install]
WantedBy=default.target
```

## Next Steps

1. **Test with real queries** - Ask Claude about learning, jobs, projects
2. **Review context quality** - Check if retrieved context is relevant
3. **Update Relationships.md** - Add more use cases if needed
4. **Monitor performance** - Check retrieval latency stays <200ms

## Support

If you encounter issues:
1. Check logs: `python mcp_server.py` shows detailed logging
2. Run tests: `python test_retrieval.py`
3. Check health: `curl http://localhost:8080/health`
