# MemFlow Implementation Task - Parts 2 & 3

## Context Files (READ FIRST)
- `Problem.md` - Understanding what we're solving
- `Solution.md` - Complete architecture overview
- `Relationships.md` - Output from Part 1 (Gemini), now in vault

## Mission
Implement MemFlow Parts 2 (Retrieval Engine) and 3 (Connection Layer) to enable automatic memory injection from Obsidian vault into AI conversations.

---

## Part 2: LlamaIndex Retrieval Engine

### Objective
Create a Python service that indexes the Obsidian vault and retrieves relevant context based on queries, using the explicit relationships from `Relationships.md`.

### Sub-Task 2.1: Environment Setup
**Goal:** Install dependencies and verify vault access

**Actions:**
```bash
# Create virtual environment
python -m venv memflow_env
source memflow_env/bin/activate  # or memflow_env\Scripts\activate on Windows

# Install core dependencies
pip install llama-index==0.9.48
pip install llama-index-readers-file
pip install sentence-transformers
pip install watchdog
pip install fastapi
pip install uvicorn
```

**Validation:**
- [ ] All packages install without errors
- [ ] Can import: `from llama_index.core import VectorStoreIndex`
- [ ] Vault path is accessible: `/path/to/obsidian/vault`

**Deliverable:** `requirements.txt` file with pinned versions

---

### Sub-Task 2.2: Vault Indexing Script
**Goal:** Create initial indexing of vault with local embeddings

**File:** `index_vault.py`

**Requirements:**
- Read all .md files from vault (including subdirectories)
- Use local embedding model: `BAAI/bge-small-en-v1.5` (fast, good quality)
- Parse markdown including:
  - Obsidian wikilinks [[Note Name]]
  - Frontmatter (YAML)
  - Relationships.md structure
- Store index locally (FAISS or similar)
- Index metadata: modified time, folder, backlink count

**Pseudo-code:**
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path

VAULT_PATH = "/path/to/obsidian/vault"  # REPLACE WITH ACTUAL PATH
INDEX_STORAGE = ".memflow_index"

def index_vault():
    # Load all markdown files
    reader = SimpleDirectoryReader(
        input_dir=VAULT_PATH,
        required_exts=[".md"],
        recursive=True
    )
    documents = reader.load_data()
    
    # Use local embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        show_progress=True
    )
    
    # Persist to disk
    index.storage_context.persist(persist_dir=INDEX_STORAGE)
    print(f"✅ Indexed {len(documents)} documents")
    
if __name__ == "__main__":
    index_vault()
```

**Validation:**
- [ ] Script runs without errors
- [ ] `.memflow_index/` directory is created
- [ ] Can query index: `index.as_query_engine().query("test")`
- [ ] Logs show document count matches vault

**Deliverable:** `index_vault.py` + `.memflow_index/` directory

---

### Sub-Task 2.3: Query Engine with Relationships
**Goal:** Create retrieval function that uses Relationships.md for better context

**File:** `query_engine.py`

**Requirements:**
- Load persisted index from Sub-Task 2.2
- Parse Relationships.md to identify relevant files for query types
- Implement hybrid retrieval:
  1. Vector similarity search (semantic)
  2. Relationship-guided retrieval (explicit connections)
  3. Metadata boosting (recent files, high backlinks)
- Return top 3-5 most relevant document chunks
- Format output as structured context (not raw markdown dump)

**Key Function:**
```python
def get_context(query: str, max_tokens: int = 1000) -> dict:
    """
    Args:
        query: User's question
        max_tokens: Max context to return
    
    Returns:
        {
            "context": "Structured context from vault",
            "sources": ["file1.md", "file2.md"],
            "confidence": 0.85
        }
    """
    # 1. Load index
    # 2. Check if query matches Relationships.md patterns
    # 3. Retrieve via vector + relationships
    # 4. Format structured output
    pass
```

**Output Format Example:**
```
[MemFlow Context - Learning Decision]

JOB TIMELINE:
- Target roles: Solutions Engineer (Job/Identity/Identity.md)
- Deadline: March 2026 (Job/README For AI.md)

POSITIONING:
- Niche: Enterprise SaaS + AI security bridge
- Work style: AI orchestration, not heavy coding

CURRENT WORK:
- Local LLM Browser Automation
- Facebook OAuth integration

[End MemFlow Context]
```

**Validation:**
- [ ] Query "Should I learn LangChain?" returns Job/, Projects/, Identity/ context
- [ ] Query "Write LinkedIn post" returns Identity/Linkedin/ + Projects/ context
- [ ] Context is structured (not raw dumps)
- [ ] Source files are listed

**Deliverable:** `query_engine.py` with `get_context()` function

---

### Sub-Task 2.4: File Watcher for Auto-Updates
**Goal:** Monitor vault for changes, re-index modified files automatically

**File:** `file_watcher.py`

**Requirements:**
- Use `watchdog` library
- Watch vault directory for .md file modifications
- Debounce changes (30-second window to batch updates)
- Re-index only changed files (incremental update)
- Run in background without blocking
- Log all update operations

**Pseudo-code:**
```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

class VaultWatcher(FileSystemEventHandler):
    def __init__(self, index):
        self.pending_changes = set()
        self.last_change_time = 0
        
    def on_modified(self, event):
        if event.src_path.endswith('.md'):
            self.pending_changes.add(event.src_path)
            self.last_change_time = time.time()
    
    def check_and_process(self):
        # If idle for 30 seconds, process batch
        if time.time() - self.last_change_time > 30:
            if self.pending_changes:
                # Re-index changed files
                print(f"Updating {len(self.pending_changes)} files")
                # ... update logic
                self.pending_changes.clear()
```

**Validation:**
- [ ] Watcher starts without errors
- [ ] Editing a note triggers pending change
- [ ] After 30 seconds, re-indexing occurs
- [ ] Multiple rapid edits are batched (not N individual updates)

**Deliverable:** `file_watcher.py` that runs as background process

---

### Sub-Task 2.5: Integration Test
**Goal:** Validate end-to-end retrieval works with real queries

**File:** `test_retrieval.py`

**Test Cases:**
```python
TEST_QUERIES = [
    {
        "query": "Should I learn LangChain?",
        "expected_files": ["Job/Identity/Identity.md", "Job/README For AI.md", "Projects/Projects.md"],
        "expected_keywords": ["March 2026", "Solutions Engineer", "positioning"]
    },
    {
        "query": "Help me write a LinkedIn post about OAuth",
        "expected_files": ["Job/Identity/Linkedin/", "Projects/Personal/"],
        "expected_keywords": ["technical depth", "OAuth integration"]
    },
    {
        "query": "Should I apply to this company?",
        "expected_files": ["Job/Identity/Identity.md", "Job/README For AI.md"],
        "expected_keywords": ["H1B", "$70K", "B2B SaaS"]
    }
]
```

**Validation:**
- [ ] All 3 test queries return relevant context
- [ ] Expected files appear in sources
- [ ] Expected keywords present in context
- [ ] Retrieval completes in <200ms per query

**Deliverable:** `test_retrieval.py` with passing tests

---

## Part 3: MCP Connection Layer

### Objective
Create MCP server that exposes retrieval engine to Claude Desktop and other AI tools.

### Sub-Task 3.1: MCP Server Setup
**Goal:** Create FastAPI server implementing MCP protocol

**File:** `mcp_server.py`

**Requirements:**
- FastAPI server on localhost:8080
- Implements MCP protocol endpoints:
  - `/tools/list` - List available tools
  - `/tools/call` - Execute tool (context retrieval)
- Integrates with query_engine.py from Part 2
- Returns structured JSON responses

**Pseudo-code:**
```python
from fastapi import FastAPI
from query_engine import get_context

app = FastAPI()

@app.get("/tools/list")
def list_tools():
    return {
        "tools": [{
            "name": "get_vault_context",
            "description": "Retrieve relevant context from Obsidian vault",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }
        }]
    }

@app.post("/tools/call")
def call_tool(request: dict):
    query = request["params"]["arguments"]["query"]
    context = get_context(query)
    return {
        "content": [{
            "type": "text",
            "text": context["context"]
        }]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

**Validation:**
- [ ] Server starts: `python mcp_server.py`
- [ ] Accessible: `curl http://localhost:8080/tools/list`
- [ ] Returns tool definition JSON
- [ ] Can call tool: `curl -X POST http://localhost:8080/tools/call`

**Deliverable:** `mcp_server.py` running on port 8080

---

### Sub-Task 3.2: Claude Desktop Configuration
**Goal:** Configure Claude Desktop to use MCP server

**File:** `claude_desktop_config.json`

**Location:** 
- **Mac:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**Configuration:**
```json
{
  "mcpServers": {
    "memflow": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"],
      "env": {
        "VAULT_PATH": "/absolute/path/to/obsidian/vault"
      }
    }
  }
}
```

**Important:**
- Use ABSOLUTE paths (not relative)
- Ensure Python environment has all dependencies
- Restart Claude Desktop after config changes

**Validation:**
- [ ] Config file is in correct location
- [ ] Paths are absolute and correct
- [ ] Claude Desktop restarts without errors
- [ ] MCP server shows in Claude's available tools

**Deliverable:** `claude_desktop_config.json` + setup instructions

---

### Sub-Task 3.3: End-to-End Test in Claude Desktop
**Goal:** Verify MemFlow works in actual Claude Desktop conversation

**Test Procedure:**
1. Start MCP server: `python mcp_server.py`
2. Open Claude Desktop
3. Start new conversation
4. Ask: "Should I learn LangChain?"
5. Claude should automatically call memflow tool
6. Verify context is injected
7. Response should be personalized

**Expected Behavior:**
```
User: Should I learn LangChain?

[Claude automatically calls get_vault_context tool]
[Tool returns: Job timeline, positioning, current projects]

Claude: Yes, learn LangChain specifically because:
- Your March 2026 deadline gives you 3 months...
- It fits your Enterprise SaaS + AI security positioning...
- Builds on your OAuth and LLM automation work...
```

**Validation:**
- [ ] Claude automatically uses memflow tool (no manual prompt)
- [ ] Context from vault appears in response
- [ ] Response is personalized (mentions March 2026, positioning, etc.)
- [ ] No errors in MCP server logs

**Deliverable:** Screenshot or transcript of working conversation

---

## Deliverables Checklist

### Part 2: Retrieval Engine
- [ ] `requirements.txt` - All dependencies
- [ ] `index_vault.py` - Initial indexing
- [ ] `.memflow_index/` - Persisted index
- [ ] `query_engine.py` - Context retrieval
- [ ] `file_watcher.py` - Auto-updates
- [ ] `test_retrieval.py` - Validation tests

### Part 3: MCP Connection
- [ ] `mcp_server.py` - MCP protocol server
- [ ] `claude_desktop_config.json` - Configuration
- [ ] Setup documentation (paths, installation)
- [ ] End-to-end test proof (screenshot/transcript)

---

## Success Criteria

### Functional Requirements
✅ Query "Should I learn LangChain?" returns context from 3+ vault files  
✅ Context includes: March 2026 deadline, positioning, current projects  
✅ File watcher updates index when notes are edited  
✅ MCP server responds to Claude Desktop tool calls  
✅ End-to-end conversation shows personalized responses  

### Performance Requirements
✅ Context retrieval: <200ms per query  
✅ Index update: <60 seconds for 500-note vault  
✅ No crashes or connection failures  

### User Experience
✅ Zero manual context injection needed  
✅ Claude Desktop automatically uses vault context  
✅ Responses reference specific vault information  

---

## Implementation Notes

### Vault Path Configuration
**CRITICAL:** Replace placeholders with actual paths:
```python
VAULT_PATH = "/Users/[username]/Documents/Obsidian/MyVault"  # Mac
VAULT_PATH = "C:/Users/[username]/Documents/Obsidian/MyVault"  # Windows
```

### Testing Without Claude Desktop
If Claude Desktop isn't available, test with CLI:
```python
# test_manual.py
from query_engine import get_context

result = get_context("Should I learn LangChain?")
print(result["context"])
```

### Debugging Tips
- Check MCP server logs for tool calls
- Verify index contains Relationships.md content
- Test vector search with simple queries first
- Validate file watcher with manual edits

---

## Troubleshooting Common Issues

### Issue: ImportError for llama_index
**Solution:** Check package version
```bash
pip install llama-index==0.9.48 --force-reinstall
```

### Issue: Embedding model download fails
**Solution:** Pre-download model
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
```

### Issue: Claude Desktop doesn't see MCP server
**Solution:** 
1. Check config file location
2. Use absolute paths
3. Restart Claude Desktop completely (quit + reopen)
4. Check MCP server is running (`curl localhost:8080/tools/list`)

### Issue: Context retrieval is slow (>500ms)
**Solution:**
1. Verify embedding model is cached (not re-downloading)
2. Check vault size (>1000 notes may need optimization)
3. Add query result caching

---

## Phase 2 Extensions (Future)

After basic system works:
- [ ] Browser extension for claude.ai (web interface)
- [ ] ChatGPT integration
- [ ] Context confidence scoring
- [ ] Usage analytics (which notes retrieved most)
- [ ] Relationship validation (suggest new connections)

---

## Questions for Human Review

After implementation, confirm:
1. Does Claude Desktop automatically call memflow tool?
2. Is the retrieved context relevant and useful?
3. Does file watcher work without manual intervention?
4. Any errors in MCP server logs?
5. What's the average retrieval latency?

---

## Final Validation

Run this complete test:
1. Edit a note in vault (test file watcher)
2. Wait 30 seconds (debounce period)
3. Ask Claude: "Should I learn [new skill]?"
4. Verify response includes recently edited content
5. Check MCP logs show tool call
6. Confirm <200ms retrieval time

**If all pass: MemFlow Part 2 & 3 are DONE ✅**
