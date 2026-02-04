# MemFlow: Solution Architecture

## Overview

**MemFlow** is a 3-part system that automatically enriches AI conversations with personalized context from your Obsidian vault.

```
┌─────────────────────────────────────────────────┐
│  PART 1: RELATIONSHIP EXTRACTION (One-time)     │
│  Gemini reads vault → Creates explicit          │
│  relationship mappings                           │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  PART 2: RETRIEVAL ENGINE (Continuous)          │
│  LlamaIndex indexes vault → Retrieves relevant  │
│  context based on relationships                  │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  PART 3: CONNECTION LAYER (Cross-platform)      │
│  MCP + Extensions → Injects context into        │
│  Claude/ChatGPT/CLI tools                        │
└─────────────────────────────────────────────────┘
```

---

## Part 1: Relationship Extraction (Gemini)

### Purpose
Make implicit relationships in your vault **explicit** so retrieval systems can follow them.

### What It Does

**Input:** Your Obsidian vault (raw notes with implicit connections)

**Output:** `Relationships.md` file with explicit relationship mappings

**Example Transformation:**

**BEFORE** (implicit):
```markdown
# Identity.md
Target: Solutions Engineer roles
Timeline: March 2026

# Projects.md  
Current: AI automation, OAuth integration
```

**AFTER** (explicit relationships added):
```markdown
# Relationships.md

## Career Learning → Multi-Domain Impact

WHEN: User asks about learning new skills/technologies

PULL CONTEXT FROM:
- Job/Identity/Identity.md (target roles, positioning requirements)
- Job/README For AI.md (March 2026 deadline, visa timeline)
- Projects/Projects.md (current technical work, skill gaps)
- Job/Identity/Linkedin/ (content strategy and patterns)

RELATIONSHIPS TO EXPLAIN:
- New skill SUPPORTS "technical credibility" (Identity requirement)
- New skill CREATES LinkedIn content opportunity (Strategy pattern)
- Timeline CONSTRAINS learning completion (Must finish by Jan 2026)
- Portfolio REQUIRES demonstrable projects (For interviews)
- Current work INFORMS next logical skill progression

---

## Job Applications → Context Requirements

WHEN: User discusses job applications or career moves

PULL CONTEXT FROM:
- Job/Identity/Identity.md (target company profile, compensation floor)
- Job/README For AI.md (H1B sponsorship requirement)
- Job/Career-History/ (work experience, stories, accomplishments)
- Projects/ (portfolio to reference in applications)

RELATIONSHIPS TO EXPLAIN:
- Company MUST sponsor H1B (Visa constraint)
- Offer needed BY March 2026 (Deadline constraint)
- Roles REQUIRE demonstrated enterprise experience (Career History)
- Applications LEVERAGE portfolio projects (Projects folder)

---

## LinkedIn Content → Strategic Alignment  

WHEN: User discusses LinkedIn posts or content creation

PULL CONTEXT FROM:
- Job/Identity/Linkedin/ (content strategy, patterns, past posts)
- Job/Identity/Identity.md (positioning: technical depth, not generic)
- Projects/ (recent work to showcase)
- Job/Upskill/ (learning journeys to document)

RELATIONSHIPS TO EXPLAIN:
- Content SUPPORTS positioning strategy (Technical + business bridge)
- Projects ENABLE authentic technical content (Not theory, real work)
- Learning CREATES content opportunities (Document journey)
- Posts REINFORCE unique positioning (Enterprise SaaS + AI security)
```

### Implementation (You Have Gemini CLI)

Since you have **Gemini CLI access**, you can run this directly in your Obsidian vault folder:



**To Run:**
```bash
cd /path/to/your/obsidian/vault
gemini-cli --prompt "$(cat prompt.txt)" --context "Job/**, Projects/**" > Relationships.md
```

**Time:** 30 seconds  
**Cost:** <$0.01  
**Frequency:** Run once, update when vault structure changes significantly

### What You'll Review

After Gemini generates `Relationships.md`:
1. Check if relationships make sense
2. Add any missing connections you know about
3. Remove any wrong assumptions
4. Save the file in your vault root

---

## Part 2: Retrieval Engine (LlamaIndex)

### Purpose
Index your vault and retrieve relevant context based on queries and relationships.

### Technology Choice: LlamaIndex

**Why LlamaIndex over cognee/mem0:**
- **Simpler:** Easier setup and maintenance
- **Faster:** Sub-100ms retrieval with explicit relationships
- **Good enough:** With Gemini relationships explicit, no need for complex graph traversal
- **Well-supported:** Native Obsidian reader, active development

### What It Does

**Indexing (one-time):**
```
1. Reads all .md files in your vault
2. Creates vector embeddings (semantic search)
3. Reads Relationships.md (explicit connections)
4. Builds metadata index (modified dates, backlinks, folder structure)
5. Stores in local database
```

**Retrieval (per query):**
```
Query: "Should I learn LangChain?"
  ↓
1. Embed query (20ms - local ONNX model)
2. Vector search for semantic matches (15ms)
3. Check Relationships.md for explicit mappings (5ms)
4. Metadata boost (recent notes, highly-linked notes)
5. Return top 3-5 relevant notes
  ↓
Total: ~40-60ms
```

### Installation

```bash
# Install LlamaIndex
pip install llama-index llama-index-readers-obsidian

# Install local embedding model (faster than API)
pip install sentence-transformers

# Create memory service
# [We'll build this together in next step]
```

### Configuration

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers import ObsidianReader

# Point to your vault
vault_path = "/path/to/your/obsidian/vault"

# Load documents including Relationships.md
reader = ObsidianReader(vault_path)
documents = reader.load_data()

# Create index with local embeddings
index = VectorStoreIndex.from_documents(
    documents,
    embed_model="local:BAAI/bge-small-en-v1.5"  # Fast local model
)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What are my job search priorities?")
```

### File Watching (Auto-Update)

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class VaultWatcher(FileSystemEventHandler):
    def __init__(self, index):
        self.index = index
        self.pending_changes = set()
        
    def on_modified(self, event):
        if event.src_path.endswith('.md'):
            # Debounce: batch changes over 30 seconds
            self.pending_changes.add(event.src_path)
            
    def process_batch(self):
        if self.pending_changes:
            # Re-index only changed files
            for filepath in self.pending_changes:
                self.index.update_document(filepath)
            self.pending_changes.clear()
```

### Performance Optimizations

**Sub-100ms targets:**
- ONNX quantized embeddings: 20ms (vs 50-80ms standard)
- LRU cache for common queries: 0ms on cache hit
- Pre-fetch on textarea focus: Anticipatory loading
- Skip retrieval for simple queries (<8 words): Instant fallback

**CPU management:**
- Background re-indexing: Don't block user
- CPU throttling: Max 30% utilization during updates
- Batch file changes: 30-second inactivity window

---

## Part 3: Connection Layer (MCP + Extensions)

### Purpose
Connect the retrieval engine to all your AI tools (Claude, ChatGPT, CLI).

### Technology: Model Context Protocol (MCP)

**What is MCP:**
- Standard protocol for AI tools to access external data
- Supported natively by Claude Desktop
- Can be added to web/CLI via plugins

**Why MCP:**
- Single memory service works everywhere
- No DOM hacks or brittle browser injection
- Official support from Anthropic
- Growing ecosystem

### Implementation Options

#### **Option A: Claude Desktop (Easiest)**

Claude Desktop has native MCP support:

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "memflow": {
      "command": "python",
      "args": ["/path/to/memflow_server.py"],
      "env": {
        "VAULT_PATH": "/path/to/obsidian/vault"
      }
    }
  }
}
```

**User experience:**
```
You type in Claude Desktop: "Should I learn LangChain?"
  ↓
Claude automatically calls memflow MCP server
  ↓
Server returns relevant context from vault
  ↓
Claude responds with personalized answer
```

**Setup time:** 10 minutes  
**Works with:** Claude Desktop app only

#### **Option B: Browser Extension (Web-based)**

For claude.ai, ChatGPT, other web interfaces:

**Architecture:**
```
Browser extension intercepts textarea
  ↓
Sends query to local MCP server (localhost:8080)
  ↓
Server returns context
  ↓
Extension injects enriched prompt
  ↓
Web AI receives and responds
```

**Installation:**
1. Install browser extension (we'll build/provide)
2. Run local MCP server in background
3. Extension auto-connects to localhost

**User experience:**
```
You type in claude.ai: "Should I learn LangChain?"
  ↓
Extension detects, sends to local server
  ↓
Server returns context
  ↓
Extension adds: "[Context: March 2026 deadline, SE roles, AI projects...]"
  ↓
You see enriched prompt, can edit if needed
  ↓
Send to Claude
```

**Setup time:** 15 minutes  
**Works with:** Any web-based LLM

#### **Option C: CLI Wrapper**

For command-line AI tools:

```bash
# Instead of: claude "Should I learn LangChain?"
# Use wrapper:
memflow claude "Should I learn LangChain?"

# Wrapper enriches query, calls claude with context
```

**Setup time:** 5 minutes  
**Works with:** Any CLI tool

### Context Injection Format

**Structured injection (not raw dump):**

```
[MemFlow Context - Career Decision]

JOB TIMELINE:
- Target roles: Solutions Engineer, Implementation Consultant
- Deadline: March 2026 (3 months remaining)
- Constraint: Need H1B sponsorship

POSITIONING STRATEGY:
- Niche: Enterprise SaaS + AI security
- Differentiator: Technical + business bridge
- Focus: Demonstrable agent/automation skills

CURRENT STATE:
- Projects: AI automation, OAuth integrations
- Skills: System design, enterprise implementations
- Gap: Need agent orchestration portfolio piece

CONTENT STRATEGY:
- Platform: LinkedIn technical depth posts
- Pattern: Learning journeys with practical insights
- Recent: OAuth security, enterprise SSO patterns

[End MemFlow Context]

User Query: Should I learn LangChain?
```

**Why structured:**
- LLM can easily parse domains
- You can verify relevance at a glance
- Clearer than dumping raw note contents
- Enables better multi-domain reasoning

---

## Complete Implementation Timeline

### Week 1: Relationship Extraction
- [ ] Run Gemini on vault (30 seconds)
- [ ] Review Relationships.md output (15 minutes)
- [ ] Edit/refine relationships (15 minutes)
- [ ] Save to vault root

### Week 2: Retrieval Engine Setup
- [ ] Install LlamaIndex + dependencies (10 minutes)
- [ ] Configure vault indexing (20 minutes)
- [ ] Test retrieval queries (20 minutes)
- [ ] Set up file watcher (30 minutes)

### Week 3: Connection Layer
- [ ] Choose primary interface (Claude Desktop vs web)
- [ ] Set up MCP server (30 minutes)
- [ ] Install extension/configure (15 minutes)
- [ ] Test end-to-end flow (30 minutes)

**Total time investment: ~3-4 hours**

---

## Maintenance Requirements

### Ongoing (Automatic)
- File watcher updates index when notes change
- Cache management (automatic cleanup)
- Background re-indexing (batched)

### Periodic (Manual)
- Update Relationships.md when vault structure changes (quarterly)
- Review context injection quality (monthly)
- Adjust relevance thresholds if needed (as needed)

**Expected maintenance: <30 minutes/month**

---

## Success Metrics

### Functional
- [ ] Query latency: <100ms average
- [ ] Context relevance: >85% of injected context is useful
- [ ] False positive rate: <10% irrelevant context
- [ ] Cross-platform: Works in 2+ AI tools

### User Experience  
- [ ] Zero manual context injection needed
- [ ] Natural conversation flow maintained
- [ ] Can verify injected context before sending
- [ ] Answers are noticeably more personalized

### Technical
- [ ] File watcher triggers within 30 seconds
- [ ] Re-indexing completes in <60 seconds
- [ ] CPU usage <30% during updates
- [ ] No crashes or connection failures

---

## Fallback Strategy

**If retrieval is slow/fails:**
```python
def get_context(query, timeout=200):
    try:
        context = memflow.retrieve(query, timeout_ms=timeout)
        return context
    except TimeoutError:
        # Return cached last-good context (stale but usable)
        return memflow.get_cached_context()
    except Exception:
        # Fail gracefully, send query without enrichment
        return None
```

**User never experiences broken chat** - worst case is generic answer (same as without MemFlow).

---

## Phase 2 Enhancements (Future)

Once basic system is working:

### Advanced Features
- **Query complexity detection:** Skip retrieval for trivial questions
- **Context confidence scoring:** Only inject if >70% confidence
- **Multi-hop reasoning:** Follow relationship chains deeper
- **Temporal awareness:** Weight recent notes higher

### Integrations
- **Gmail:** Inject context when drafting emails
- **Slack:** Enrich messages with vault context  
- **Notion:** Sync relationships to external PKM
- **Voice assistants:** Works with voice-based AI

### Analytics
- **Context usage tracking:** Which notes get retrieved most
- **Relevance feedback loop:** Improve retrieval over time
- **Relationship validation:** Auto-suggest new connections

---

## Technical Stack Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Relationship extraction | Gemini 2.0 Flash | One-time relationship mapping |
| Embeddings | sentence-transformers (local) | Fast semantic search |
| Retrieval engine | LlamaIndex | Context retrieval + indexing |
| Vector store | FAISS (local) | Embedding storage |
| File watcher | watchdog (Python) | Auto-update on changes |
| Connection protocol | MCP | Standard AI integration |
| Desktop integration | Native MCP | Claude Desktop support |
| Web integration | Browser extension | claude.ai, ChatGPT support |
| CLI integration | Shell wrapper | Command-line tools |

**All components run locally** - no cloud dependencies, full privacy.

---

## Next Steps

1. **You:** Run Gemini relationship extraction on vault
2. **Us:** Review Relationships.md output together
3. **Us:** Set up LlamaIndex retrieval engine
4. **You:** Choose primary AI tool (Desktop vs web)
5. **Us:** Configure MCP/extension for chosen tool
6. **You:** Test with real queries
7. **Us:** Iterate and refine

**Ready to start with Part 1?**
