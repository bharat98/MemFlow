# MemFlow

MemFlow is a local memory layer that enriches AI conversations with your Obsidian context. It extracts relationships, indexes your vault, and injects relevant context into tools like Claude Desktop via MCP.

**Core goals**
- Zero manual context injection
- Multi-domain reasoning across your vault
- Fast, local, privacy‑first retrieval

**Architecture (3 parts)**
1. Relationship extraction (`Relationships.md`)
2. Retrieval engine (LlamaIndex + local embeddings)
3. Connection layer (MCP server, optional browser/CLI wrappers)

## Quick Start
1. Set your vault path:
   - `VAULT_PATH=/path/to/your/vault`
2. Build the index:
   - `src/venv/bin/python src/index_vault.py --rebuild`
3. Run the MCP server:
   - `src/venv/bin/python src/mcp_server.py`

## Vault Scoping (memflow_config.json)
MemFlow uses `memflow_config.json` to decide **which vault content is indexed**. This keeps results high‑quality and reduces noise.

**Personal overrides**
- The repo `memflow_config.json` is a **safe template**.
- Create `memflow_config.local.json` (ignored by git) for your personal folders/files. It overrides the template.

**Root path for all config entries**
- All paths are **relative to the vault root** (`VAULT_PATH`).
- Example vault root: `/path/to/vault`.

**Config fields**
- `include_dirs`: Top‑level folders in the vault to include.
- `include_files`: Specific files to include (relative to vault root).
- `exclude_dirs`: Folder paths to exclude (relative to vault root).

**Behavior rules**
- If `include_dirs` is set, only those top‑level folders are indexed.
- `include_files` always wins, even if outside `include_dirs`.
- `exclude_dirs` removes any path under that folder.

**Example**
```json
{
  "include_dirs": ["FolderA", "FolderB"],
  "include_files": ["FolderB/Example.md"],
  "exclude_dirs": ["FolderB/Archive", "FolderA/Old"]
}
```

After any config change, rebuild the index:
```bash
src/venv/bin/python src/index_vault.py --rebuild
```

## Notes
- `Relationships.md` should live at the **vault root**.
- The `Relationships.md` in this repo is a **sanitized template**. Create your real file in the vault root (not inside the repo).
- The system runs fully local; no cloud dependencies are required.
