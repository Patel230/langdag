# Migrating from LangGraph to LangDAG

LangDAG provides first-class tooling to import your existing LangGraph conversation history into a LangDAG SQLite database. The migration preserves message content, roles, tool calls, tool results, timestamps, and token counts.

---

## Overview

The migration is a two-step process:

1. **Export** your LangGraph data (SQLite or PostgreSQL) to a JSON file using the `langgraph-export` Python tool — or skip this step and import directly from a LangGraph SQLite database.
2. **Import** into LangDAG using the `langdag import langgraph` CLI command.

---

## Step 1 — Export from LangGraph

### Option A: Direct SQLite import (no Python required)

If your LangGraph app uses a SQLite checkpoint database you can import it directly — no export step needed:

```bash
langdag import langgraph --sqlite /path/to/langgraph.db --output langdag.db
```

### Option B: Export via the Python tool

Use the `langgraph-export` Python package (located in [`tools/langgraph-export/`](../tools/langgraph-export/)) to read from SQLite or PostgreSQL and produce a portable JSON file.

**Install:**

```bash
# SQLite support only
pip install ./tools/langgraph-export

# With PostgreSQL support
pip install "./tools/langgraph-export[postgres]"
```

**Export from SQLite:**

```bash
langgraph-export --sqlite /path/to/langgraph.db --output export.json
```

**Export from PostgreSQL:**

```bash
langgraph-export --postgres "postgresql://user:pass@host/db" --output export.json
```

**Python API** (e.g. from an `InMemorySaver` in a script or test):

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_export import LangGraphExporter

saver = InMemorySaver()
# ... populate saver by running your graphs ...

exporter = LangGraphExporter.from_memory(saver)
exporter.export().save("export.json")
```

---

## Step 2 — Import into LangDAG

### From a JSON export file

```bash
langdag import langgraph --file export.json --output langdag.db
```

### From a LangGraph SQLite database (direct)

```bash
langdag import langgraph --sqlite /path/to/langgraph.db --output langdag.db
```

### Useful flags

| Flag | Description |
|------|-------------|
| `--output <path>` | Target LangDAG SQLite database (created if it does not exist). Defaults to the configured database. |
| `--dry-run` | Preview what would be imported without writing anything. |
| `--skip-existing` | Skip threads already present in the target database (matched by original thread ID). |

### Preview before importing

```bash
langdag import langgraph --file export.json --dry-run
```

### Incremental / repeated imports

Use `--skip-existing` to safely re-run an import without creating duplicate threads:

```bash
langdag import langgraph --file export.json --output langdag.db --skip-existing
```

---

## What gets migrated

| LangGraph concept | LangDAG concept |
|-------------------|-----------------|
| Thread | Conversation tree (root node + children) |
| `HumanMessage` | `user` node |
| `AIMessage` | `assistant` node |
| `ToolMessage` | `tool_result` node |
| `SystemMessage` | System prompt on the root node |
| Tool calls | `tool_call` child nodes under the assistant node |
| Message timestamps | Node `created_at` |
| Token counts | `tokens_in` / `tokens_out` on assistant nodes |
| Thread metadata | `metadata.thread_metadata` on the root node |

Each imported node carries a `metadata` field with:

```json
{
  "source": "langgraph",
  "original_thread_id": "<thread-uuid>",
  "original_message_id": "<message-id>"
}
```

This metadata is used by `--skip-existing` and is also available for your own queries.

---

## Programmatic import (Go)

If you are building a Go application and want to trigger the import from code rather than the CLI, you can use the `internal/migrate/langgraph` package directly:

```go
import (
    "context"

    lgmigrate "github.com/langdag/langdag/internal/migrate/langgraph"
    "github.com/langdag/langdag/pkg/langdag"
)

client, _ := langdag.New(langdag.Config{StoragePath: "langdag.db", /* ... */})
defer client.Close()

result, err := lgmigrate.ImportFromFile(
    context.Background(),
    "export.json",
    client.Storage(),
    lgmigrate.ImportOptions{
        SkipExisting: true,
        Progress: func(i, total int, threadID string) {
            fmt.Printf("[%d/%d] %s\n", i, total, threadID)
        },
    },
)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Imported %d threads, %d messages\n",
    result.ThreadsImported, result.MessagesImported)
```

---

## Troubleshooting

**"cannot access file" / "cannot access SQLite database"**
The path does not exist or is not readable. Check the path and file permissions.

**Duplicate threads after re-import**
Run with `--skip-existing`. Threads are matched by their original LangGraph thread ID stored in node metadata.

**Empty threads not imported**
Threads that contain only system messages, or no messages at all, are silently skipped.

**Tool call content appears as JSON**
Tool calls are stored as JSON objects in node content: `{"name": "...", "input": {...}}`. This is the expected LangDAG representation.
