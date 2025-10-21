# MCP-R: Meta MCP Server with Semantic Tool Discovery

A meta-level Model Context Protocol (MCP) server that provides semantic search and automatic discovery of MCP tools across 306+ registered servers.

## Overview

MCP-R acts as a registry and orchestrator for MCP servers, allowing LLMs to dynamically discover and execute tools from any registered MCP server through natural language queries.

## Architecture

### High-Level Flow

```
User Query (Natural Language)
    ↓
  [LLM]
    ↓
Calls: search_mcp_tools(query="search GitHub repositories")
    ↓
[Meta-MCP Server - FAISS Semantic Search]
    ↓
Returns: [
  {
    "server": "@smithery-ai/github",
    "tool": "search_repositories",
    "description": "Search for GitHub repositories...",
    "inputSchema": {...}
  },
  ...
]
    ↓
  [LLM] (picks best tool)
    ↓
Calls: execute_mcp_tool(
    server="@smithery-ai/github",
    tool="search_repositories",
    arguments={"query": "machine learning", "per_page": 10}
)
    ↓
[Meta-MCP Server]
  ├─ Auto-connect to @smithery-ai/github (if not connected)
  ├─ Connection pool management
  └─ Execute tool
    ↓
Returns: Actual results from GitHub API
    ↓
  [LLM]
```

## Core Components

### 1. Semantic Search Backend (FAISS + BGE-M3)

- **Vector Database**: FAISS (Facebook AI Similarity Search)
  - Index Type: `IndexFlatIP` (Inner Product for normalized vectors)
  - Dimensions: 1024
  - Scale: 306 servers, 1000+ tools
- **Embeddings**: BGE-M3 (BAAI General Embedding)
  - Model: `BAAI/bge-m3`
  - Languages: 100+ languages supported
  - Context Length: Up to 8192 tokens
  - Quality: State-of-the-art for multilingual retrieval
  - Special Features: Dense + sparse + multi-vector retrieval
- **Index**: `tool_descriptions.ndjson` → BGE-M3 embeddings → FAISS index
- **Search Strategy**: Semantic similarity search with metadata filtering
- **Query Enhancement**: Instruction-based prompting for better retrieval

### 2. Meta-MCP Server Tools

#### Tool 1: `search_mcp_tools`

Search for MCP tools across 306+ servers using semantic search.

**Input Schema:**
```json
{
  "query": "string - Natural language description of what you need",
  "limit": "number - Max results to return (default: 5)",
  "filters": {
    "server": "string - (optional) Filter by server name",
    "tags": "array - (optional) Filter by tags"
  }
}
```

**Output:**
```json
[
  {
    "server": "string - Qualified server name",
    "tool": "string - Tool name",
    "description": "string - Tool description",
    "inputSchema": "object - JSON schema for tool inputs",
    "similarity_score": "number - Relevance score (0-1)"
  }
]
```

#### Tool 2: `execute_mcp_tool`

Execute a tool from any registered MCP server with automatic connection management.

**Input Schema:**
```json
{
  "server": "string - Qualified server name (e.g., '@smithery-ai/github')",
  "tool": "string - Tool name (e.g., 'search_repositories')",
  "arguments": "object - Tool-specific arguments"
}
```

**Output:**
```json
{
  "status": "success | error",
  "result": "any - Tool execution result",
  "server": "string - Server that executed the tool",
  "tool": "string - Tool that was executed",
  "error": "string - (optional) Error message if failed"
}
```

### 3. Connection Pool Manager

- **Purpose**: Maintain persistent connections to MCP servers
- **Strategy**: Lazy initialization - connect only when needed
- **Reuse**: Keep connections alive for repeated tool calls
- **Cleanup**: Automatic cleanup of idle connections
- **Implementation**: Uses `MCPManager.build_client()` from existing codebase

### 4. Tool Indexing Pipeline

```
tool_descriptions.ndjson (Source Data)
    ↓
[BGE-M3 Embedding Generation]
  For each tool, create embedding from:
  - Tool name
  - Tool description
  - Server name
  - Input schema summary
  - Combined text with instruction prefix
    ↓
  Query instruction: "Represent this sentence for searching relevant passages: "
  Document instruction: (none for BGE-M3)
    ↓
[Normalize Embeddings]
  L2 normalization for cosine similarity
    ↓
[FAISS Index Creation]
  - Index Type: faiss.IndexFlatIP (Inner Product on normalized vectors = cosine similarity)
  - Dimensions: 1024
  - Metadata store: Separate JSON file
    ↓
Saved Artifacts:
  - tools.faiss (FAISS index)
  - tools_metadata.json (Server name, tool name, description, inputSchema)
  - embeddings_config.json (Model name, version, dimensions)
```

## Project Structure

```
MCP-R/
├── README.md                           # This file
├── MCP_INFO_MGR/                       # MCP information management
│   ├── tool_descriptions.ndjson        # All tools from 306 servers
│   ├── remote_server_configs.json     # Server connection configs
│   ├── reachability_ok_servers.ndjson  # Reachable servers list
│   │
│   ├── fetch_tool_descriptions.py      # Fetch tools from servers
│   ├── build_search_index.py           # Build FAISS index (TODO)
│   ├── mcp_registry_server.py          # Meta-MCP server (TODO)
│   │
│   └── semantic_search/                # Semantic search components
│       ├── faiss_backend.py            # FAISS indexing & search (TODO)
│       └── embeddings.py               # Embedding generation (TODO)
│
└── Orchestrator/                       # MCP orchestration framework
    └── mcpuniverse/
        └── mcp/
            ├── client.py               # MCP client implementation
            └── manager.py              # MCP manager with connection handling
```

## Data Flow

### 1. Indexing Phase (One-time / Periodic Update)

```bash
# Step 1: Fetch tool descriptions from all servers
python MCP_INFO_MGR/fetch_tool_descriptions.py

# Step 2: Build FAISS search index
python MCP_INFO_MGR/build_search_index.py \
    --input MCP_INFO_MGR/tool_descriptions.ndjson \
    --output MCP_INFO_MGR/semantic_search/
```

### 2. Runtime Phase (LLM Interaction)

1. **LLM queries for tools**: Calls `search_mcp_tools(query="...")`
2. **Semantic search**: FAISS finds relevant tools by similarity
3. **LLM selects tool**: Analyzes results and picks best match
4. **Execute tool**: Calls `execute_mcp_tool(server, tool, args)`
5. **Auto-connect**: Meta-server connects to target MCP server (if needed)
6. **Proxy execution**: Meta-server executes tool and returns result
7. **LLM processes result**: Uses the data to respond to user

## Key Features

### Automatic Discovery
- LLMs can discover tools without hardcoded knowledge
- Natural language queries match to technical tool descriptions
- Reduces need for manual tool selection

### Connection Pooling
- Reuses connections across multiple tool calls
- Reduces latency for repeated operations
- Efficient resource management

### Scalability
- Currently indexes 306 servers with 1000+ tools
- FAISS can scale to billions of vectors
- Fast search even with large registries

### Extensibility
- Easy to add new MCP servers to the registry
- Support for metadata filtering (tags, categories)
- Hybrid search (semantic + keyword) possible

## Technology Stack

- **Vector Search**: FAISS (Meta AI)
  - Index Type: `IndexFlatIP` for cosine similarity
  - Optimized for fast similarity search
- **Embeddings**: BGE-M3 (`BAAI/bge-m3`)
  - Framework: Sentence Transformers
  - Dimensions: 1024
  - Multilingual: 100+ languages
  - Best-in-class for multilingual retrieval tasks
- **MCP Protocol**: `mcp==1.9.4`
- **Connection Management**: Custom MCPManager with pooling
- **Data Format**: NDJSON (Newline-delimited JSON)
- **Language**: Python 3.11+
- **Dependencies**:
  - `faiss-cpu` or `faiss-gpu`: Vector similarity search
  - `sentence-transformers`: BGE-M3 model loading and inference
  - `torch`: Deep learning backend for embeddings

## Use Cases

### 1. Dynamic Tool Selection
```
User: "I need to analyze stock prices for Tesla"
LLM: search_mcp_tools("stock price analysis")
     → finds yfinance tool
     execute_mcp_tool("yfinance", "get_stock_data", {"ticker": "TSLA"})
```

### 2. Multi-Tool Workflows
```
User: "Search GitHub for React repos and create an issue"
LLM: search_mcp_tools("search github repositories")
     → execute_mcp_tool("@smithery-ai/github", "search_repositories", ...)
     search_mcp_tools("create github issue")
     → execute_mcp_tool("@smithery-ai/github", "create_issue", ...)
```

### 3. Capability Discovery
```
User: "What can you help me with regarding weather?"
LLM: search_mcp_tools("weather forecasting")
     → Returns available weather-related tools across all servers
```

## Implementation Status

- [x] MCP client implementation
- [x] MCP connection manager
- [x] Tool description fetching from 306 servers
- [x] Fix asyncio task isolation issues
- [ ] FAISS semantic search backend
- [ ] Embedding generation pipeline
- [ ] Build search index script
- [ ] Meta-MCP server with search_mcp_tools
- [ ] Meta-MCP server with execute_mcp_tool
- [ ] Connection pool implementation
- [ ] End-to-end testing with LLM

## Configuration

### Environment Variables

```bash
# Required for Smithery servers
SMITHERY_API_KEY=your_api_key_here
```

### Server Configuration

Server configurations are stored in `MCP_INFO_MGR/remote_server_configs.json`:

```json
{
  "@smithery-ai/github": {
    "streamable_http": {
      "url": "https://server.smithery.ai/@smithery-ai/github/mcp",
      "headers": {
        "Authorization": "Bearer {{SMITHERY_API_KEY}}"
      }
    }
  }
}
```

## Future Enhancements

- [ ] Hybrid search (semantic + keyword BM25)
- [ ] Tool usage analytics and popularity ranking
- [ ] Automatic tool categorization/tagging
- [ ] Multi-language support for tool descriptions
- [ ] WebUI for browsing available tools
- [ ] Rate limiting and quota management
- [ ] Tool versioning and compatibility tracking
- [ ] Caching layer for frequently used tools

## Contributing

This is a research project exploring meta-level MCP orchestration and semantic tool discovery.

## License

[To be determined]