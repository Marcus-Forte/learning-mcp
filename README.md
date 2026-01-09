# Minimal MCP client and server Example.

This repository demonstrates the power and flexibility of MCP. Lan

1. In a terminal, call `uv run python -m app.server` to start MCP server. We're using HTTP streamable.

2. In another terminal, call `uv run python -m app.client`. Check `uv run -- python -m app.client -h` for options. If using ollama, make sure to serve and pull the models.

- `uv run python -m app.client --prompt "what's 9+9?" --ollama_model qwen3 --llm ollama`
- `uv run python -m app.client --prompt "what's 9+9?" --llm ollama --ollama_model deepseek-r1`
- `uv run python -m app.client --prompt "what's 9+9?" --llm google`


# Generate proto services

- `uv run python -m grpc_tools.protoc -Iproto_gen=app/proto --python_out=. --pyi_out=. --grpc_python_out=. ./app/proto/*.proto`

## Resources

- https://modelcontextprotocol.io/introduction: MCP Theory.
- https://python.langchain.com/api_reference/reference.html: Abstract object that represents various LLM models.
- https://docs.mcp-use.com/: Abstract interface to between MCP server and various LLM models, remotely or local (ollama).
- https://gofastmcp.com/servers/server: FastMCP documentation.