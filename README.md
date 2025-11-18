# Minimal MCP client and server Example.

This repository demonstrates the power and flexibility of MCP.

1. In a terminal, call `python src/server.py` to start MCP server. We're using HTTP streamable.

2. In another terminal, call `python src/client.py`. Check `python client.py -h` for options. If using ollama, make sure to serve and pull the models.

- `python src/client.py --prompt "what's 9+9?" --ollama_model qwen3 --llm ollama`
- `python src/client.py --prompt "what's 9+9?" --llm ollama --ollama_model deepseek-r1`
- `python src/client.py --prompt "what's 9+9?" --llm google`

## Resources

- https://modelcontextprotocol.io/introduction: MCP Theory.
- https://python.langchain.com/api_reference/reference.html: Abstract object that represents various LLM models.
- https://docs.mcp-use.com/: Abstract interface to between MCP server and various LLM models, remotely or local (ollama).

- https://gofastmcp.com/servers/server: FastMCP documentation.
