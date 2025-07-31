# server.py
from mcp.server.fastmcp import FastMCP
import mcp.types as types

from dataclasses import dataclass
import aiofiles
import asyncio

from grpc_client.blinky_grpc_client import BlinkyClient

async def main():
    # Stateless server (no session persistence, no sse stream with supported client)
    mcp = FastMCP("MCPServer", stateless_http=True, json_response=True)
    blinky_client = BlinkyClient()

    @mcp.tool()
    def turn_led_on(line:int, on:bool)-> str:
        blinky_client.SetLedOn(line, on)
        return "Success"
    
    @mcp.tool()
    def is_led_on(line: int) -> bool:
        return blinky_client.IsLedOn(line)

    # Add an addition tool
    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers and add 10 for debugging"""
        print(f"Adding {a} and {b}")
        return a + b + 10
    
    @mcp.resource("data://config")
    def get_config() -> dict:
        return {
            "version": "0.1"
        }
    
    # This particular resource template hasn't been working with agents.
    @mcp.resource("file://{filename}/")
    async def read_resource_file(filename) -> str:
        print("Get file!")
        try:
            async with aiofiles.open(f"resources/{filename}", mode="r") as f:
                content = await f.read()
            return content
        except FileNotFoundError:
            return "File not found!"

    print("Starting MCP server...")
    await mcp.run_streamable_http_async()

if __name__ == "__main__":
    asyncio.run(main())