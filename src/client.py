import asyncio
from langchain_ollama.chat_models import ChatOllama
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from mcp_use import MCPAgent, MCPClient
from fastmcp import Client
import argparse
from dotenv import load_dotenv

async def main():
    
    args = argparse.ArgumentParser(description="Run MCP Agent with LLM")
    args.add_argument("--llm", type=str, default="google", help="Select LLM backend: 'ollama' or 'google'")
    args.add_argument("--ollama_model", type=str, default="qwen3", help="Ollama model to use. Default is 'qwen3'.")
    args.add_argument("--prompt", type=str, required=True, help="Prompt to send to the LLM")
    
    # Call load_dotenv() to load the environment variables from the .env file
    load_dotenv()

    parsed_args = args.parse_args()
    if parsed_args.llm == "ollama":
        print("Using Ollama LLM. Make sure Ollama server is running (ollama serve).")
        llm = ChatOllama(model=parsed_args.ollama_model, base_url="http://localhost:11434")
        llm.extract_reasoning = True # Remove thinking
    elif parsed_args.llm == "google":
        print("Using Google Generative AI. Make sure you have set up the environment variables for Google API.")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    else:
        raise ValueError("Invalid LLM backend specified. Use 'ollama' or 'google'.")

    # Test tools / resource direct query
    fast_client = Client("http://localhost:8000/mcp") 
    async with fast_client:
        res = await fast_client.list_resources()
        print(res)
        t_res = await fast_client.list_resource_templates()
        print(t_res)
        file_res = await fast_client.read_resource("file://some_resource.txt")
        print(file_res)
     
    # Use agent
    client = MCPClient.from_config_file("servers.json")

    # Wire the LLM to the client
    agent = MCPAgent(llm=llm, client=client, max_steps=20)
    
    # Give prompt to the agent
    try:
        # Agent Mode
        result = await agent.run(parsed_args.prompt)
        print("\n🔥 Result:", result)
        
        # Simple prompt
        # result = llm.invoke(parsed_args.prompt).content
        # print("\n🔥 Result:", result)

        # Streamed prompt
        # for chunk in llm.stream(parsed_args.prompt):
        #     print(chunk.content, end="", flush=True)

    except Exception as e:
        print(f"Error during agent run: {e}")

    # Always clean up running MCP sessions
    await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(main())