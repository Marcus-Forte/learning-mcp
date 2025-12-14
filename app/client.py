import asyncio
import mcp_use
from langchain_ollama.chat_models import ChatOllama

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_openai.chat_models import ChatOpenAI


from mcp_use import MCPAgent, MCPClient
import argparse
from dotenv import load_dotenv


async def main():

    args = argparse.ArgumentParser(description="Run MCP Agent with LLM")
    args.add_argument(
        "--llm",
        type=str,
        default="google",
        help="Select LLM backend: 'ollama' or 'google'",
    )
    args.add_argument(
        "--ollama_model",
        type=str,
        default="qwen3",
        help="Ollama model to use. Default is 'qwen3'.",
    )
    args.add_argument(
        "--prompt",
        type=str,
        help="Prompt to send to the LLM",
    )

    # Call load_dotenv() to load the environment variables from the .env file
    load_dotenv()

    mcp_use.set_debug(1)

    parsed_args = args.parse_args()
    if parsed_args.llm == "ollama":
        print("Using Ollama LLM. Make sure Ollama server is running (ollama serve).")
        llm = ChatOllama(
            model=parsed_args.ollama_model, base_url="http://localhost:11434", reasoning=False
        )
        # llm.extract_reasoning = True  # Remove thinking
    elif parsed_args.llm == "google":
        print(
            "Using Google Generative AI. Make sure you have set up the environment variables for Google API."
        )
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
    elif parsed_args.llm == "openai":
        llm = ChatOpenAI(model="o4-mini")
    else:
        raise ValueError("Invalid LLM backend specified. Use 'ollama' or 'google'.")

    # Use agent
    client = MCPClient.from_config_file("servers.json")

    # Wire the LLM to the client
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=20,
        system_prompt="You are an helpful, but angry AI assistant"
        "Use the tools provided to interact with the hardware, but do not be limited by them."
        "No need to check if LED is on or off, just turn it on or off."
        "Reply in the languagu`e the user requests.",
    )

    print(
        "\033c## AI Agent Conected! ##\n",
        end="",
    )

    model_name: str = ""
    if hasattr(llm, "model_name"):
        model_name = llm.model_name
    elif hasattr(llm, "model"):
        model_name = llm.model

    print(f"Using Model: {model_name}")
    # Interactive prompt
    if not parsed_args.prompt:
        while True:
            user_input = input("YOU: ")
            print("...")

            # Surely there is a better way to process outputs?
            try:
                result = await agent.run(user_input)
                print(result)
            except Exception as e:
                print(f"Error during agent run: {e}")

    # Give a single prompt to the agent

    try:
        # Agent Mode (Tools)
        result = await agent.run(parsed_args.prompt)
        print(
            "\n🔥 Result:",
            result,
        )

        # Normal (no tools)
        # result = llm.invoke(parsed_args.prompt).content
        # print("\n🔥 Result:", result)

        # Streamed prompt (no tools)
        # for chunk in llm.stream(parsed_args.prompt):
        #     print(chunk.content, end="", flush=True)

    except Exception as e:
        print(f"Error during agent run: {e}")

    # Always clean up running MCP sessions
    await client.close_all_sessions()


if __name__ == "__main__":
    asyncio.run(main())
