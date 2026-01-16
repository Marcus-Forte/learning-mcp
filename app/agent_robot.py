import asyncio
import argparse

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage

from .helpers import _extract_response_text, _extract_image_base64_from_response

mcp_servers = {"robot": {"transport": "streamable_http", "url": "http://localhost:8000/mcp"}}


async def main():

    args = argparse.ArgumentParser(description="Run MCP Agent with LLM")
    args.add_argument(
        "--llm",
        type=str,
        default="openai",
        help="Select LLM backend: 'ollama', 'google', or 'openai'",
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

    parsed_args = args.parse_args()
    if parsed_args.llm == "ollama":
        print("Using Ollama LLM. Make sure Ollama server is running (ollama serve).")
        llm = ChatOllama(
            model=parsed_args.ollama_model,
            base_url="http://192.168.3.188:11434",
            reasoning=False,
        )
        # llm.extract_reasoning = True  # Remove thinking
    elif parsed_args.llm == "google":
        print(
            "Using Google Generative AI. Make sure you have set up the environment variables for Google API."
        )
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
    elif parsed_args.llm == "openai":
        llm = ChatOpenAI(model="gpt-5-nano")
    else:
        raise ValueError("Invalid LLM backend specified. Use 'ollama' or 'google'.")

    client = MultiServerMCPClient(mcp_servers)
    tools = await client.get_tools()

    camera_tool = next((tool for tool in tools if tool.name == "get_camera_frame"), None)
    if camera_tool is None:
        raise RuntimeError("get_camera_frame tool not found")

    @tool("grab_and_describe_camera_frame", description="Grab a camera frame and describe it.")
    async def describe_camera_frame(query: str) -> str:
        image_payload = await camera_tool.ainvoke({})
        image_b64 = _extract_image_base64_from_response(image_payload)
        if not image_b64:
            raise RuntimeError("No image_base64 found in camera payload")

        message = HumanMessage(
            content=[
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ]
        )
        response = llm.invoke([message])
        return getattr(response, "content", str(response))

    tools.append(describe_camera_frame)
    agent = create_agent(model=llm, tools=tools, debug=True)

    SYSTEM_RULES = """
    You control a real robot.

    Loop policy:
    1) Always call `grab_and_describe_camera_frame` first to see the world.
    2) Then choose ONE action: move_forward, turn, or stop.
    3) Prefer small moves (<= 0.3m) and small turns (<= 30deg).
    4) If unsure or something looks unsafe, call stop.
    5) If identical square tiles are detected in front of the robot stop the loop.

    Goal: navigate safely while following the user’s objective.
    """

    messages = [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": parsed_args.prompt},
    ]

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

    # Give a single prompt to the agent
    try:
        response = await agent.ainvoke({"messages": messages})
        print(
            "\n🔥 Result:",
            _extract_response_text(response),
        )

    except Exception as e:
        print(f"Error during agent run: {e}")

    if hasattr(client, "aclose"):
        await client.aclose()
    elif hasattr(client, "close"):
        close_result = client.close()
        if asyncio.iscoroutine(close_result):
            await close_result


if __name__ == "__main__":
    asyncio.run(main())
