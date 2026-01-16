import asyncio
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import create_agent
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
import argparse
from dotenv import load_dotenv

from .helpers import _extract_response_text

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
    args.add_argument(
        "--speech-to-text",
        action="store_true",
        help="Use microphone speech input instead of typing",
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

    recognizer = None
    microphone = None
    if parsed_args.speech_to_text:
        import speech_recognition as sr  # type: ignore

        recognizer = sr.Recognizer()
        try:
            microphone = sr.Microphone()
        except Exception as exc:  # noqa: PERF203
            raise SystemExit(
                "Microphone not available. Ensure PyAudio is installed and a default input device exists."
            ) from exc

    # Use agent
    client = MultiServerMCPClient(mcp_servers)
    tools = await client.get_tools()

    # Wire the LLM to the client
    agent = create_agent(model=llm, tools=tools)

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
            if parsed_args.speech_to_text and recognizer and microphone:
                try:
                    with microphone as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        print("Speak now...")
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    user_input = recognizer.recognize_google(audio)
                    print(f"YOU (voice): {user_input}")
                except Exception as e:  # noqa: PERF203
                    print(f"Speech capture failed: {e}")
                    continue
            else:
                user_input = input("YOU: ")
                print("...")

            try:
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": user_input},
                    ]
                )
                result = await agent.ainvoke({"messages": [message]})
                print(_extract_response_text(result))
            except Exception as e:
                print(f"Error during agent run: {e}")

    # Give a single prompt to the agent
    message = HumanMessage(
        content=[
            {"type": "text", "text": parsed_args.prompt},
        ]
    )

    try:
        # Agent Mode (Tools)
        result = await agent.ainvoke({"messages": [message]})
        print(
            "\n🔥 Result:",
            _extract_response_text(result),
        )

    except Exception as e:
        print(f"Error during agent run: {e}")


if __name__ == "__main__":
    asyncio.run(main())
