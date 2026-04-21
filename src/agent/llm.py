import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

GROQ_MODEL = "llama-3.3-70b-versatile"


def get_llm(temperature: float = 0.1) -> ChatGroq:
    """
    Build and return a ChatGroq instance using project settings.

    The model served by Groq runs on quantized hardware (LPU) by default —
    no additional quantization configuration is required.

    Args:
        temperature: Sampling temperature (default 0.1 for deterministic responses).

    Returns:
        ChatGroq: Configured LLM ready for use in the ReAct agent.
    """
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=os.environ["GROQ_API_KEY"],
        temperature=temperature,
    )
