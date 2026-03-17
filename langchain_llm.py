"""
LangChain LLM 封装
-------------------

目的：用 LangChain 封装通义千问 API，提供一个简洁的
    generate_answer(prompt: str, temperature: float) -> str
接口，便于在 RAGSystem 中替换直接的 dashscope 调用。
"""

import os

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage


def _build_client() -> ChatTongyi:
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError("DASHSCOPE_API_KEY not configured")

    # ChatTongyi 会自动从环境变量读取 key
    return ChatTongyi(model="qwen-turbo")


def generate_answer(prompt: str, temperature: float = 0.3) -> str:
    """
    使用 LangChain + 通义千问生成回答。
    """
    client = _build_client()
    # 为安全起见，仅传入单轮 user 消息
    resp = client.invoke(
        [HumanMessage(content=prompt)],
        temperature=temperature,
        max_tokens=1024,
    )
    # ChatTongyi 返回的是一个 Message 对象
    return resp.content or ""

