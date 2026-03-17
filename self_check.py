"""
自检脚本：用于毕业设计交付/答辩前的一键检查。

运行：
  python scripts/self_check.py

检查项：
  - 数据文件是否存在
  - 向量库是否可用、文档数是否 > 0
  - Embedding 模型路径是否可用
  - LLM 模式与 API Key 配置提示
"""

import os
import sys

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def _ok(msg: str):
    print(f"[OK] {msg}")


def _warn(msg: str):
    print(f"[WARN] {msg}")


def _err(msg: str):
    print(f"[ERROR] {msg}")


def main() -> int:
    print("=" * 72)
    print("Anti-fraud RAG self check")
    print("=" * 72)

    required_files = [
        os.path.join(PROJECT_ROOT, "data", "fraud_cases_optimized.csv"),
        os.path.join(PROJECT_ROOT, "data", "test_dev.csv"),
        os.path.join(PROJECT_ROOT, "data", "test_holdout.csv"),
    ]
    optional_files = [
        os.path.join(PROJECT_ROOT, "data", "fraud_policies.txt"),
    ]

    for p in required_files:
        if os.path.exists(p):
            _ok(f"Found: {os.path.relpath(p, PROJECT_ROOT)}")
        else:
            _err(f"Missing: {os.path.relpath(p, PROJECT_ROOT)}")

    for p in optional_files:
        if os.path.exists(p):
            _ok(f"Found (optional): {os.path.relpath(p, PROJECT_ROOT)}")
        else:
            _warn(f"Missing (optional): {os.path.relpath(p, PROJECT_ROOT)}")

    llm_mode = (os.getenv("LLM_MODE") or "auto").strip().lower()
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if llm_mode not in ("auto", "online", "offline"):
        _warn(f"LLM_MODE={llm_mode!r} invalid; using 'auto' behavior in code")
    else:
        _ok(f"LLM_MODE={llm_mode}")

    if api_key and api_key.startswith("sk-"):
        _ok("DASHSCOPE_API_KEY configured")
    else:
        _warn("DASHSCOPE_API_KEY not configured (offline fallback will be used in auto/offline mode)")

    try:
        from src.rag_system import RAGSystem
        rag = RAGSystem()
        if not rag._init_system():
            _err("RAGSystem init failed")
            return 1
        _ok("RAGSystem init ok")

        if not rag.collection or rag.collection.count() <= 0:
            _err("Vector store empty; please run: python scripts/build_kg.py")
            return 1
        _ok(f"Vector store docs: {rag.collection.count()}")

    except Exception as e:
        _err(f"Self check failed: {e}")
        return 1

    print("-" * 72)
    _ok("Self check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

