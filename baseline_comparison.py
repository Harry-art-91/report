"""
baseline_comparison.py
======================
BM25 基准检索 vs 当前混合检索（向量+关键词）的对比实验脚本。

运行方式：
    python baseline_comparison.py --dataset holdout
    python baseline_comparison.py --dataset development
    python baseline_comparison.py --dataset both

依赖安装：
    pip install rank-bm25 jieba
"""

import os
import sys
import csv
import json
import math
import argparse
import time
from pathlib import Path
from collections import defaultdict

# 添加项目根目录和 src 目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    """读取测试集 CSV，返回 list[dict]。
    支持 expected_keywords 字段含逗号（最后一列为 expected_type）。
    """
    cases = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return cases
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        cases.append({
            "id": parts[0],
            "query": parts[1],
            "expected_keywords": [k.strip() for k in parts[2:-1] if k.strip()],
            "expected_type": parts[-1].strip(),
        })
    return cases


def tokenize_zh(text: str) -> list[str]:
    """中文分词，优先用 jieba，不可用时退回字符级 bigram。"""
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        # 退回 bigram
        tokens = []
        for i in range(len(text) - 1):
            tokens.append(text[i : i + 2])
        return tokens or list(text)


def precision_recall_f1(retrieved_contents: list[str],
                         expected_keywords: list[str],
                         expected_type: str) -> tuple[float, float, float]:
    """计算单条查询的 Precision / Recall / F1。
    相关判定：检索到的文本包含 expected_type 或至少一个 expected_keyword。
    """
    if not retrieved_contents:
        return 0.0, 0.0, 0.0

    # Precision：检索结果中相关的比例
    relevant_count = sum(
        1 for doc in retrieved_contents
        if expected_type in doc or any(kw in doc for kw in expected_keywords)
    )
    precision = relevant_count / len(retrieved_contents)

    # Recall：expected_keywords 中被覆盖的比例
    if not expected_keywords:
        recall = 1.0
    else:
        found = set(kw for kw in expected_keywords
                    for doc in retrieved_contents if kw in doc)
        recall = len(found) / len(expected_keywords)

    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


# ──────────────────────────────────────────────
# BM25 检索器
# ──────────────────────────────────────────────

class BM25Retriever:
    """基于 rank-bm25 的 BM25 基准检索器。
    语料来自项目知识库 CSV 文件中每条记录的拼接文本。
    """

    def __init__(self, corpus_path: str, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[str] = []
        self._bm25 = None
        self._load_corpus(corpus_path)

    def _load_corpus(self, path: str):
        """从 fraud_cases_optimized.csv 加载语料。"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "请先安装依赖：pip install rank-bm25\n"
                "（可选中文分词：pip install jieba）"
            )

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_text = (
                    f"【诈骗类型】{row.get('fraud_type', '')}\n"
                    f"【场景】{row.get('scenario', '')}\n"
                    f"【手段】{row.get('modus_operandi', '')}\n"
                    f"【警示】{row.get('warning_signs', '')}\n"
                    f"【防范】{row.get('prevention_tips', '')}\n"
                    f"【关键词】{row.get('keywords', '')}\n"
                    f"【描述】{row.get('detailed_description', '')}"
                )
                self._docs.append(doc_text)

        tokenized = [tokenize_zh(d) for d in self._docs]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        print(f"  [BM25] 加载语料 {len(self._docs)} 条")

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """检索 top-k 相关文档，返回文档文本列表。"""
        tokens = tokenize_zh(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._docs[i] for i in top_indices]


# ──────────────────────────────────────────────
# 混合检索器（复用现有 RAGSystem）
# ──────────────────────────────────────────────

class HybridRetrieverWrapper:
    """包装现有 RAGSystem 中的混合检索器，提供统一接口。"""

    def __init__(self):
        from rag_system import RAGSystem
        self.rag = RAGSystem()
        print("  [Hybrid] 初始化 RAG 系统...")
        ok = self.rag._init_system()
        if not ok:
            raise RuntimeError("RAG 系统初始化失败，请先运行 `python scripts/build_kg.py`")
        print("  [Hybrid] 初始化完成")

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        docs = self.rag.retriever.invoke(query, k=k)
        return [d.page_content for d in docs]


# ──────────────────────────────────────────────
# 评估引擎
# ──────────────────────────────────────────────

class Evaluator:
    def __init__(self, k: int = 3):
        self.k = k

    def evaluate(self, retriever, test_cases: list[dict]) -> dict:
        """在测试集上运行检索，返回汇总指标和分类型指标。"""
        all_p, all_r, all_f1 = [], [], []
        type_stats: dict[str, dict] = defaultdict(lambda: {"p": [], "r": [], "f1": [], "latency": []})

        total_latency = 0.0
        for case in test_cases:
            t0 = time.time()
            docs = retriever.retrieve(case["query"], k=self.k)
            latency = time.time() - t0
            total_latency += latency

            p, r, f1 = precision_recall_f1(docs, case["expected_keywords"], case["expected_type"])
            all_p.append(p)
            all_r.append(r)
            all_f1.append(f1)

            et = case["expected_type"]
            type_stats[et]["p"].append(p)
            type_stats[et]["r"].append(r)
            type_stats[et]["f1"].append(f1)
            type_stats[et]["latency"].append(latency)

        n = len(test_cases)
        avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

        type_summary = {}
        for t, s in type_stats.items():
            type_summary[t] = {
                "count": len(s["p"]),
                "precision": round(avg(s["p"]), 4),
                "recall": round(avg(s["r"]), 4),
                "f1": round(avg(s["f1"]), 4),
                "avg_latency_ms": round(avg(s["latency"]) * 1000, 1),
            }

        return {
            "n": n,
            "avg_precision": round(avg(all_p), 4),
            "avg_recall": round(avg(all_r), 4),
            "avg_f1": round(avg(all_f1), 4),
            "avg_latency_ms": round(total_latency / n * 1000, 1) if n else 0,
            "by_type": type_summary,
        }


# ──────────────────────────────────────────────
# 报告打印
# ──────────────────────────────────────────────

def _grade(f1: float) -> str:
    if f1 >= 0.8: return "优秀 ★★★"
    if f1 >= 0.6: return "良好 ★★☆"
    if f1 >= 0.4: return "一般 ★☆☆"
    return "需优化 ☆☆☆"


def print_comparison_report(
    dataset_name: str,
    bm25_result: dict,
    hybrid_result: dict,
    test_cases: list[dict],
):
    sep = "=" * 72

    print(f"\n{sep}")
    print(f"  对比实验报告 — {dataset_name}")
    print(sep)
    print(f"  测试集大小：{bm25_result['n']} 条")
    print(f"  检索 top-k ：3\n")

    # 整体指标表
    header = f"{'指标':12}{'BM25（基准）':>16}{'混合检索（本文）':>18}{'△':>8}"
    print(header)
    print("-" * 56)
    for metric, label in [("avg_precision", "Precision"),
                           ("avg_recall",    "Recall"),
                           ("avg_f1",        "F1")]:
        bv = bm25_result[metric]
        hv = hybrid_result[metric]
        delta = hv - bv
        sign = "+" if delta >= 0 else ""
        print(f"  {label:10}{bv:>16.4f}{hv:>18.4f}{sign+f'{delta:.4f}':>8}")
    print(f"  {'延迟(ms)':10}{bm25_result['avg_latency_ms']:>16.1f}"
          f"{hybrid_result['avg_latency_ms']:>18.1f}")
    print()
    print(f"  BM25   等级：{_grade(bm25_result['avg_f1'])}")
    print(f"  混合   等级：{_grade(hybrid_result['avg_f1'])}")

    # 逐类型对比
    print(f"\n{sep}")
    print("  按诈骗类型对比（F1）")
    print(sep)
    all_types = sorted(set(list(bm25_result["by_type"]) + list(hybrid_result["by_type"])))
    print(f"  {'类型':18}{'样本':>6}{'BM25 F1':>12}{'混合 F1':>12}{'△':>8}")
    print("-" * 60)
    for t in all_types:
        bv = bm25_result["by_type"].get(t, {}).get("f1", 0.0)
        hv = hybrid_result["by_type"].get(t, {}).get("f1", 0.0)
        cnt = hybrid_result["by_type"].get(t, {}).get("count", 0)
        delta = hv - bv
        sign = "+" if delta >= 0 else ""
        # win_mark: 简单文本标记，避免 Windows 控制台编码问题
        win_mark = "better" if delta > 0 else ("worse" if delta < 0 else "same")
        print(f"  {t:18}{cnt:>6}{bv:>12.4f}{hv:>12.4f}{sign+f'{delta:.4f}':>8}  {win_mark}")

    # 失败案例分析（混合检索 F1 < 0.5 的）
    print(f"\n{sep}")
    print("  混合检索低精度案例（F1 < 0.5，前 5 条）")
    print(sep)

    # 需要重新运行一遍收集逐条结果——这里做简化，仅展示类型统计中的低分类型
    low_types = [t for t, s in hybrid_result["by_type"].items() if s["f1"] < 0.5]
    if low_types:
        print(f"  低精度诈骗类型：{', '.join(low_types)}")
        print("  建议：检查知识库中该类型的案例数量及关键词覆盖度。")
    else:
        print("  所有类型 F1 均 ≥ 0.5，检索质量良好。")

    print(f"\n{sep}")
    print("  结论")
    print(sep)
    delta_f1 = hybrid_result["avg_f1"] - bm25_result["avg_f1"]
    if delta_f1 > 0:
        print(f"  混合检索（向量+关键词）相比 BM25 基准，F1 提升 {delta_f1:+.4f}，")
        print(f"  验证了本文混合检索策略的有效性。")
    elif delta_f1 == 0:
        print("  两种方法 F1 持平，混合检索在延迟方面存在开销，需进一步优化。")
    else:
        print(f"  BM25 基准 F1 高于混合检索 {-delta_f1:.4f}，")
        print("  建议检查向量模型是否适合中文语料，或调整关键词权重策略。")
    print()


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

def run_experiment(dataset_name: str, csv_path: str, corpus_path: str, output_dir: str):
    print(f"\n{'='*72}")
    print(f"  加载测试集：{dataset_name} ({csv_path})")
    test_cases = load_csv(csv_path)
    if not test_cases:
        print("  [ERROR] 测试集为空，跳过")
        return
    print(f"  共 {len(test_cases)} 条测试用例")

    evaluator = Evaluator(k=3)

    # ── BM25 基准
    print("\n  初始化 BM25 基准检索器...")
    try:
        bm25 = BM25Retriever(corpus_path)
        print("  运行 BM25 评估...")
        bm25_result = evaluator.evaluate(bm25, test_cases)
    except ImportError as e:
        print(f"  [WARN] {e}")
        print("  BM25 跳过，请安装依赖后重新运行。")
        bm25_result = None

    # ── 混合检索
    print("\n  初始化混合检索器（向量+关键词）...")
    try:
        hybrid = HybridRetrieverWrapper()
        print("  运行混合检索评估...")
        hybrid_result = evaluator.evaluate(hybrid, test_cases)
    except Exception as e:
        print(f"  [ERROR] 混合检索初始化失败：{e}")
        hybrid_result = None

    # ── 报告
    if bm25_result and hybrid_result:
        print_comparison_report(dataset_name, bm25_result, hybrid_result, test_cases)

        # 保存 JSON
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"baseline_comparison_{dataset_name}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": dataset_name,
                "n": len(test_cases),
                "bm25": bm25_result,
                "hybrid": hybrid_result,
            }, f, ensure_ascii=False, indent=2)
        print(f"  [OK] 结果已保存：{save_path}\n")
    elif bm25_result:
        print("\n  [WARN] 仅有 BM25 结果，跳过对比。")
    elif hybrid_result:
        print("\n  [WARN] 仅有混合检索结果，跳过对比。")


def main():
    parser = argparse.ArgumentParser(description="BM25 vs 混合检索对比实验")
    parser.add_argument("--dataset", choices=["development", "holdout", "both"],
                        default="both", help="使用的测试集（默认：both）")
    parser.add_argument("--dev-csv",     default="test_dev.csv")
    parser.add_argument("--holdout-csv", default="test_holdout.csv")
    parser.add_argument("--corpus",      default="fraud_cases_optimized.csv")
    parser.add_argument("--output",      default="output")
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def p(filename):
        return os.path.join(base, filename)

    # 检查知识库文件路径
    corpus_path = os.path.join(base, "data", args.corpus)
    if not os.path.exists(corpus_path):
        print(f"[ERROR] 找不到知识库文件：{corpus_path}")
        sys.exit(1)

    if args.dataset in ("development", "both"):
        dev_csv = os.path.join(base, "data", args.dev_csv)
        if os.path.exists(dev_csv):
            run_experiment("开发集", dev_csv, corpus_path, p(args.output))
        else:
            print(f"[WARN] 开发集文件不存在：{dev_csv}")

    if args.dataset in ("holdout", "both"):
        hold_csv = os.path.join(base, "data", args.holdout_csv)
        if os.path.exists(hold_csv):
            run_experiment("保留集", hold_csv, corpus_path, p(args.output))
        else:
            print(f"[WARN] 保留集文件不存在：{hold_csv}")


if __name__ == "__main__":
    main()
