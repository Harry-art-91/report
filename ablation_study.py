"""
ablation_study.py
=================
消融实验：分析混合检索中各组件对性能的独立贡献。

实验设计（4组，对应论文"消融实验"章节）：
  A  纯向量检索（禁用关键词权重）
  B  纯关键词检索（跳过向量模型，用 BM25 代替）
  C  向量 + 关键词（本文方案，权重系数 α=1.0）
  D  向量 + 关键词（降低权重系数 α=0.5）

运行方式：
    python ablation_study.py --dataset holdout
    python ablation_study.py --dataset development
"""

import os
import sys
import json
import time
import argparse
import csv
from collections import defaultdict

# 添加项目根目录和 src 目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))


# ──────────────────────────────────────────────
# 工具（与 baseline_comparison.py 相同）
# ──────────────────────────────────────────────

def load_csv(path):
    cases = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
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


def tokenize_zh(text):
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        return [text[i:i+2] for i in range(len(text)-1)] or list(text)


def prf(docs, keywords, etype):
    if not docs:
        return 0.0, 0.0, 0.0
    rel = sum(1 for d in docs if etype in d or any(k in d for k in keywords))
    p = rel / len(docs)
    r = (len({k for k in keywords for d in docs if k in d}) / len(keywords)
         if keywords else 1.0)
    f = 2*p*r/(p+r) if p+r else 0.0
    return p, r, f


def avg(lst):
    return sum(lst)/len(lst) if lst else 0.0


# ──────────────────────────────────────────────
# 检索变体
# ──────────────────────────────────────────────

class PureVectorRetriever:
    """纯向量检索：不使用关键词分数，仅用余弦相似度排序。"""
    def __init__(self, rag):
        self.rag = rag

    def retrieve(self, query, k=3):
        # 直接用 chromadb 语义检索，绕过关键词权重
        qe = self.rag.model.encode(query)
        results = self.rag.collection.query(
            query_embeddings=[qe.tolist()],
            n_results=k
        )
        if results["documents"] and results["documents"][0]:
            return results["documents"][0][:k]
        return []


class PureBM25Retriever:
    """纯 BM25 关键词检索（同 baseline_comparison.py）。"""
    def __init__(self, corpus_path):
        from rank_bm25 import BM25Okapi
        self._docs = []
        with open(corpus_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._docs.append(
                    f"{row.get('fraud_type','')} {row.get('scenario','')} "
                    f"{row.get('modus_operandi','')} {row.get('keywords','')} "
                    f"{row.get('detailed_description','')}"
                )
        self._bm25 = BM25Okapi([tokenize_zh(d) for d in self._docs])

    def retrieve(self, query, k=3):
        scores = self._bm25.get_scores(tokenize_zh(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._docs[i] for i in top]


class HybridRetriever:
    """混合检索（向量+关键词），支持调整关键词权重系数 alpha。"""
    def __init__(self, rag, alpha=1.0):
        from hybrid_retriever import HybridRetriever as RRFHybridRetriever
        self.retriever = RRFHybridRetriever(rag.model, rag.collection, alpha=alpha)

    def retrieve(self, query, k=3):
        docs = self.retriever.invoke(query, k=k)
        return [d.page_content for d in docs]


# ──────────────────────────────────────────────
# 评估
# ──────────────────────────────────────────────

def evaluate(retriever, test_cases, k=3):
    all_p, all_r, all_f1, latencies = [], [], [], []
    type_stats = defaultdict(lambda: {"p":[],"r":[],"f1":[]})

    for case in test_cases:
        t0 = time.time()
        docs = retriever.retrieve(case["query"], k=k)
        latencies.append(time.time() - t0)
        p, r, f1 = prf(docs, case["expected_keywords"], case["expected_type"])
        all_p.append(p); all_r.append(r); all_f1.append(f1)
        et = case["expected_type"]
        type_stats[et]["p"].append(p)
        type_stats[et]["r"].append(r)
        type_stats[et]["f1"].append(f1)

    by_type = {t: {"count": len(s["p"]),
                   "precision": round(avg(s["p"]), 4),
                   "recall": round(avg(s["r"]), 4),
                   "f1": round(avg(s["f1"]), 4)}
               for t, s in type_stats.items()}

    return {
        "n": len(test_cases),
        "avg_precision": round(avg(all_p), 4),
        "avg_recall": round(avg(all_r), 4),
        "avg_f1": round(avg(all_f1), 4),
        "avg_latency_ms": round(avg(latencies)*1000, 1),
        "by_type": by_type,
    }


# ──────────────────────────────────────────────
# 报告
# ──────────────────────────────────────────────

def print_ablation_report(dataset_name, results):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  消融实验报告 — {dataset_name}")
    print(sep)

    variants = ["A. 纯向量", "B. 纯BM25", "C. 混合(α=1.0)", "D. 混合(α=0.5)"]
    keys     = ["pure_vector", "pure_bm25", "hybrid_10", "hybrid_05"]

    # 整体指标
    print(f"\n  {'变体':16}{'Precision':>12}{'Recall':>10}{'F1':>10}{'延迟ms':>10}")
    print("-" * 60)
    for v, k_ in zip(variants, keys):
        r = results.get(k_)
        if r:
            print(f"  {v:16}{r['avg_precision']:>12.4f}{r['avg_recall']:>10.4f}"
                  f"{r['avg_f1']:>10.4f}{r['avg_latency_ms']:>10.1f}")
        else:
            print(f"  {v:16}{'N/A':>12}{'N/A':>10}{'N/A':>10}{'N/A':>10}")

    # 逐类型 F1 热力表
    print(f"\n{sep}")
    print("  各类型 F1 热力表")
    print(sep)
    all_types = sorted(set(
        t for k_ in keys for t in (results.get(k_, {}).get("by_type") or {})))
    print(f"  {'类型':18}", end="")
    for v in variants:
        print(f"{v[:8]:>12}", end="")
    print()
    print("-" * (18 + 12*len(variants) + 4))
    for t in all_types:
        print(f"  {t:18}", end="")
        for k_ in keys:
            f1 = results.get(k_, {}).get("by_type", {}).get(t, {}).get("f1", float("nan"))
            if f1 != f1:  # NaN
                print(f"{'—':>12}", end="")
            else:
                bar = "█" * int(f1 * 8)
                print(f"{f1:>6.2f} {bar:<5}", end="")
        print()

    # 组件贡献分析
    print(f"\n{sep}")
    print("  组件贡献分析")
    print(sep)
    r_pv = results.get("pure_vector", {})
    r_bm = results.get("pure_bm25", {})
    r_h1 = results.get("hybrid_10", {})
    r_h5 = results.get("hybrid_05", {})

    if r_pv and r_bm and r_h1:
        lift_over_vec = r_h1["avg_f1"] - r_pv["avg_f1"]
        lift_over_bm  = r_h1["avg_f1"] - r_bm["avg_f1"]
        lift_alpha    = r_h1["avg_f1"] - r_h5["avg_f1"] if r_h5 else float("nan")
        print(f"  混合(α=1.0) vs 纯向量  ：F1 {'+'if lift_over_vec>=0 else ''}{lift_over_vec:+.4f}")
        print(f"  混合(α=1.0) vs 纯BM25  ：F1 {'+'if lift_over_bm>=0 else ''}{lift_over_bm:+.4f}")
        if lift_alpha == lift_alpha:
            print(f"  混合α=1.0  vs α=0.5   ：F1 {lift_alpha:+.4f}（关键词权重敏感性）")

    print(f"\n{sep}\n")


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

def run(dataset_name, csv_path, corpus_path, output_dir):
    print(f"\n{'='*72}")
    print(f"  加载测试集：{dataset_name}")
    test_cases = load_csv(csv_path)
    if not test_cases:
        print("  ❌ 测试集为空，跳过")
        return

    # 初始化 RAG 系统（混合和纯向量共用）
    from rag_system import RAGSystem
    rag = RAGSystem()
    print("  初始化 RAG 系统...")
    if not rag._init_system():
        print("  ❌ RAG 系统初始化失败")
        return

    results = {}

    # A. 纯向量
    print("  [A] 纯向量检索...")
    results["pure_vector"] = evaluate(PureVectorRetriever(rag), test_cases)

    # B. 纯 BM25
    try:
        print("  [B] 纯 BM25 检索...")
        results["pure_bm25"] = evaluate(PureBM25Retriever(corpus_path), test_cases)
    except ImportError:
        print("  [B] 跳过（未安装 rank-bm25）")

    # C. 混合 α=1.0（本文方案）
    print("  [C] 混合检索 α=1.0...")
    results["hybrid_10"] = evaluate(HybridRetriever(rag, alpha=1.0), test_cases)

    # D. 混合 α=0.5
    print("  [D] 混合检索 α=0.5...")
    results["hybrid_05"] = evaluate(HybridRetriever(rag, alpha=0.5), test_cases)

    print_ablation_report(dataset_name, results)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"ablation_{dataset_name}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_name, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"  ✅ 消融结果已保存：{save_path}\n")


def main():
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument("--dataset", choices=["development", "holdout", "both"],
                        default="holdout")
    parser.add_argument("--dev-csv",     default="test_dev.csv")
    parser.add_argument("--holdout-csv", default="test_holdout.csv")
    parser.add_argument("--corpus",      default="fraud_cases_optimized.csv")
    parser.add_argument("--output",      default="output")
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    p = lambda f: os.path.join(base, f)
    
    # 检查知识库文件路径
    corpus = os.path.join(base, "data", args.corpus)
    if not os.path.exists(corpus):
        print(f"❌ 找不到知识库文件：{corpus}")
        sys.exit(1)

    if args.dataset in ("development", "both"):
        dev_csv = os.path.join(base, "data", args.dev_csv)
        if os.path.exists(dev_csv):
            run("开发集", dev_csv, corpus, p(args.output))
        else:
            print(f"⚠️  开发集文件不存在：{dev_csv}")
    
    if args.dataset in ("holdout", "both"):
        hold_csv = os.path.join(base, "data", args.holdout_csv)
        if os.path.exists(hold_csv):
            run("保留集", hold_csv, corpus, p(args.output))
        else:
            print(f"⚠️  保留集文件不存在：{hold_csv}")


if __name__ == "__main__":
    main()
