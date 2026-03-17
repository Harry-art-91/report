"""
tune_alpha.py
=============
在开发集上搜索 RRF 的最优 alpha，然后在保留集上报告最终结果。

运行：
    python tune_alpha.py
    python tune_alpha.py --dev-csv test_cases_development_v2.csv --hold-csv test_cases_holdout_v2.csv
"""

import os, sys, csv, argparse, json
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# ── 工具函数 ──────────────────────────────────────────────

def load_csv(path):
    cases = []
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 4:
            continue
        cases.append({
            'id': parts[0],
            'query': parts[1],
            'expected_keywords': [k.strip() for k in parts[2:-1] if k.strip()],
            'expected_type': parts[-1].strip(),
        })
    return cases


def prf(docs, keywords, etype):
    if not docs:
        return 0.0, 0.0, 0.0
    rel = sum(1 for d in docs if etype in d or any(k in d for k in keywords))
    p = rel / len(docs)
    r = (len({k for k in keywords for d in docs if k in d}) / len(keywords)
         if keywords else 1.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(retriever, cases, k=3):
    ps, rs, fs = [], [], []
    type_f1 = defaultdict(list)
    for c in cases:
        docs = [d.page_content for d in retriever.invoke(c['query'], k=k)]
        p, r, f = prf(docs, c['expected_keywords'], c['expected_type'])
        ps.append(p); rs.append(r); fs.append(f)
        type_f1[c['expected_type']].append(f)
    avg = lambda l: sum(l) / len(l) if l else 0.0
    return {
        'precision': round(avg(ps), 4),
        'recall':    round(avg(rs), 4),
        'f1':        round(avg(fs), 4),
        'by_type':   {t: round(avg(v), 4) for t, v in type_f1.items()},
    }


# ── 主程序 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-csv',  default='test_cases_development_v2.csv')
    parser.add_argument('--hold-csv', default='test_cases_holdout_v2.csv')
    parser.add_argument('--model-dir', default='',
                        help='中文模型路径，留空则用环境中已有的模型')
    parser.add_argument('--output', default='output')
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dev_path  = os.path.join(base, "data", args.dev_csv)
    hold_path = os.path.join(base, "data", args.hold_csv)

    for p, name in [(dev_path, '开发集'), (hold_path, '保留集')]:
        if not os.path.exists(p):
            print(f'找不到{name}：{p}')
            sys.exit(1)

    dev_cases  = load_csv(dev_path)
    hold_cases = load_csv(hold_path)
    print(f'开发集：{len(dev_cases)} 条  保留集：{len(hold_cases)} 条')

    # 初始化模型和向量库
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from sentence_transformers import SentenceTransformer
    import chromadb
    from hybrid_retriever import HybridRetriever

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_path = args.model_dir or os.getenv("EMBEDDING_MODEL_PATH") or os.path.join(base, "models", "bge-small-zh-v1.5")
    print(f'加载模型：{model_path}')
    model = SentenceTransformer(model_path, device='cpu')
    client = chromadb.PersistentClient(path=os.path.join(base, 'chroma_db'))
    col = client.get_collection('antifraud_knowledge')
    print(f'向量库文档数：{col.count()}')

    # ── Step 1：在开发集上搜索最优 alpha ──
    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    print(f'\n{"="*60}')
    print('Step 1  开发集 alpha 搜索')
    print(f'{"="*60}')
    print(f'  {"alpha":>6}  {"Precision":>10}  {"Recall":>8}  {"F1":>8}')
    print(f'  {"-"*6}  {"-"*10}  {"-"*8}  {"-"*8}')

    dev_results = {}
    best_alpha, best_f1 = 0.0, 0.0
    for a in alphas:
        ret = HybridRetriever(model, col, alpha=a)
        r = evaluate(ret, dev_cases)
        dev_results[a] = r
        marker = '  <-- best' if r['f1'] > best_f1 else ''
        if r['f1'] > best_f1:
            best_f1, best_alpha = r['f1'], a
        print(f'  {a:>6.2f}  {r["precision"]:>10.4f}  {r["recall"]:>8.4f}  {r["f1"]:>8.4f}{marker}')

    print(f'\n  最优 alpha = {best_alpha}  (开发集 F1 = {best_f1:.4f})')

    # ── Step 2：用最优 alpha 在保留集上报告最终结果 ──
    print(f'\n{"="*60}')
    print(f'Step 2  保留集最终评估（alpha={best_alpha}）')
    print(f'{"="*60}')

    final_ret = HybridRetriever(model, col, alpha=best_alpha)
    final = evaluate(final_ret, hold_cases)

    print(f'  Precision : {final["precision"]:.4f}')
    print(f'  Recall    : {final["recall"]:.4f}')
    print(f'  F1        : {final["f1"]:.4f}')
    print()
    print(f'  按类型 F1：')
    for t, f in sorted(final['by_type'].items(), key=lambda x: x[1]):
        bar = '█' * int(f * 20)
        print(f'    {t:20}  {f:.4f}  {bar}')

    # ── Step 3：与基准对比 ──
    print(f'\n{"="*60}')
    print('Step 3  与历史结果对比')
    print(f'{"="*60}')
    baselines = [
        ('BM25 基准',              0.6125),
        ('混合（英文模型 α=1.0）', 0.4291),
        ('纯向量（中文模型）',     0.7709),
        ('混合（中文 α=1.0）',     0.6547),
    ]
    for name, f1 in baselines:
        delta = final['f1'] - f1
        print(f'  {name:24}  F1={f1:.4f}  delta={delta:+.4f}')
    print(f'  {"本文（RRF α="+str(best_alpha)+"）":24}  F1={final["f1"]:.4f}  <-- 最终结果')

    # ── 保存 ──
    os.makedirs(os.path.join(base, args.output), exist_ok=True)
    save = {
        'best_alpha':   best_alpha,
        'dev_results':  {str(k): v for k, v in dev_results.items()},
        'holdout_final': final,
    }
    out_path = os.path.join(base, args.output, 'rrf_tuning_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存：{out_path}')


if __name__ == '__main__':
    main()
