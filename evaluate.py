"""
端到端评估脚本
评估大模型生成的回答质量
"""

import csv
import os
import sys
import json
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_system import RAGSystem

def load_test_cases():
    """加载测试用例"""
    test_cases = []
    # 从保留集中选择20-30条测试用例
    with open('data/test_holdout.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append({
                'id': row['id'],
                'query': row['query'],
                'expected_type': row['expected_type']
            })
    # 随机选择30条
    import random
    random.seed(42)  # 固定种子，保证结果可重现
    selected_cases = random.sample(test_cases, min(30, len(test_cases)))
    print(f"选择了 {len(selected_cases)} 条测试用例进行评估")
    return selected_cases

def evaluate_retrieval_quality(retrieved_docs, expected_type):
    """
    评估检索质量
    检查 Top-3 文档是否命中正确类型
    """
    if not retrieved_docs:
        return 0
    
    # 检查前3个文档是否包含正确的诈骗类型
    for doc in retrieved_docs[:3]:
        if expected_type in doc:
            return 1
    return 0

def evaluate_answer_quality(answer, query, expected_type):
    """
    评估回答质量
    使用包含关键要素的 checklist 评分
    0: 完全错误
    1: 部分正确
    2: 完全正确
    """
    # 检查是否包含正确的诈骗类型
    if expected_type not in answer:
        return 0
    
    # 关键要素检查
    checklist = {
        '风险评估': any(keyword in answer for keyword in ['风险', '诈骗', '警惕', '危险', '注意']),
        '防范建议': any(keyword in answer for keyword in ['防范', '建议', '注意', '不要', '避免', '警惕']),
        '报警方式': any(keyword in answer for keyword in ['报警', '110', '反诈中心', '96110']),
        '证据收集': any(keyword in answer for keyword in ['证据', '收集', '保存', '记录']),
        '法律后果': any(keyword in answer for keyword in ['法律', '后果', '处罚', '量刑'])
    }
    
    # 计算命中的要素数量
    hit_count = sum(checklist.values())
    
    if hit_count >= 4:
        return 2
    elif hit_count >= 2:
        return 1
    else:
        return 0

def evaluate_end_to_end():
    """端到端评估"""
    print("端到端评估开始")
    print("=" * 70)
    
    # 加载测试用例
    test_cases = load_test_cases()
    
    # 初始化RAG系统
    print("\n初始化RAG系统...")
    rag = RAGSystem()
    if not rag._init_system():
        print("RAG系统初始化失败")
        return
    
    # 评估结果
    results = []
    type_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'partial': 0, 'incorrect': 0})
    
    # 执行评估
    for case in test_cases:
        print(f"\n评估测试用例: {case['id']}")
        print(f"查询: {case['query']}")
        print(f"预期类型: {case['expected_type']}")
        
        # 获取回答
        result = rag.answer_query(case['query'])
        answer = result.get('answer', '')
        retrieved_docs = result.get('retrieved_docs', [])
        
        print(f"回答: {answer[:100]}...")
        
        # 评估检索质量
        retrieval_score = evaluate_retrieval_quality(retrieved_docs, case['expected_type'])
        
        # 评估生成质量
        generation_score = evaluate_answer_quality(answer, case['query'], case['expected_type'])
        
        # 综合评分
        score = generation_score
        
        # 更新统计
        type_stats[case['expected_type']]['total'] += 1
        if score == 2:
            type_stats[case['expected_type']]['correct'] += 1
        elif score == 1:
            type_stats[case['expected_type']]['partial'] += 1
        else:
            type_stats[case['expected_type']]['incorrect'] += 1
        
        results.append({
            'id': case['id'],
            'query': case['query'],
            'expected_type': case['expected_type'],
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'retrieval_score': retrieval_score,
            'generation_score': generation_score,
            'score': score
        })
    
    # 计算整体统计
    total = len(results)
    correct = sum(1 for r in results if r['score'] == 2)
    partial = sum(1 for r in results if r['score'] == 1)
    incorrect = sum(1 for r in results if r['score'] == 0)
    
    accuracy = correct / total if total > 0 else 0
    
    # 输出结果
    print("\n" + "=" * 70)
    print("端到端评估结果")
    print("=" * 70)
    print(f"总测试用例: {total}")
    print(f"完全正确: {correct} ({correct/total*100:.2f}%)")
    print(f"部分正确: {partial} ({partial/total*100:.2f}%)")
    print(f"完全错误: {incorrect} ({incorrect/total*100:.2f}%)")
    print(f"准确率: {accuracy*100:.2f}%")
    
    print("\n按诈骗类型统计:")
    for fraud_type, stats in type_stats.items():
        type_total = stats['total']
        type_correct = stats['correct']
        type_accuracy = type_correct / type_total if type_total > 0 else 0
        print(f"{fraud_type}: {type_correct}/{type_total} ({type_accuracy*100:.2f}%)")
    
    # 保存评估结果
    output_path = 'output/end_to_end_evaluation.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'overall': {
                'total': total,
                'correct': correct,
                'partial': partial,
                'incorrect': incorrect,
                'accuracy': accuracy
            },
            'by_type': dict(type_stats),
            'details': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估结果已保存到: {output_path}")
    return accuracy

def main():
    """主函数"""
    evaluate_end_to_end()

if __name__ == "__main__":
    main()
