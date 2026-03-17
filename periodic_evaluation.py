#!/usr/bin/env python3
"""
定期评估脚本
用于定期评估反诈RAG系统的性能，并生成评估报告
"""

import os
import sys
import json
import csv
import time
import logging
from collections import defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/periodic_evaluation.log'
)

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_system import RAGSystem

def load_test_cases():
    """加载测试用例"""
    test_cases = []
    
    # 加载开发集
    if os.path.exists('data/test_dev.csv'):
        with open('data/test_dev.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_cases.append({
                    'id': row['id'],
                    'query': row['query'],
                    'expected_type': row['expected_type'],
                    'set': 'dev'
                })
    
    # 加载保留集
    if os.path.exists('data/test_holdout.csv'):
        with open('data/test_holdout.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_cases.append({
                    'id': row['id'],
                    'query': row['query'],
                    'expected_type': row['expected_type'],
                    'set': 'holdout'
                })
    
    logging.info(f"加载了 {len(test_cases)} 条测试用例")
    return test_cases

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

def evaluate_system():
    """评估系统性能"""
    logging.info("开始定期评估")
    print("\n开始定期评估...")
    
    # 加载测试用例
    test_cases = load_test_cases()
    if not test_cases:
        logging.error("无测试用例可评估")
        return None
    
    # 初始化RAG系统
    logging.info("初始化RAG系统")
    print("初始化RAG系统...")
    rag = RAGSystem()
    if not rag._init_system():
        logging.error("RAG系统初始化失败")
        return None
    
    # 评估结果
    results = []
    type_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'partial': 0, 'incorrect': 0})
    set_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'partial': 0, 'incorrect': 0})
    
    # 执行评估
    for case in test_cases:
        logging.info(f"评估测试用例: {case['id']} (set: {case['set']})")
        print(f"评估: {case['id']} ({case['set']})")
        
        # 获取回答
        result = rag.answer_query(case['query'])
        answer = result.get('answer', '')
        retrieved_docs = result.get('retrieved_docs', [])
        
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
        
        set_stats[case['set']]['total'] += 1
        if score == 2:
            set_stats[case['set']]['correct'] += 1
        elif score == 1:
            set_stats[case['set']]['partial'] += 1
        else:
            set_stats[case['set']]['incorrect'] += 1
        
        results.append({
            'id': case['id'],
            'set': case['set'],
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
    
    # 生成评估报告
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'overall': {
            'total': total,
            'correct': correct,
            'partial': partial,
            'incorrect': incorrect,
            'accuracy': accuracy
        },
        'by_type': dict(type_stats),
        'by_set': dict(set_stats),
        'details': results
    }
    
    # 保存评估报告
    output_dir = 'output/periodic_evaluations'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'evaluation_{time.strftime("%Y%m%d_%H%M%S")}.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logging.info(f"评估报告已保存到: {output_path}")
    print(f"\n评估报告已保存到: {output_path}")
    
    # 打印评估结果
    print("\n" + "=" * 70)
    print("定期评估结果")
    print("=" * 70)
    print(f"总测试用例: {total}")
    print(f"完全正确: {correct} ({correct/total*100:.2f}%)")
    print(f"部分正确: {partial} ({partial/total*100:.2f}%)")
    print(f"完全错误: {incorrect} ({incorrect/total*100:.2f}%)")
    print(f"准确率: {accuracy*100:.2f}%")
    
    print("\n按数据集统计:")
    for set_name, stats in set_stats.items():
        set_total = stats['total']
        set_correct = stats['correct']
        set_accuracy = set_correct / set_total if set_total > 0 else 0
        print(f"{set_name}: {set_correct}/{set_total} ({set_accuracy*100:.2f}%)")
    
    print("\n按诈骗类型统计:")
    for fraud_type, stats in type_stats.items():
        type_total = stats['total']
        type_correct = stats['correct']
        type_accuracy = type_correct / type_total if type_total > 0 else 0
        print(f"{fraud_type}: {type_correct}/{type_total} ({type_accuracy*100:.2f}%)")
    
    # 识别需要优化的类型
    print("\n需要优化的诈骗类型:")
    for fraud_type, stats in type_stats.items():
        type_total = stats['total']
        type_correct = stats['correct']
        type_accuracy = type_correct / type_total if type_total > 0 else 0
        if type_accuracy < 0.5:
            print(f"- {fraud_type}: {type_accuracy*100:.2f}%")
    
    return report

def generate_optimization_suggestions(report):
    """生成优化建议"""
    suggestions = []
    
    # 分析误判案例
    incorrect_cases = [r for r in report['details'] if r['score'] == 0]
    
    if incorrect_cases:
        suggestions.append("### 误判案例分析")
        suggestions.append(f"发现 {len(incorrect_cases)} 个误判案例，需要重点关注。")
        
        # 按类型统计误判
        type_errors = defaultdict(int)
        for case in incorrect_cases:
            type_errors[case['expected_type']] += 1
        
        suggestions.append("\n#### 误判类型分布:")
        for fraud_type, count in sorted(type_errors.items(), key=lambda x: x[1], reverse=True):
            suggestions.append(f"- {fraud_type}: {count} 次")
    
    # 分析数据集性能差异
    dev_stats = report['by_set'].get('dev', {})
    holdout_stats = report['by_set'].get('holdout', {})
    
    dev_accuracy = dev_stats.get('correct', 0) / dev_stats.get('total', 1)
    holdout_accuracy = holdout_stats.get('correct', 0) / holdout_stats.get('total', 1)
    
    if dev_accuracy > holdout_accuracy + 0.1:
        suggestions.append("\n### 过拟合风险")
        suggestions.append("开发集性能明显高于保留集，可能存在过拟合风险。")
        suggestions.append("建议：")
        suggestions.append("1. 增加保留集的测试用例多样性")
        suggestions.append("2. 调整模型参数，减少过拟合")
        suggestions.append("3. 增加数据增强，提高模型泛化能力")
    
    # 生成具体优化建议
    suggestions.append("\n### 优化建议")
    suggestions.append("1. **知识库优化**:")
    suggestions.append("   - 为误判率高的类型添加更多关键词和场景")
    suggestions.append("   - 优化知识库结构，提高信息检索效率")
    suggestions.append("   - 定期更新知识库，添加新的诈骗案例")
    
    suggestions.append("\n2. **检索策略优化**:")
    suggestions.append("   - 调整HybridRetriever的alpha参数")
    suggestions.append("   - 优化关键词权重和类型触发规则")
    suggestions.append("   - 考虑增加更多检索通道，如语义检索")
    
    suggestions.append("\n3. **生成质量优化**:")
    suggestions.append("   - 优化提示词模板，提高回答质量")
    suggestions.append("   - 调整大模型参数，如温度、top_p等")
    suggestions.append("   - 增加上下文长度，提高回答的连贯性")
    
    suggestions.append("\n4. **系统监控**:")
    suggestions.append("   - 建立实时监控机制，及时发现系统异常")
    suggestions.append("   - 收集用户反馈，持续改进系统")
    suggestions.append("   - 定期进行A/B测试，验证优化效果")
    
    # 保存优化建议
    output_dir = 'output/periodic_evaluations'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'optimization_suggestions_{time.strftime("%Y%m%d_%H%M%S")}.md')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(suggestions))
    
    logging.info(f"优化建议已保存到: {output_path}")
    print(f"\n优化建议已保存到: {output_path}")
    
    return suggestions

def main():
    """主函数"""
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    # 执行评估
    report = evaluate_system()
    if report:
        # 生成优化建议
        generate_optimization_suggestions(report)
    
    logging.info("定期评估完成")
    print("\n定期评估完成！")

if __name__ == "__main__":
    main()
