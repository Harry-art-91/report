"""
构建知识库和向量存储
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_vector_db_path() -> str:
    configured = (os.getenv("VECTOR_DB_PATH") or "").strip()
    if configured:
        return configured
    return os.path.join(PROJECT_ROOT, "chroma_db")

def get_embedding_model_path() -> str:
    configured = (os.getenv("EMBEDDING_MODEL_PATH") or "").strip()
    if configured:
        return configured
    return os.path.join(PROJECT_ROOT, "models", "bge-small-zh-v1.5")

def setup_local_model():
    """设置本地模型"""
    print("=" * 60)
    print("配置本地模型")
    print("=" * 60)

    # 设置离线模式
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    print("[OK] 设置为离线模式")
    print("  仅使用本地模型，不尝试下载")
    print()

    return True

def check_api_keys():
    """检查API密钥是否配置"""
    print("=" * 60)
    print("检查API密钥配置")
    print("=" * 60)

    dashscope_key = os.getenv('DASHSCOPE_API_KEY')

    if not dashscope_key:
        print("[WARN] DASHSCOPE_API_KEY 未配置")
        print("   请编辑 .env 文件添加密钥")
        print("   获取地址: https://dashscope.console.aliyun.com/")
        return False

    print(f"[OK] DASHSCOPE_API_KEY: {dashscope_key[:10]}...")
    print("\n说明: 使用本地模型，完全离线")
    print("       无需下载，直接使用本地模型文件\n")
    return True

def load_data():
    """加载数据"""
    print("=" * 60)
    print("加载数据")
    print("=" * 60)

    import json
    cases = []
    
    # 直接从CSV文件加载诈骗案例
    # 使用绝对路径，确保从项目根目录加载
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'fraud_cases_optimized.csv')
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for idx, row in enumerate(reader):
                # 构建案例对象
                case = {
                    "id": f"case_{idx+1}",
                    "title": f"{row['fraud_type']}-{row['scenario']}",
                    "content": f"{row['fraud_type']}诈骗是一种常见的网络诈骗手段。\n\n案例：{row['detailed_description']}\n\n防范措施：{row['prevention_tips']}",
                    "fraud_type": row['fraud_type'],
                    "keywords": [kw.strip() for kw in row['keywords'].split(',')]
                }
                cases.append(case)
        print(f"[OK] 加载CSV知识库: {len(cases)} 条")
    
    # AI换脸诈骗案例已集成到CSV知识库中
    
    if not cases:
        print("[ERROR] 没有加载到任何诈骗案例")
        print("   请确保数据文件存在")
        return None, None

    print(f"[OK] 总诈骗案例: {len(cases)} 条")
    
    # 提取唯一的诈骗类型
    fraud_types = list(set(case['fraud_type'] for case in cases))
    print(f"  包含类型: {', '.join(fraud_types)}")
    print("  包含关键词字段: 是")

    # 加载反诈政策
    policies_path = os.path.join(PROJECT_ROOT, 'data', 'fraud_policies.txt')
    if not os.path.exists(policies_path):
        print(f"[WARN] 文件不存在: {policies_path}")
        return cases, None

    with open(policies_path, 'r', encoding='utf-8') as f:
        policies = f.read()
    print(f"[OK] 加载反诈政策文件: {len(policies)} 字符")
    print()

    return cases, policies

def process_fraud_cases(cases):
    """处理诈骗案例数据"""
    print("=" * 60)
    print("处理诈骗案例数据")
    print("=" * 60)

    documents = []

    for idx, case in enumerate(cases):
        # 为每个案例构建结构化文本
        # 检查是否有keywords字段
        if 'keywords' in case and case.get('keywords'):
            keywords = ', '.join(case['keywords'])
            doc_text = f"""
【诈骗类型】{case['fraud_type']}

【标题】{case['title']}

【内容】{case['content']}

【关键词】{keywords}
""".strip()
        else:
            doc_text = f"""
【诈骗类型】{case['fraud_type']}

【标题】{case['title']}

【内容】{case['content']}
""".strip()

        documents.append({
            'content': doc_text,
            'fraud_type': case['fraud_type'],
            'case_id': case['id'],
            # 结构化字段：若 CSV 中存在则附带到 metadata
            'victim_group': case.get('victim_group', ''),
            'channel': case.get('channel', ''),
            'money_flow': case.get('money_flow', ''),
            'risk_level': case.get('risk_level', ''),
        })

    print(f"[OK] 处理了 {len(documents)} 个诈骗案例")
    print()
    return documents

def clean_text(text):
    """文本清洗"""
    # 移除多余的空白字符
    text = ' '.join(text.split())
    # 移除特殊字符
    text = ''.join([c for c in text if c.isprintable() or c in '\n\t'])
    return text

def chunk_text(text, chunk_size=512, overlap=50):
    """对文本进行分块"""
    # 先清洗文本
    text = clean_text(text)
    
    chunks = []
    start = 0
    text_length = len(text)
    
    # 安全检查，防止无限循环
    min_step = max(1, chunk_size - overlap)

    while start < text_length:
        end = start + chunk_size
        # 确保end不超过文本长度
        end = min(end, text_length)
        
        # 尝试在句子边界处分割
        if end < text_length:
            # 寻找最近的句号、问号或感叹号
            punctuation = ['.', '。', '!', '！', '?', '？']
            for p in punctuation:
                pos = text.rfind(p, start, end)
                if pos != -1:
                    end = pos + 1
                    break
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        # 计算下一步的start位置，确保至少前进1个字符
        next_start = end - overlap
        if next_start <= start:
            next_start = start + min_step
        
        start = next_start
        
        # 防止无限循环的最终安全检查
        if start >= text_length:
            break

    return chunks

def adaptive_chunk_text(text, content_type):
    """根据内容类型自适应分块"""
    if content_type == 'fraud_case':
        # 诈骗案例：较大的分块以保持案例完整性
        return chunk_text(text, chunk_size=600, overlap=80)
    elif content_type == 'policy':
        # 政策文件：较小的分块以提高检索精度
        return chunk_text(text, chunk_size=400, overlap=60)
    else:
        # 默认分块策略
        return chunk_text(text, chunk_size=512, overlap=50)

def build_vector_store(documents, policies):
    """构建向量存储（国内镜像版本）"""
    print("=" * 60)
    print("构建向量存储")
    print("=" * 60)

    # 初始化 Embedding 模型
    print("初始化本地 Embedding 模型...")
    print("  模型: BAAI/bge-small-zh-v1.5")
    print("  大小: 仅 130 MB")
    print("  使用本地模型文件，无需下载")
    print()

    try:
        from sentence_transformers import SentenceTransformer
        import gc
        import json

        # 使用本地模型
        print("  正在加载本地模型...")
        model_path = get_embedding_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "本地 Embedding 模型路径不存在。请在 .env 中设置 EMBEDDING_MODEL_PATH，"
                "或将模型文件放到项目目录 `models/bge-small-zh-v1.5/`。"
            )
        model = SentenceTransformer(model_path)
        print(f"[OK] Embedding 模型已初始化: {model_path}")
    except Exception as e:
        print(f"[ERROR] 初始化 Embedding 失败: {str(e)}")
        print("   请运行: pip install sentence-transformers")
        return False

    # 创建存储目录（在项目根目录）
    chroma_db_path = get_vector_db_path()
    processed_data_path = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(chroma_db_path, exist_ok=True)
    os.makedirs(processed_data_path, exist_ok=True)

    # 初始化 ChromaDB 客户端（持久化模式）
    print("  正在初始化 ChromaDB...")
    try:
        import chromadb
        # 使用兼容的配置
        client = chromadb.PersistentClient(path=chroma_db_path)

        # 删除旧的 collection（如果存在）
        try:
            client.delete_collection(name="antifraud_knowledge")
            print("  已删除旧的 collection")
        except:
            pass

        # 创建新 collection
        collection = client.create_collection(
            name="antifraud_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        print("  [OK] Collection 已创建")
    except Exception as e:
        print(f"[ERROR] 初始化 ChromaDB 失败: {str(e)}")
        return False

    # 准备保存chunks
    chunks = []

    # 分批处理和添加诈骗案例
    print("  正在处理诈骗案例...")
    case_count = 0
    batch_size = 3  # 进一步减小批处理大小
    
    for doc in documents:
        # 不进行分块，直接使用完整的案例文本
        text = doc['content']
        case_count += 1
        
        # 保存到chunks列表
        chunks.append({
            "chunk_id": f"case_{doc['case_id']}",
            "text": text,
            "metadata": {
                "fraud_type": doc['fraud_type'],
                "source": "fraud_case",
                "case_id": doc['case_id'],
                "victim_group": doc.get('victim_group', ''),
                "channel": doc.get('channel', ''),
                "money_flow": doc.get('money_flow', ''),
                "risk_level": doc.get('risk_level', ''),
            }
        })

    # 分批添加到向量库
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        batch_docs = documents[i:batch_end]
        batch_texts = [doc['content'] for doc in batch_docs]
        batch_ids = [f"case_{doc['case_id']}" for doc in batch_docs]
        batch_metadatas = [{
            'source': 'fraud_case',
            'fraud_type': doc['fraud_type'],
            'case_id': doc['case_id'],
            'victim_group': doc.get('victim_group', ''),
            'channel': doc.get('channel', ''),
            'money_flow': doc.get('money_flow', ''),
            'risk_level': doc.get('risk_level', ''),
        } for doc in batch_docs]
        
        # 向量化并添加
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        batch_embeddings = batch_embeddings.tolist()
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas
        )
        
        # 释放内存
        del batch_texts, batch_embeddings
        gc.collect()

    print(f"  诈骗案例: {case_count} 个文本块")

    # 处理反诈政策
    policy_count = 0
    if policies:
        print("  正在处理反诈政策...")
        policy_chunks = adaptive_chunk_text(policies, 'policy')
        policy_count = len(policy_chunks)
        
        # 保存政策chunks
        for j, chunk in enumerate(policy_chunks):
            chunks.append({
                "chunk_id": f"policy_{j}",
                "text": chunk,
                "metadata": {
                    "source": "policy"
                }
            })
        
        # 分批添加到向量库
        for i in range(0, len(policy_chunks), batch_size):
            batch_end = min(i + batch_size, len(policy_chunks))
            batch_chunks = policy_chunks[i:batch_end]
            batch_ids = [f"policy_{j}" for j in range(i, batch_end)]
            batch_metadatas = [{'source': 'policy'} for _ in batch_chunks]
            
            # 向量化并添加
            batch_embeddings = model.encode(batch_chunks, show_progress_bar=False)
            batch_embeddings = batch_embeddings.tolist()
            
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_chunks,
                metadatas=batch_metadatas
            )
            
            # 释放内存
            del batch_chunks, batch_embeddings
            gc.collect()

        print(f"  反诈政策: {policy_count} 个文本块")

    # 保存chunks.json文件
    print("  正在保存chunks.json文件...")
    chunks_path = os.path.join(processed_data_path, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  [OK] 已保存chunks.json文件: {chunks_path}")

    # 释放模型内存
    del model
    gc.collect()

    # 验证添加结果
    doc_count = collection.count()
    print(f"[OK] 向量存储已创建: {chroma_db_path}")
    print(f"  包含 {doc_count} 个文本块")
    print()

    if doc_count == 0:
        print("[WARN] Collection 中没有文档！")
        return False

    return True

def test_retrieval(documents):
    """测试向量检索"""
    print("=" * 60)
    print("测试向量检索")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
        import gc

        # 加载模型
        print("  加载模型...")
        model_path = get_embedding_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "本地 Embedding 模型路径不存在。请在 .env 中设置 EMBEDDING_MODEL_PATH，"
                "或将模型文件放到项目目录 `models/bge-small-zh-v1.5/`。"
            )
        model = SentenceTransformer(model_path)
        print(f"  使用模型: {model_path}")

        # 连接到 ChromaDB
        client = chromadb.PersistentClient(path=get_vector_db_path())
        collection = client.get_collection(name="antifraud_knowledge")

        # 显示 collection 信息
        doc_count = collection.count()
        print(f"[OK] Collection 包含 {doc_count} 个文档")
        print()

        # 测试查询
        test_queries = [
            "我找到了一个刷单兼职，需要先充值",
            "教务处发短信要我点击链接",
            "有人说可以给我办理低息贷款"
        ]

        print("[OK] 运行测试查询...\n")

        for query in test_queries:
            print(f"查询: {query}")

            # 向量化查询
            query_embedding = model.encode(query)

            # 搜索相似文本
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=2
            )

            if results['documents'] and results['documents'][0]:
                print(f"   [OK] 找到 {len(results['documents'][0])} 条相关结果:")
                for i, doc in enumerate(results['documents'][0], 1):
                    content = doc.replace('\n', ' ')
                    print(f"     {i}. {content[:100]}...")
            else:
                print("   ✗ 没有找到相关结果")
            print()
            
            # 释放内存
            del query_embedding, results
            gc.collect()

        # 释放模型内存
        del model
        gc.collect()

        return True

    except Exception as e:
        print(f"[ERROR] 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False



def main():
    """主程序"""
    print("\n")
    print("=" * 60)
    print("构建知识库和向量存储（国内镜像版本）")
    print("=" * 60)
    print()

    try:
        # 0. 设置本地模型
        setup_local_model()

        # 1. 检查API密钥
        if not check_api_keys():
            print("\n[WARN] 提示: API 密钥仅用于后续的大模型调用")
            print("       本步骤可以离线完成")
            # 继续执行，不停止

        # 2. 加载数据
        df, policies = load_data()
        if df is None:
            return

        # 3. 处理数据
        documents = process_fraud_cases(df)

        # 4. 构建向量存储
        success = build_vector_store(documents, policies)
        if not success:
            print("\n[ERROR] 向量存储构建失败！")
            return

        # 5. 测试检索
        test_retrieval(documents)

        # 总结
        print("=" * 60)
        print("[OK] 知识库构建完成！")
        print("=" * 60)
        print("\n接下来的步骤：")
        print("1. 测试RAG系统")
        print("2. 启动Web应用")
        print("3. 根据需要调整和优化")
        print("模型信息：")
        print("- Embedding 模型：BAAI/bge-small-zh-v1.5")
        print("- 模型大小：仅 130 MB")
        print("- 向量维度：384")
        print("- 镜像源：hf-mirror.com（国内最快）")
        print("- 完全离线：是（下载后）")

    except Exception as e:
        print(f"\n[ERROR] 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()