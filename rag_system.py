"""
校园反诈智能问答系统 - RAG 核心系统
"""

import os
import json
import time
import chromadb
import logging
from dotenv import load_dotenv

# 配置环境变量
load_dotenv()

# 设置为离线模式，仅使用本地模型
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
# 移除镜像源设置，避免尝试下载
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# 禁用日志
for logger_name in ['huggingface_hub', 'transformers', 'sentence_transformers',
                     'urllib3', 'requests', 'fsspec']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)


class RAGSystem:
    """完整的反诈RAG系统（正确版本）"""

    def __init__(self):
        """初始化RAG系统"""
        self.model = None
        self.collection = None
        self.retriever = None
        self.initialized = False
        self.conversation_history = []  # 对话历史

        self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _get_vector_db_path(self) -> str:
        configured = (os.getenv("VECTOR_DB_PATH") or "").strip()
        if configured:
            return configured
        return os.path.join(self._project_root, "chroma_db")

    def _get_embedding_model_path(self) -> str:
        configured = (os.getenv("EMBEDDING_MODEL_PATH") or "").strip()
        if configured:
            return configured
        return os.path.join(self._project_root, "models", "bge-small-zh-v1.5")

    def _llm_mode(self) -> str:
        """
        LLM_MODE:
          - auto    : 有合法 key 才调用，否则走离线退化回答
          - online  : 强制调用（无 key 则提示配置）
          - offline : 强制不调用（适合答辩离线演示）
        """
        mode = (os.getenv("LLM_MODE") or "auto").strip().lower()
        return mode if mode in ("auto", "online", "offline") else "auto"

    def _has_valid_api_key(self) -> bool:
        api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
        return bool(api_key and api_key != "your_api_key_here" and api_key.startswith("sk-"))

    def _offline_answer(self, query: str, retrieved_docs: list, history: list[tuple[str, str]]):
        """
        无大模型情况下的退化回答：基于检索文本给出结构化建议，保证可离线演示。
        """
        excerpts = []
        for doc in retrieved_docs[:3]:
            text = (doc.page_content or "").strip().replace("\n", " ")
            if text:
                excerpts.append(text[:160])

        lines = [
            "当前为离线模式：未调用大模型（DashScope）。以下建议来自本地知识库检索结果。",
            "",
            "1. 风险评估",
            "该情况存在较高诈骗风险。若涉及转账/充值/点击不明链接/提供验证码，请立即停止操作。",
            "",
            "2. 警示信号",
            "- 要求点击链接、下载陌生App、填写银行卡/身份证信息",
            "- 要求先转账/充值/交保证金/激活费/刷流水",
            "- 催促、威胁、制造紧迫感（\"不操作就失效/影响学籍/涉案\"）",
            "",
            "3. 建议处置",
            "- 不转账、不泄露验证码，不点不明链接",
            "- 通过学校官方渠道核验（教务处/辅导员/官方电话）",
            "- 保存证据：聊天记录、转账凭证、链接/二维码截图、对方账号信息",
            "- 如已造成损失：及时报警 110 / 反诈 96110，并联系银行/支付平台尝试止付",
        ]

        if excerpts:
            lines += ["", "4. 参考摘录（Top-3）"]
            for i, ex in enumerate(excerpts, 1):
                lines.append(f"- 资料{i}：{ex}...")

        answer_text = "\n".join(lines)
        new_history = (history or []).copy()
        new_history.append(("用户", query))
        new_history.append(("助手", answer_text))
        return {
            "query": query,
            "retrieved_docs": [doc.page_content[:300] for doc in retrieved_docs],
            "answer": answer_text,
            "success": True,
            "history": new_history,
        }

    def _init_system(self):
        """初始化系统组件（懒加载）"""
        if self.initialized:
            return True

        print("Initializing RAG system...\n")

        try:
            # 加载 Embedding 模型 - 优化内存使用
            print("  Loading embedding model...")
            from sentence_transformers import SentenceTransformer

            # 优化模型加载，减少内存使用
            # 仅使用本地模型，不尝试下载
            model_path = self._get_embedding_model_path()
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    "本地 Embedding 模型路径不存在。请在 .env 中设置 EMBEDDING_MODEL_PATH，"
                    "或将模型文件放到项目目录 `models/bge-small-zh-v1.5/`。"
                )
            self.model = SentenceTransformer(
                model_path,
                device="cpu",
                model_kwargs={"torch_dtype": "float32"}  # 使用float32减少内存
            )
            print(f"  [OK] Using model: {model_path}")
            print("    [OK] Embedding model loaded")

            # 加载向量存储
            print("  Loading vector store...")
            vector_db_path = self._get_vector_db_path()
            if not os.path.exists(vector_db_path):
                raise FileNotFoundError(
                    f"向量库目录不存在：{vector_db_path}。请先运行 `python scripts/build_kg.py` 构建知识库，"
                    "或在 .env 中设置 VECTOR_DB_PATH。"
                )

            # 简化ChromaDB配置，使用兼容版本
            import chromadb
            # 移除settings参数，使用默认配置
            client = chromadb.PersistentClient(path=vector_db_path)
            self.collection = client.get_collection(name="antifraud_knowledge")

            doc_count = self.collection.count()
            if doc_count == 0:
                raise ValueError("向量存储为空")

            print(f"    [OK] Vector store loaded ({doc_count} docs)")

            # 创建检索器
            print("  Creating retriever...")
            self.retriever = self._create_retriever()
            print("    [OK] Retriever created")

            # 检查 API 配置
            print("  Checking API config...")
            api_key = os.getenv('DASHSCOPE_API_KEY', '').strip()
            if api_key and api_key != 'your_api_key_here' and api_key.startswith('sk-'):
                print("    [OK] API key configured")
            else:
                print("    [WARN] API key missing/invalid")

            self.initialized = True
            return True

        except Exception as e:
            print(f"  [ERROR] Init failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _create_retriever(self):
        """创建检索器"""
        from hybrid_retriever import HybridRetriever
        return HybridRetriever(self.model, self.collection, alpha=0.8)

    def answer_query(self, query, k=5, temperature=0.3, history=None):
        """完整的 RAG 流程：检索 + 生成
        
        Args:
            query: 用户查询
            k: 检索文档数量
            temperature: 生成温度
            history: 对话历史，格式为 [("用户", "问题"), ("助手", "回答")]
        """
        # 懒加载初始化
        if not self.initialized:
            if not self._init_system():
                return {
                    "query": query,
                    "retrieved_docs": [],
                    "answer": "系统初始化失败",
                    "success": False
                }
        
        if not self.retriever:
            return {
                "query": query,
                "retrieved_docs": [],
                "answer": "系统未正确初始化",
                "success": False
            }

        try:
            # 使用外部传入的历史，不维护内部状态
            current_history = history or []
            # 创建新的历史记录
            new_history = current_history.copy()
            new_history.append(("用户", query))

            # 检查是否为非诈骗相关的问候或无关问题
            if self._is_non_fraud_query(query):
                # 对于非诈骗相关问题，使用通用对话模式
                # 构建包含历史的 prompt
                history_str = """
【对话历史】
"""
                for role, content in current_history:
                    history_str += f"{role}：{content}\n"
                
                prompt = f"""你是一个智能助手，友好且乐于助人。请针对用户的问题提供自然、友好的回答。

{history_str}
【用户提问】
{query}

【回答要求】
- 回答要自然、友好
- 保持对话的连贯性
- 提供有帮助的信息
- 语言简洁明了"""
                
                print(f"    [生成] 处理非诈骗问题...")
                answer_text = self._call_llm(prompt, temperature=temperature)
                
                # 更新对话历史
                new_history.append(("助手", answer_text))
                
                return {
                    "query": query,
                    "retrieved_docs": [],
                    "answer": answer_text,
                    "success": True,
                    "history": new_history
                }

            # 第 1 步：检索
            print(f"    [检索] 搜索相关资料...")
            
            # 检索时仅使用当前轮查询，避免历史信息稀释向量表示
            # 历史信息仅用于 Prompt 构造
            full_query = query
            
            # 对于询问常见诈骗例子、类型等问题，增加检索结果数量
            query_lower = query.lower()
            if any(keyword in query_lower for keyword in ['例子', '类型', '种类', '案例', '常见']):
                retrieved_docs = self.retriever.invoke(full_query, k=10)  # 增加到10条
            else:
                retrieved_docs = self.retriever.invoke(full_query, k=k)

            if not retrieved_docs:
                return {
                    "query": query,
                    "retrieved_docs": [],
                    "answer": "没有找到相关资料，请联系学校保卫部门。",
                    "success": True,
                    "history": new_history
                }

            print(f"    [检索] 找到 {len(retrieved_docs)} 条资料")

            # 离线/无 key 退化路径（便于答辩可离线演示）
            mode = self._llm_mode()
            if mode == "offline" or (mode == "auto" and not self._has_valid_api_key()):
                return self._offline_answer(query=query, retrieved_docs=retrieved_docs, history=current_history)

            # 第 2 步：构造 Prompt - 优化版
            # 限制上下文长度，减少内存使用
            max_context_length = 1500
            context_parts = []
            current_length = 0
            
            for doc in retrieved_docs:
                doc_content = doc.page_content
                if current_length + len(doc_content) <= max_context_length:
                    context_parts.append(doc_content)
                    current_length += len(doc_content) + 2  # +2 for newline
                else:
                    break
            
            context = "\n\n".join(context_parts)

            # 构建包含历史的 prompt
            history_str = """
【对话历史】
"""
            if len(current_history) > 0:
                for role, content in current_history:
                    history_str += f"{role}：{content}\n"

            # 优化提示词，减少幻觉
            core_instructions = """
【核心指令】
- 严格基于提供的背景知识回答，不要编造任何信息
- 只使用背景知识中的具体内容
- 若背景知识不足，明确说明无法提供更多信息
- 保持回答简洁、准确、权威
- 考虑对话历史，保持回答的连贯性"""

            # 根据查询意图分类并调整Prompt
            # 事后处置类关键词
            aftermath_keywords = ['被骗', '怎么办', '报警', '处理', '应对', '止损', '举报', '追回', '损失']
            # 预防咨询类关键词
            prevention_keywords = ['如何防范', '怎么预防', '怎样避免', '防范措施', '预防方法']
            # 识别判断类关键词
            identification_keywords = ['是诈骗吗', '是不是诈骗', '真假', '可信吗', '靠谱吗', '是否', '辨别', '识别']
            
            # 判断查询类型
            is_aftermath = any(keyword in query_lower for keyword in aftermath_keywords)
            is_prevention = any(keyword in query_lower for keyword in prevention_keywords)
            is_identification = any(keyword in query_lower for keyword in identification_keywords)
            is_example = any(keyword in query_lower for keyword in ['例子', '类型', '种类', '案例', '常见'])
            
            if is_aftermath:
                # 事后处置类模板
                prompt = f"""你是一个专业的校园反诈安全顾问。

{core_instructions}

{history_str}
【背景知识】
{context}

【用户提问】
{query}

请按以下格式简洁回答：
1. 事件性质判断
2. 立即止损步骤
3. 报警流程指南
4. 后续处理建议
5. 证据收集方法"""
            elif is_prevention:
                # 预防咨询类模板
                prompt = f"""你是一个专业的校园反诈安全顾问。

{core_instructions}

{history_str}
【背景知识】
{context}

【用户提问】
{query}

请按以下格式简洁回答：
1. 常见诈骗类型
2. 警示信号
3. 防范建议
4. 识别方法
5. 求助渠道"""
            elif is_identification:
                # 识别判断类模板
                prompt = f"""你是一个专业的校园反诈安全顾问。

{core_instructions}

{history_str}
【背景知识】
{context}

【用户提问】
{query}

请按以下格式简洁回答：
1. 诈骗风险评估
2. 警示信号
3. 防范建议
4. 遇害处理方式
5. 法律后果"""
            elif is_example:
                # 例子类型模板
                prompt = f"""你是一个专业的校园反诈安全顾问。

{core_instructions}

{history_str}
【背景知识】
{context}

【用户提问】
{query}

请按以下格式简洁回答：
1. 常见诈骗类型
2. 风险评估
3. 警示信号
4. 防范建议
5. 法律后果"""
            else:
                # 默认模板
                prompt = f"""你是一个专业的校园反诈安全顾问。

{core_instructions}

{history_str}
【背景知识】
{context}

【用户提问】
{query}

请按以下格式简洁回答：
1. 诈骗风险评估
2. 警示信号
3. 防范建议
4. 遇害处理方式
5. 法律后果"""


            # 第 3 步：调用大模型
            print(f"    [生成] 调用大模型...")
            answer_text = self._call_llm(prompt, temperature=temperature)

            # 更新对话历史
            new_history.append(("助手", answer_text))

            return {
                "query": query,
                "retrieved_docs": [doc.page_content[:300] for doc in retrieved_docs],
                "answer": answer_text,
                "success": True,
                "history": new_history,
                "metadata": [getattr(doc, "metadata", {}) for doc in retrieved_docs],
            }

        except Exception as e:
            return {
                "query": query,
                "retrieved_docs": [],
                "answer": f"处理失败: {str(e)}",
                "success": False,
                "history": history or []
            }
    
    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []
    
    def _is_non_fraud_query(self, query):
        """判断是否为非诈骗相关的查询"""
        # 常见的问候语和无关问题
        non_fraud_patterns = [
            '你好', '您好', 'hi', 'hello', '嗨',
            '在吗', '在不在', '你是谁', '你是什么',
            '你好吗', '最近怎么样', '今天天气',
            '谢谢', '再见', '拜拜',
            '早上好', '下午好', '晚上好',
            '你叫什么', '你能', '你会',
            '时间', '日期', '星期', '节日',
            '学校', '课程', '学习', '考试',
            '生活', '健康', '娱乐', '体育',
            '科技', '新闻', '音乐', '电影'
        ]
        
        # 检查是否包含常见的诈骗关键词
        fraud_keywords = [
            '诈骗', '刷单', '兼职', '充值', '转账',
            '贷款', '校园贷', '教务处', '奖学金',
            '账号', '密码', '链接', '付款', '押金',
            '实习', '理赔', '直播', '打赏', '换脸',
            '钓鱼', '中奖', '汇款', '投资', '理财',
            '手续费', '激活', '验证', '解冻', '保证金',
            '例子', '类型', '种类', '案例', '常见'
        ]
        
        # 转换为小写进行匹配
        query_lower = query.lower()
        
        # 检查是否为关于诈骗的查询
        if '诈骗' in query_lower:
            return False
        
        # 检查是否包含诈骗相关关键词
        if any(keyword in query_lower for keyword in fraud_keywords):
            return False
        
        # 检查是否为常见的非诈骗问题
        for pattern in non_fraud_patterns:
            if pattern in query_lower:
                # 检查是否同时包含诈骗关键词
                has_fraud_keyword = any(keyword in query_lower for keyword in fraud_keywords)
                if not has_fraud_keyword:
                    return True
        
        # 检查是否为非常简短的无关问题
        if len(query) < 10:
            # 检查是否包含诈骗关键词
            has_fraud_keyword = any(keyword in query_lower for keyword in fraud_keywords)
            if not has_fraud_keyword:
                return True
        
        # 检查是否为明显的非诈骗相关主题
        non_fraud_topics = [
            '天气', '时间', '日期', '星期',
            '学校', '课程', '学习', '考试',
            '生活', '健康', '娱乐', '体育',
            '科技', '新闻', '音乐', '电影',
            '旅游', '美食', '购物', '游戏',
            '历史', '地理', '数学', '英语'
        ]
        
        for topic in non_fraud_topics:
            if topic in query_lower:
                # 检查是否同时包含诈骗关键词
                has_fraud_keyword = any(keyword in query_lower for keyword in fraud_keywords)
                if not has_fraud_keyword:
                    return True
        
        return False

    def _call_llm(self, prompt, temperature=0.3):
        """调用大模型：优先通过 LangChain 封装 ChatTongyi。"""
        # 验证 API 密钥
        api_key = os.getenv('DASHSCOPE_API_KEY', '').strip()
        if not api_key or api_key == 'your_api_key_here':
            return "ERROR: API key not configured. Please set DASHSCOPE_API_KEY in .env"
        if not api_key.startswith('sk-'):
            return "ERROR: Invalid API key format (should start with sk-)"

        # 优先走 LangChain 封装
        try:
            from src.langchain_llm import generate_answer
            return generate_answer(prompt, temperature=temperature)
        except Exception as e:
            # 回退到 dashscope 原生调用，避免 LangChain 异常导致系统不可用
            try:
                from dashscope import Generation

                response = Generation.call(
                    model="qwen-turbo",
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=temperature,
                    max_tokens=1024,
                    top_p=0.95,
                )
                if hasattr(response, 'status_code') and response.status_code == 200:
                    if hasattr(response, 'output') and response.output and getattr(response.output, 'text', None):
                        return response.output.text
                    return "ERROR: API response missing output.text"
                status = getattr(response, 'status_code', '未知')
                message = getattr(response, 'message', '')
                return f"ERROR: API call failed: {status} {message}"
            except Exception as e2:
                return f"ERROR: LLM call failed (langchain={e}, dashscope={e2})"


