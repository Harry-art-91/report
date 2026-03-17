"""
校园反诈智能问答系统 - Streamlit 应用
"""

import streamlit as st
import time
import os
import logging
from dotenv import load_dotenv

# 配置页面
st.set_page_config(
    page_title="校园反诈智能问答系统",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载环境变量
load_dotenv()

# 禁用日志
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)

# 配置 RAG 系统环境
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ============================================
# 初始化 Session State
# ============================================
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""

if 'last_result' not in st.session_state:
    st.session_state.last_result = None

if 'show_result' not in st.session_state:
    st.session_state.show_result = False

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# ============================================
# 页面标题
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.8em;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.3em;
        color: #666;
        text-align: center;
        margin-bottom: 1.5em;
    }
    .stButton>button {
        border-radius: 8px;
        border: 2px solid #4CAF50;
        background-color: #f8f9fa;
        color: #4CAF50;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4CAF50;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton>button:active {
        transform: translateY(0);
    }
    .stTextArea>div>textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 12px;
        font-size: 14px;
    }
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stExpander {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    .stSuccess {
        border-radius: 8px;
        padding: 15px;
    }
    .stWarning {
        border-radius: 8px;
        padding: 15px;
    }
    .stError {
        border-radius: 8px;
        padding: 15px;
    }
    .stInfo {
        border-radius: 8px;
        padding: 15px;
    }
    .footer {
        margin-top: 2em;
        padding-top: 1em;
        border-top: 1px solid #e0e0e0;
        text-align: center;
        color: #999;
        font-size: 0.9em;
    }
</style>

<div class="main-header">🛡️ 校园反诈智能问答系统</div>
<div class="sub-header">基于 RAG 技术的校园诈骗识别和防范助手</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================
# 侧边栏配置
# ============================================

with st.sidebar:
    st.markdown("### 📚 系统信息")

    st.info("""
    **系统特点：**
    - 🔍 向量检索（本地离线）
    - 🤖 大模型生成（通义千问）
    - 📊 多类别诈骗识别
    - ⚡ 秒级响应
    
    **技术栈：**
    - Python + Streamlit
    - Embedding: BAAI/bge-small-zh-v1.5
    - Vector DB: ChromaDB
    - LLM: DashScope API
    """)

    st.markdown("---")
    st.markdown("### ⚙️ 系统设置")

    k = st.slider(
        "检索结果数量",
        min_value=1,
        max_value=5,
        value=3,
        help="从知识库中检索多少条相关资料"
    )

    temperature = st.slider(
        "回答创意度",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="低值=保守，高值=创意"
    )

# ============================================
# 初始化 RAG 系统
# ============================================

@st.cache_resource
def load_rag_system():
    """加载 RAG 系统（缓存）"""
    try:
        from src.rag_system import RAGSystem
        return RAGSystem(), None
    except Exception as e:
        return None, str(e)

rag, error = load_rag_system()

if error:
    st.error(f"""
    ❌ 无法加载 RAG 系统
    
    错误信息：{error}
    
    **解决步骤：**
    1. 确保已运行知识库构建：`python scripts/build_kg.py`
    2. 确保 `chroma_db/` 目录存在
    3. 确保已安装所有依赖
    4. 检查 `.env` 文件中是否有有效的 API 密钥
    """)
    st.stop()

# ============================================
# 示例问题定义
# ============================================

examples = [
    ("刷单兼职", "我在网上找到了一个刷单兼职，对方说需要先充值100块钱才能开始，这是诈骗吗？"),
    ("假冒教务处", "教务处发来短信说我的学费有问题，要我点击链接确认，我该怎么办？"),
    ("校园贷", "有人说可以帮我申请低息校园贷，利息很低，我能信任吗？"),
    ("游戏账号诈骗", "网上有人说可以便宜卖游戏账号，还能帮我代练，信吗？"),
    ("冒充辅导员", "老师的微信说班费没交齐，要求我转账，真的吗？"),
    ("虚假实习", "某公司说提供高薪实习，需要先交押金，这靠谱吗？"),
    ("快递理赔", "有人说我的快递丢失，提供理赔链接，要我填写信息，是真的吗？"),
    ("网络直播", "主播说和我恋爱，要求我打赏礼物，我该怎么办？"),
    ("AI换脸", "视频中看到朋友说急需借钱，但是感觉不太对，是诈骗吗？")
]

# ============================================
# 快速分析函数
# ============================================

def analyze_query(query_text, k=3, temperature=0.3):
    """分析查询并返回结果"""
    with st.spinner("⏳ 正在分析你的问题..."):
        start_time = time.time()
        result = rag.answer_query(query_text, k=k, temperature=temperature, history=st.session_state.conversation_history)
        elapsed_time = time.time() - start_time
        # 更新对话历史
        if result.get('history'):
            st.session_state.conversation_history = result['history']

    return result, elapsed_time

# ============================================
# 快速示例按钮
# ============================================

st.markdown("### ⚡ 快速示例")

# 第一行
col1, col2, col3 = st.columns(3)

with col1:
    if st.button(examples[0][0], use_container_width=True, key="ex1"):
        st.session_state.user_query = examples[0][1]
        result, elapsed_time = analyze_query(examples[0][1], k=k, temperature=temperature)
        st.session_state.last_result = {
            'result': result,
            'elapsed_time': elapsed_time,
            'query': examples[0][1]
        }
        st.session_state.show_result = True
        st.rerun()

with col2:
    if st.button(examples[1][0], use_container_width=True, key="ex2"):
        st.session_state.user_query = examples[1][1]
        result, elapsed_time = analyze_query(examples[1][1], k=k, temperature=temperature)
        st.session_state.last_result = {
            'result': result,
            'elapsed_time': elapsed_time,
            'query': examples[1][1]
        }
        st.session_state.show_result = True
        st.rerun()

with col3:
    if st.button(examples[2][0], use_container_width=True, key="ex3"):
        st.session_state.user_query = examples[2][1]
        result, elapsed_time = analyze_query(examples[2][1], k=k, temperature=temperature)
        st.session_state.last_result = {
            'result': result,
            'elapsed_time': elapsed_time,
            'query': examples[2][1]
        }
        st.session_state.show_result = True
        st.rerun()

# 第二行
col4, col5, col6 = st.columns(3)

with col4:
    if st.button(examples[3][0], use_container_width=True, key="ex4"):
        st.session_state.user_query = examples[3][1]
        result, elapsed_time = analyze_query(examples[3][1], k=k, temperature=temperature)
        st.session_state.last_result = {
            'result': result,
            'elapsed_time': elapsed_time,
            'query': examples[3][1]
        }
        st.session_state.show_result = True
        st.rerun()

with col5:
    if st.button(examples[4][0], use_container_width=True, key="ex5"):
        st.session_state.user_query = examples[4][1]
        result, elapsed_time = analyze_query(examples[4][1], k=k, temperature=temperature)
        st.session_state.last_result = {
            'result': result,
            'elapsed_time': elapsed_time,
            'query': examples[4][1]
        }
        st.session_state.show_result = True
        st.rerun()

with col6:
    if st.button(examples[5][0], use_container_width=True, key="ex6"):
        st.session_state.user_query = examples[5][1]
        result, elapsed_time = analyze_query(examples[5][1], k=k, temperature=temperature)
        st.session_state.last_result = {
            'result': result,
            'elapsed_time': elapsed_time,
            'query': examples[5][1]
        }
        st.session_state.show_result = True
        st.rerun()

# 第三行
col7, col8, col9 = st.columns(3)

with col7:
    if st.button(examples[6][0], use_container_width=True, key="ex7"):
        st.session_state.user_query = examples[6][1]
        result, elapsed_time = analyze_query(examples[6][1], k=k, temperature=temperature)
        st.session_state.last_result = {
            'result': result,
            'elapsed_time': elapsed_time,
            'query': examples[6][1]
        }
        st.session_state.show_result = True
        st.rerun()

with col8:
    if st.button(examples[7][0], use_container_width=True, key="ex8"):
        st.session_state.user_query = examples[7][1]
        result, elapsed_time = analyze_query(examples[7][1], k=k, temperature=temperature)
        st.session_state.last_result = {
            'result': result,
            'elapsed_time': elapsed_time,
            'query': examples[7][1]
        }
        st.session_state.show_result = True
        st.rerun()

with col9:
    if st.button(examples[8][0], use_container_width=True, key="ex9"):
        st.session_state.user_query = examples[8][1]
        result, elapsed_time = analyze_query(examples[8][1], k=k, temperature=temperature)
        st.session_state.last_result = {
            'result': result,
            'elapsed_time': elapsed_time,
            'query': examples[8][1]
        }
        st.session_state.show_result = True
        st.rerun()

# ============================================
# 主要输入区域
# ============================================

st.markdown("---")
st.markdown("### 📝 或输入你自己的问题")

user_query = st.text_area(
    "请详细描述你遇到的情况（建议 20-200 字）",
    value=st.session_state.user_query,
    height=120,
    placeholder="例如：我在某个app上看到一个兼职机会，说需要充值激活账户才能开始工作...",
    key="main_input"
)

st.session_state.user_query = user_query

# ============================================
# 分析按钮
# ============================================

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    analyze_button = st.button(
        "🔍 分析问题",
        use_container_width=True,
        type="primary"
    )

with col2:
    if st.button("🗑️ 清空", use_container_width=True):
        st.session_state.user_query = ""
        st.session_state.last_result = None
        st.session_state.show_result = False
        st.session_state.conversation_history = []
        rag.clear_history()  # 清空 RAG 系统中的对话历史
        st.rerun()

with col3:
    show_stats = st.checkbox("📊 显示统计", value=False)

# ============================================
# 处理手动分析请求
# ============================================

if analyze_button and user_query:
    result, elapsed_time = analyze_query(user_query, k=k, temperature=temperature)
    st.session_state.last_result = {
        'result': result,
        'elapsed_time': elapsed_time,
        'query': user_query
    }
    st.session_state.show_result = True
    st.rerun()

elif analyze_button and not user_query:
    st.warning("⚠️ 请先输入你的问题")

# ============================================
# 显示结果
# ============================================

if st.session_state.show_result and st.session_state.last_result:
    result = st.session_state.last_result['result']
    elapsed_time = st.session_state.last_result['elapsed_time']

    st.markdown("---")
    st.success(f"✅ 分析完成（耗时 {elapsed_time:.2f} 秒）")

    # 创建两列布局
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 🤖 AI 安全顾问的建议")
        st.write(result['answer'])

    with col2:
        st.markdown("### 📈 分析数据")
        st.metric("响应耗时", f"{elapsed_time:.2f}s")
        st.metric("检索文档", len(result['retrieved_docs']))
        st.metric("答案长度", len(result['answer']))

    # 参考资料
    if result.get('retrieved_docs'):
        st.markdown("---")
        st.markdown("### 📚 参考资料（含结构化信息）")

        metadatas = result.get('metadata') or []
        for i, doc in enumerate(result['retrieved_docs'], 1):
            meta = metadatas[i - 1] if i - 1 < len(metadatas) else {}
            with st.expander(f"📄 资料 {i}"):
                st.write(doc)
                if meta:
                    st.markdown("---")
                    st.markdown("**结构化信息：**")
                    if meta.get("fraud_type"):
                        st.write(f"- 诈骗类型：{meta.get('fraud_type')}")
                    if meta.get("victim_group"):
                        st.write(f"- 受害群体：{meta.get('victim_group')}")
                    if meta.get("channel"):
                        st.write(f"- 主要渠道：{meta.get('channel')}")
                    if meta.get("money_flow"):
                        st.write(f"- 资金流向：{meta.get('money_flow')}")
                    if meta.get("risk_level"):
                        st.write(f"- 风险等级：{meta.get('risk_level')}")

    # 底部提示
    st.markdown("---")
    st.warning("""
    🚨 **安全提示：**
    
    如果你真的遇到了诈骗或可疑情况，请立即：
    - 📞 拨打报警电话：**110**
    - 📱 拨打反诈中心：**96110**
    - 👨‍💼 向学校保卫部门报告
    - ⚠️ 不要转账或提供个人信息
    """)

# ============================================
# 页脚和额外信息
# ============================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📖 了解更多
    - [校园诈骗类型](https://www.baidu.com)
    - [防范知识](https://www.baidu.com)
    - [法律条款](https://www.baidu.com)
    """)

with col2:
    st.markdown("""
    ### 📞 紧急联系
    - 🚨 报警：**110**
    - 🛡️ 反诈：**96110**
    - 🏫 学校保卫处
    - 👨‍💼 班主任
    """)

with col3:
    st.markdown("""
    ### 💻 技术信息
    - RAG 系统（检索增强生成）
    - 向量检索（本地离线）
    - 大模型生成（通义千问）
    - 实时响应（< 3 秒）
    """)

# ============================================
# 统计信息（可选）
# ============================================

if show_stats:
    st.markdown("---")
    st.markdown("### 📊 系统统计")
    
    # 动态获取统计信息
    def get_statistics():
        """获取系统统计信息"""
        # 获取向量库文档数量
        vector_count = 0
        try:
            if rag.collection:
                vector_count = rag.collection.count()
        except:
            vector_count = 0
        
        # 获取支持的诈骗类型数量
        fraud_types_count = 0
        try:
            import pandas as pd
            df = pd.read_csv('data/fraud_cases_optimized.csv', encoding='utf-8')
            fraud_types_count = df['fraud_type'].nunique()
        except:
            fraud_types_count = 0
        
        # 计算知识库规模
        knowledge_base_size = 0
        try:
            # 统计fraud_cases_optimized.csv中的条目
            df = pd.read_csv('data/fraud_cases_optimized.csv', encoding='utf-8')
            knowledge_base_size += len(df)
            # 统计fraud_policies.txt中的条目（假设每条政策为一个条目）
            with open('data/fraud_policies.txt', 'r', encoding='utf-8') as f:
                policies = f.read().split('\n\n')
                knowledge_base_size += len([p for p in policies if p.strip()])
        except:
            knowledge_base_size = 0
        
        return vector_count, fraud_types_count, knowledge_base_size
    
    # 获取统计数据
    vector_count, fraud_types_count, knowledge_base_size = get_statistics()
    
    # 显示系统统计信息
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("向量库文档", f"{vector_count} 个")
    
    with col2:
        st.metric("支持诈骗类型", f"{fraud_types_count} 种")
    
    with col3:
        st.metric("知识库规模", f"{knowledge_base_size}+ 条目")

st.markdown("""
---
<div style='text-align: center; color: #999; font-size: 0.9em;'>
    <p>校园反诈智能问答系统 | 河南大学 2026</p>
    <p>基于 RAG 技术的智能防诈系统</p>
</div>
""", unsafe_allow_html=True)