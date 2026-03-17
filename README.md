# 校园反诈智能问答系统

基于 RAG（检索增强生成）技术的校园诈骗识别和防范助手，使用中文 Embedding 模型和混合检索策略，提供准确、快速的诈骗风险评估和防范建议。

## 功能特点

- 🔍 **智能检索**：采用混合检索策略（向量+关键词），F1 分数达 0.8232
- 🤖 **大模型生成**：基于通义千问大模型，提供专业、准确的回答
- 📊 **多类别识别**：支持 16 种常见校园诈骗类型
- ⚡ **快速响应**：平均响应时间 < 20 秒
- 🔄 **多轮对话**：支持上下文管理，处理追问和相关问题
- 🌐 **国内优化**：使用国内镜像源，下载速度快

## 技术栈

- **前端**：Streamlit
- **后端**：Python
- **Embedding 模型**：BAAI/bge-small-zh-v1.5（130MB）
- **向量存储**：ChromaDB
- **大模型**：通义千问（DashScope API）
- **检索策略**：混合检索（向量+关键词）+ RRF 融合

## 支持的诈骗类型

1. 兼职刷单
2. 假冒教务处
3. 校园贷
4. 冒充辅导员
5. 虚假实习
6. 快递理赔诈骗
7. 网络直播诈骗
8. 游戏账号诈骗
9. 虚拟人设交友
10. AI换脸诈骗
11. 电信诈骗
12. 网络购物诈骗
13. 投资理财诈骗
14. 信息诈骗
15. 话费诈骗
16. 求职诈骗

## 性能指标

### 检索性能（保留集）
- **F1 分数**：0.8232（优秀）
- **精度**：0.9697
- **召回**：0.7455

### 端到端性能
- **回答准确率**：76.67%
- **完全正确**：76.67%
- **部分正确**：0.00%
- **完全错误**：23.33%

## 知识库规模

- **诈骗案例**：203 条
- **诈骗类型**：16 种
- **向量存储**：210 个文本块

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 文件为 `.env`，并填写 API 密钥：

```env
# 通义千问 API 密钥
DASHSCOPE_API_KEY=your_api_key_here

# Embedding 模型路径（可选）
EMBEDDING_MODEL_PATH=D:\models\bge-small-zh-v1.5
```

### 3. 构建知识库

```bash
python scripts/build_kg.py
```

### 4. 启动 Web 应用

```bash
streamlit run app.py
```

### 5. 运行评估

```bash
# 运行基准对比实验
python scripts/experiments/baseline_comparison.py

# 运行端到端评估
python scripts/evaluate.py
```

## 项目结构

```
anti-fraud-rag-complete/
├── src/                  # 核心代码
│   ├── __init__.py
│   └── rag_system.py      # RAG 系统核心
├── scripts/               # 脚本
│   ├── experiments/       # 实验脚本
│   │   ├── ablation_study.py
│   │   ├── baseline_comparison.py
│   │   └── tune_alpha.py
│   ├── build_kg.py        # 构建知识库
│   └── evaluate.py        # 端到端评估
├── data/                  # 数据
│   ├── fraud_cases_optimized.csv  # 诈骗案例
│   ├── fraud_policies.txt         # 反诈政策
│   ├── test_dev.csv               # 开发集
│   └── test_holdout.csv           # 保留集
├── output/                # 实验结果
├── chroma_db/             # 向量存储
├── app.py                 # Streamlit 应用
├── hybrid_retriever.py    # 混合检索器
├── requirements.txt       # 依赖
└── .env.example           # 环境变量示例
```

## 核心功能

### 1. 智能检索

采用混合检索策略，结合向量检索和关键词检索，通过 RRF（Reciprocal Rank Fusion）融合，平衡精度和召回。

### 2. 风险评估

对用户输入的情况进行风险评估，识别诈骗类型，提供详细的警示信号和防范建议。

### 3. 多轮对话

支持上下文管理，能够处理用户的追问，如"那这种诈骗怎么举报？"等相关问题。

### 4. 知识库扩展

目前知识库扩展方式为：编辑 `data/fraud_cases_optimized.csv` 添加新案例/关键词，然后重新运行 `python scripts/build_kg.py` 构建向量库。

## 常见问题

### 1. 模型下载失败

- 确保网络连接正常
- 检查 `.env` 文件中的 `EMBEDDING_MODEL_PATH` 配置
- 系统会自动尝试使用在线模型作为回退

### 2. API 密钥错误

- 确保在 `.env` 文件中正确配置 `DASHSCOPE_API_KEY`
- 确保 API 密钥以 `sk-` 开头

### 3. 向量库构建失败

- 确保 `data/fraud_cases_optimized.csv` 文件存在且格式正确
- 确保 ChromaDB 依赖已正确安装

## 性能优化

1. **政策文件软惩罚**：将政策文件排到检索结果末尾，优先返回具体案例
2. **校园贷专属关键词**：添加高区分度关键词，提高校园贷识别准确率
3. **RRF 融合**：使用标准 RRF 算法，平衡向量和关键词检索

## 未来计划

- [ ] 增加更多诈骗类型和案例
- [ ] 优化模型推理速度
- [ ] 添加用户反馈机制
- [ ] 支持多语言
- [ ] 部署到云端

## 贡献

欢迎贡献代码、报告问题或提出建议！

## 许可证

MIT License

## 联系方式

如有问题，请联系项目维护者。

---

**注意**：本系统仅用于教育和防范诈骗目的，不能替代专业的法律和安全建议。如遇真实诈骗，请立即报警。
