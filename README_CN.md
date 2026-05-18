# 基于 RAG 技术的本地手册 QA 系统

[日本語](README.md) | [English](README_EN.md) | 中文

## 概述

本项目使用 RAG（检索增强生成）技术，让你可以用自然语言对本地存储的手册和文档进行提问。嵌入模型使用 ZhipuAI API，回答生成使用性价比极高的 DeepSeek API。配备基于 Streamlit 的 Web UI，在浏览器中即可直观操作。

### Web UI 主要功能

- **聊天式对话**：输入问题后，RAG 检索相关文档，DeepSeek 流式输出回答。会话历史在会话期间保留。
- **回答风格选择**：可选概要报告 / 学习指南 / 博客文章 / 自定义格式四种风格，每种风格应用专用提示词模板。自定义格式可自由输入任意指示。
- **回答长度设置**：可选 200 / 400 / 800 令牌，反映到 `max_tokens` 参数。
- **检索结果可视化**：Top-K 文本块展开显示，包含来源文件名、相似度分数（余弦相似度）、关键词高亮。
- **知识库管理**：将文件放入 `data/` 目录即可在启动时自动向量化。Web UI 支持拖放添加/删除文件，仅对变更文件进行增量索引。
- **问题样例**：根据知识库中的文件名自动生成问题，一键点击即可提问。

### 运行结果

![操作演示](images/result1.gif)

![结果2](images/result2.png)

---

## 使用方法

### 1. 环境配置（一键安裝）

需要 Python 3.10 及以上版本。使用以下命令即可一键完成虚拟环境创建与依赖包安装。

**Windows (PowerShell):**
```bash
python -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt
```

**macOS / Linux：**
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

> 如果使用 `uv`，运行 `uv venv .venv && uv pip install -r requirements.txt` 可更快速地完成安装。

### 2. 获取 API 密钥

本项目使用 **ZhipuAI API** 进行嵌入，使用 **DeepSeek API** 生成回答。DeepSeek 兼具高回答能力与低价格，性价比突出。在各官网注册账号后获取 API 密钥，写入 `.env` 文件：

```
ZHIPUAI_API_KEY='你的ZhipuAI API密钥'
DEEPSEEK_API_KEY='你的DeepSeek API密钥'
```

### 3. 准备数据

将待提问的 `.md` / `.txt` / `.pdf` 文件放入 `data/` 目录。

![数据配置](images/Localdata.png)

### 4. 启动

```bash
streamlit run app.py --server.port 8501
```

浏览器打开 `http://localhost:8501` 即可看到聊天界面。在侧边栏设置风格和长度，然后输入问题即可。

---

## RAG 机制

RAG（检索增强生成）按以下三个步骤运行：

1. **Retrieve（检索）**：将用户问题向量化，在知识库的文档片段（chunk）中通过余弦相似度检索 Top-K 个最相关的片段。
2. **Augment（增强）**：将检索结果作为上下文嵌入提示词，传递给 LLM。
3. **Generate（生成）**：LLM 参考上下文生成回答，即使模型未直接学习过的信息也能应对。

![RAG工作流](images/RAG_workflow.png)
*图片来源：https://blog-ja.allganize.ai/allganize_rag-1/*

![RAG结构](images/Rag_strucutre.png)

---

## 代码执行示例

通过 Python 脚本直接使用 RAG 的流程：

```python
from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.Embeddings import ZhipuEmbedding
from RAG.LLM import DeepSeekAIChat

# 读取并分割文档
docs, sources = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)

# 向量化并保存
vector = VectorStore(docs, sources)
vector.get_vector(EmbeddingModel=ZhipuEmbedding())
vector.persist(path='storage')

# 加载已保存的向量，回答问题
vector = VectorStore()
vector.load_vector('./storage')

question = '低周波治療器的使用方法是？'
content = vector.query(question, EmbeddingModel=ZhipuEmbedding(), k=1)

chat = DeepSeekAIChat(model='deepseek-chat')
print(chat.chat(question, [], content[0]['text']))
```

---

## 实现细节

### 向量化（Embeddings）

支持 `zhipu` / `jina` / `openai` 三种嵌入模型。如需添加其他模型，继承 `BaseEmbeddings` 并实现 `get_embedding()` 即可。

```python
class BaseEmbeddings:
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1, vector2) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        return 0 if not magnitude else dot_product / magnitude
```

### 向量检索（VectorBase）

将文档片段与向量以 JSON 格式存储在本地，使用 Numpy 计算余弦相似度。轻量、易理解。

```python
def query(self, query, EmbeddingModel, k=1):
    query_vector = EmbeddingModel.get_embedding(query)
    scores = np.array([self.get_similarity(query_vector, v) for v in self.vectors])
    top_indices = scores.argsort()[-k:][::-1]
    return [{"text": self.document[i], "score": float(scores[i]), "source": self.sources[i]}
            for i in top_indices]
```

> 本实现仅供学习用途。生产环境建议使用专业的向量数据库（如 Chroma、Pinecone 等）。

### LLM 模型

支持 OpenAI / InternLM2 / ZhipuAI / DeepSeek。如需添加其他模型，继承 `BaseModel` 即可。

```python
class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass
```

---

## 参考文献

<details><summary>展开</summary>

| 名称 | 链接 |
|------|------|
| Hand-on-RAG | https://github.com/SmartFlowAI/Hand-on-RAG |
| When Large Language Models Meet Vector Databases: A Survey | http://arxiv.org/abs/2402.01763 |
| Retrieval-Augmented Generation for Large Language Models: A Survey | https://arxiv.org/abs/2312.10997 |
| Learning to Filter Context for Retrieval-Augmented Generation | http://arxiv.org/abs/2311.08377 |
| In-Context Retrieval-Augmented Language Models | https://arxiv.org/abs/2302.00083 |

</details>
