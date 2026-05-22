# Study Agent — 从手搓到框架的学习项目

## 项目结构

```
study_agent/
├── requirements.txt               # Python 依赖
└── handcraft_agent/               # 手搓版 Agent（纯手工实现，无框架）
    ├── __init__.py                # 包入口，让这个文件夹成为一个可导入的 Python 包
    ├── config.py                  # 配置：DeepSeek API 密钥、模型名、最大循环轮次、OpenAI 客户端实例
    ├── tools.py                   # 工具定义：天气数据库（硬编码字典）+ get_weather 函数 + 给 LLM 看的工具说明文本
    ├── prompts.py                 # 系统提示词：告诉 LLM 你是谁、用 ReAct 格式思考、有哪些工具可用
    ├── memory.py                  # 对话记忆：一个简单的类，把所有用户/助手消息按顺序存在列表里
    ├── react.py                   # ReAct 循环核心：解析 LLM 输出 → 执行工具 → 拼接观察结果 → 判断是否结束
    └── main.py                    # 启动入口：死循环等待用户输入，调用 react_loop，打印最终回答
```

## 各模块详解

| 文件 | 干了什么 | 对应手搓时的概念 |
|------|---------|----------------|
| `config.py` | 从环境变量读 API Key，创建 DeepSeek 客户端，设置 `MAX_TURNS=10` | 你的全局变量区 |
| `tools.py` | 一个字典存 8 个城市的天气，一个函数去查，一段文本告诉 LLM 怎么调用 | 你的「死工具」 |
| `prompts.py` | 一段很长的 system prompt，定义了 Thought→Action→Observation→Final Answer 的交互协议 | 你的「指定交互格式」 |
| `memory.py` | `ConversationMemory` 类，内部就是一个 list，每次对话往里 append 消息 | 你的「对话记忆拼接」 |
| `react.py` | `parse_react_output()` 用字符串前缀匹配解析 LLM 输出；`execute_tool()` 根据 action 名字分发到对应函数；`react_loop()` 是主循环，每轮调 LLM → 解析 → 执行工具 → 拼接 Observation → 检查是否结束 | 你的「ReAct 循环」 |
| `main.py` | `while True` 循环读用户输入，调用 react_loop，把结果打印出来 | 你的交互界面 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"

# 3. 运行
python -m handcraft_agent
```

## 学习路线

1. **先看懂手搓版**（`handcraft_agent/`）—— 理解 ReAct 循环、工具调用、记忆拼接的底层原理
2. **再学框架版**（`langgraph_agent/`，待添加）—— 同样的功能，用 LangGraph 重写，对比差异
3. **对比学习** —— 手搓版的痛点就是框架要解决的问题

## 手搓版已知痛点（这些问题框架会解决）

- `parse_react_output()` 用字符串前缀匹配，LLM 输出格式稍微不对就解析失败
- 工具写死在 `if action == "get_weather"` 里，加新工具必须改 react.py
- 记忆无限拼接，对话长了会超出 token 上限
- 没有重试机制，LLM 输出解析失败直接崩
- 没有状态持久化，程序退出后对话历史丢失
