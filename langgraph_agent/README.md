# LangGraph 版 Agent —— 框架实现

## 项目结构

```
langgraph_agent/
├── __init__.py          # 包入口
├── __main__.py          # 支持 python -m 运行
├── config.py            # DeepSeek 客户端（用 LangChain 的 ChatOpenAI 封装）
├── tools.py             # 天气工具（用 @tool 装饰器替代硬编码 if-else）
├── graph.py             # StateGraph 图定义（替代手搓的 for 循环 + 字符串解析）
│                       #   → 大量中文注释，逐行对照手搓版
├── debug_trace.py       # 调试追踪器，运行时逐步打印每次 LLM 调用和工具执行
├── main.py              # 多轮对话入口（默认调试模式，--quiet 切换安静模式）
└── README.md            # 本文件
```

## 各模块详解

| 文件 | 干了什么 | 对应手搓版 |
|------|---------|-----------|
| `config.py` | 用 `ChatOpenAI` 封装 DeepSeek，不需要手动调 `openai.chat.completions.create` | `handcraft_agent/config.py` |
| `tools.py` | 用 `@tool` 装饰器定义工具，LLM 自动获取函数签名和参数类型 | `handcraft_agent/tools.py` |
| `graph.py` | 用 `StateGraph` 定义 ReAct 循环：agent 节点(思考) + tools 节点(执行) + 条件边(判断是否结束)。**每行都有对照手搓版的中文注释** | `handcraft_agent/react.py` |
| `debug_trace.py` | 包装 graph，运行时逐步打印每一步：第几轮、进了哪个节点、LLM 说了什么、工具调了什么 | `handcraft_agent/react.py` 的 verbose 打印 |
| `main.py` | 调用 `graph.invoke()`，传入消息即可，框架自动管理循环和记忆。默认启用调试模式 | `handcraft_agent/main.py` |

## 核心对比：手搓版 vs 框架版

| 功能 | 手搓版 | 框架版 |
|------|--------|--------|
| LLM 调用 | `client.chat.completions.create(...)` 手动调 API | `llm.invoke(messages)` LangChain 封装 |
| 工具注册 | 硬编码 `if action == "get_weather"` | `@tool` 装饰器 + `llm.bind_tools(TOOLS)` |
| 工具执行 | `execute_tool()` 手写 if-else 派发 | `ToolNode(TOOLS)` 自动派发 |
| 输出解析 | `parse_react_output()` 字符串前缀匹配 | LLM 返回结构化 `AIMessage`，自带 `tool_calls` 列表 |
| 循环控制 | `for turn in range(1, 11)` + `parsed["final_answer"]` 手动退出 | `should_continue` 条件边，框架自动循环 |
| 消息管理 | `react_memory += f"\n{llm_output}"` 字符串拼接 | `AgentState.messages` 列表，`add_messages` 自动合并 |
| 加新工具 | 改 `tools.py` 加函数 + 改 `react.py` 加 if 分支 | 只改 `tools.py` 加一个 `@tool` 函数并加入 `TOOLS` 列表 |

## 图结构可视化

```
     ┌──────────┐
START→│  agent   │←──────────┐
     └────┬─────┘            │
          │                   │
    LLM 要不要调工具？         │
     ┌────┴────┐              │
     │要       │不要           │
     ▼         ▼              │
  ┌──────┐  ┌────┐           │
  │tools │  │END │           │
  └──┬───┘  └────┘           │
     │                        │
     └────────────────────────┘
       (工具结果喂回 agent)
```

## 快速开始

```bash
# 安装依赖（需要 langchain + langgraph）
pip install langchain langchain-openai langgraph

# 设置 API Key
export DEEPSEEK_API_KEY="sk-你的key"

# 运行
python3 -m langgraph_agent
```

## 和手搓版对比运行

```bash
# 终端1：跑手搓版
python3 -m handcraft_agent

# 终端2：跑框架版
python3 -m langgraph_agent
```
