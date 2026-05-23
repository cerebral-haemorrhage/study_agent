"""
================================================================================
  graph.py — LangGraph 核心，整个 Agent 的"大脑"
================================================================================

这个文件是整个框架版 Agent 最核心的部分。

【先忘掉代码，用生活例子理解三个概念】

  想象你在打客服电话：

  1. State（状态）= 通话录音带
     每次你说一句话，客服回一句话，都录在录音带上。
     录音带就是"对话状态"，一直跟着整通电话。

  2. Node（节点）= 一个"处理步骤"
     节点1: 客服听你说完 → 思考一下 → 决定"我自己回答"还是"转技术部门"
     节点2: 技术部门查资料 → 告诉客服结果
     这就是两个节点，每个节点完成一个任务。

  3. Edge（边）= "下一步去哪"的路线
     客服思考完 → 如果自己会答 → 挂电话(END)
     客服思考完 → 如果不会答 → 转技术部门 → 技术部门回话 → 客服再思考 → ...

【翻译成代码就是下面三个东西】

  积木1: AgentState   → 通话录音带（存所有消息）
  积木2: agent_node   → 客服听+思考
         tools_node   → 技术部门查资料
  积木3: should_continue → "下一步去哪"的判断
         build_graph  → 把节点和边拼成一张图

================================================================================
  图的形状（可视化）:
================================================================================

         用户提问
            │
            ▼
       ┌─────────┐
       │  agent  │  ← 客服：听听用户说了啥，自己能不能答？
       │  (思考)  │
       └────┬────┘
            │
       LLM 要不要调工具？
       ┌────┴────┐
       │         │
      要         不要
       │         │
       ▼         ▼
   ┌───────┐   END
   │ tools │   (直接回答用户，结束)
   │ (执行) │
   └───┬───┘
       │
       └────→ 回到 agent，带上工具结果重新思考
              (这就是 ReAct 的"循环"！)

================================================================================
  一张图对应手搓版的 react_loop() 整个函数
================================================================================
"""
from typing import Literal, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages      # 消息自动合并，不会覆盖
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage

from .config import llm                                # 前面配置好的 DeepSeek 客户端
from .tools import TOOLS                               # 工具列表 [get_weather]


# ============================================================
# 积木1: State — 图的"记忆"
# ============================================================
# 什么是 TypedDict？
#   普通 dict = {} 想塞什么塞什么，没有约束
#   TypedDict = 规定了"只能有 messages 这个字段，而且必须是 list[BaseMessage] 类型"
#   就像你写快递单，姓名/电话/地址都有固定的格子，不能乱写
#
# 手搓版里对应什么？
#   手搓版: react_memory = ""  (字符串拼消息)
#           ConversationMemory.messages = []  (列表存消息)
#   框架版: AgentState.messages = []  (统一成一个有类型的列表)
# Annotated[list, add_messages] 的意思是：
#   "messages 是一个列表，但更新时请不要覆盖，请把新消息追加到末尾"
# 没有这个注解 → 新消息会把旧消息整个替换掉 → SystemMessage 和 HumanMessage 消失
# 有了这个注解 → 新消息自动追加 → 历史不会丢
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 准备: 把工具"装到" LLM 上
# ============================================================
# llm 是一个"裸的"语言模型，只会聊天
# llm.bind_tools(TOOLS) 之后，LLM 就变成了"知道有工具可以调的语言模型"
#
# 手搓版里怎么做这件事的？
#   手搓版: 在 SYSTEM_PROMPT 里手写 TOOLS_DESC 文本
#   "工具名: get_weather, 参数: city"
#   LLM 看到这段文字，才知道它可以调 get_weather
#
# 框架版: bind_tools 自动完成！
#   把 @tool 装饰的函数签名 → 自动转成 JSON Schema
#   每次调 LLM 时，这个 Schema 会附在请求里
#   LLM 返回的不再是纯文本，而是 AIMessage 对象：
#     .content     → "我来帮你查一下北京天气"（自然语言）
#     .tool_calls  → [{"name": "get_weather", "args": {"city": "北京"}}]（结构化数据）
llm_with_tools = llm.bind_tools(TOOLS)


# ============================================================
# 积木2: 节点函数 — 图的"每一步"
# ============================================================

def agent_node(state: AgentState) -> dict:
    """
    ┌─────────────────────────────────────────────────────────┐
    │  节点1: agent 思考节点                                    │
    │                                                         │
    │  干什么: 把当前所有消息发给 LLM，让 LLM 想下一步怎么走       │
    │                                                         │
    │  输入: state = {"messages": [之前所有消息]}                │
    │  输出: {"messages": [LLM 生成的回复消息]}                  │
    │        LangGraph 会自动把这个消息追加到 state["messages"]   │
    └─────────────────────────────────────────────────────────┘

    对应手搓版里这段（handcraft_agent/react.py 第71-78行）:

        response = client.chat.completions.create(
            model=MODEL,
            messages=current_messages,
            temperature=0.7,
            max_tokens=1024,
        )
        llm_output = response.choices[0].message.content

    关键区别:
      手搓版返回的是纯字符串，比如:
        "Thought: 我需要查天气\nAction: get_weather\nAction Input: {\"city\": \"北京\"}"
      然后 parse_react_output() 用字符串匹配去拆

      框架版返回的是 AIMessage 对象:
        .content     = "我来帮你查一下北京的天气"
        .tool_calls  = [{"name": "get_weather", "args": {"city": "北京"}}]
      不用字符串解析！
    """
    # 取出当前所有消息
    MAX_MESSAGES = 6   # 设小一点方便观察裁剪效果
    messages = list(state["messages"])

    # 记忆窗口：只保留最近 N 条，但从前面裁剪
    # 注意：不能简单 messages[-N:]，可能把 ToolMessage 的配对 AIMessage 切掉
    # 这里从前面开始裁剪，保证 ToolMessage 前面一定有对应的 AIMessage(tool_calls)
    while len(messages) > MAX_MESSAGES:
        # 如果第1条是 ToolMessage，那第0条被裁剪了但不能丢（它是 AIMessage(tool_calls)）
        # 所以多留一条
        if len(messages) > 1 and messages[1].type == "tool":
            break  # 不能再裁了
        messages.pop(0)  # 从头部删除最旧的消息

    # 如果消息列表里还没有 SystemMessage，在最前面加一条
    has_system = any(m.type == "system" for m in messages)
    if not has_system:
        messages = [SystemMessage(
            content="你是一个智能助手。你可以使用工具来帮助用户。请逐步思考并回答用户问题。"
        )] + messages

    # 调用 LLM
    print(f"    📨 实际发给 LLM 的消息: {len(messages)} 条 (窗口={MAX_MESSAGES})")
    response = llm_with_tools.invoke(messages)

    # 只返回 AI 消息，SystemMessage 不写入 state（避免重复出现）
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """
    ┌─────────────────────────────────────────────────────────┐
    │  节点2: 工具执行节点                                      │
    │                                                         │
    │  干什么: LLM 说"帮我调 get_weather"，系统真的去调这个函数   │
    │                                                         │
    │  输入: state = {"messages": [..., LLM说调工具的回复]}     │
    │  输出: {"messages": [工具返回的结果]}                      │
    │        LangGraph 自动追加到 state["messages"]             │
    └─────────────────────────────────────────────────────────┘

    对应手搓版里这段（handcraft_agent/react.py 第88-94行）:

        if parsed["action"]:
            obs = execute_tool(parsed["action"], parsed["action_input"])
            react_memory += f"\nObservation: {obs}"

    手搓版: 用 if-else 判断 "action 是不是 get_weather"
    框架版: 遍历 tool_calls 列表，按 name 找到对应工具，直接 invoke
    """
    from langchain_core.messages import ToolMessage

    # 取最后一条消息 —— LLM 刚生成的，里面可能有 tool_calls
    last_message = state["messages"][-1]

    # 准备一个列表，装所有工具执行结果
    tool_messages = []

    # 遍历 LLM 想调用的所有工具
    # 比如 LLM 可能说: "帮我同时查北京和上海天气" → tool_calls 里就有两条
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]   # 例如 "get_weather"
        tool_args = tool_call["args"]   # 例如 {"city": "北京"}

        # 在工具列表里找到名字匹配的那个工具
        for tool in TOOLS:
            if tool.name == tool_name:
                # tool.invoke(tool_args) 就是真正执行 get_weather({"city": "北京"})
                result = tool.invoke(tool_args)

                # 把执行结果包装成 ToolMessage
                # ToolMessage 是一种特殊的消息类型，告诉 LLM "这是工具返回的结果"
                # tool_call_id 是每次工具调用的唯一 ID，LLM 用这个 ID 对应请求和结果
                tool_messages.append(
                    ToolMessage(
                        content=str(result),           # 工具返回的内容，如"晴天，25°C"
                        tool_call_id=tool_call["id"]  # 这次调用的唯一编号
                    )
                )
                break  # 找到了，不需要继续遍历

    # 返回工具结果
    return {"messages": tool_messages}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """
    ┌─────────────────────────────────────────────────────────┐
    │  路由器: 决定下一步                                       │
    │                                                         │
    │  看完 LLM 的回复，决定下一步是:                            │
    │    去 tools 节点（执行工具）                               │
    │    还是结束                                              │
    │                                                         │
    │  Literal["tools", "__end__"] 意思是:                      │
    │    这个函数只能返回 "tools" 或 "__end__" 这两个字符串之一   │
    └─────────────────────────────────────────────────────────┘

    对应手搓版里这段（handcraft_agent/react.py 第88-102行）:

        if parsed["action"]:
            obs = execute_tool(...)      # 有工具要调 → 继续循环
        if parsed["final_answer"]:
            return parsed["final_answer"] # 回答完了 → 退出循环

    框架版: 不做 if-else 跳转，而是返回一个"指令"
      return "tools"   → 告诉图 "下一个去 tools 节点"
      return "__end__" → 告诉图 "到此为止"
    """
    # 取最后一条消息
    last_message = state["messages"][-1]

    # hasattr(对象, "属性名") 检查对象有没有这个属性
    # 如果 LLM 回复了 tool_calls → 说明想调工具 → 去 tools 节点
    # 对应手搓版: if parsed["action"]:
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # LLM 没有 tool_calls → 说明它已经回答完了 → 结束
    # 对应手搓版: if parsed["final_answer"]: return
    return "__end__"


# ============================================================
# 搭图: 把积木拼起来
# ============================================================
def build_graph():
    """
    ┌─────────────────────────────────────────────────────────┐
    │  这个函数的全部作用就是:                                   │
    │    告诉 LangGraph "我的 Agent 长什么样，节点怎么连"         │
    │                                                         │
    │  它不执行任何逻辑，只是"画蓝图"                             │
    │  真正的执行是之后调用 graph.invoke() 时发生的              │
    └─────────────────────────────────────────────────────────┘

    这就是手搓版的 react_loop() 整个函数做的事！

    手搓版 react_loop 的操作序列:
      1. 初始化消息列表
      2. for turn in range(1, 11):          ← 循环
      3.     调 LLM                          ← 对应 agent_node
      4.     解析 LLM 输出                   ← 框架不需要这步
      5.     如果需要工具 → 执行 → 回到 3     ← 对应 tools_node + 回到 agent
      6.     如果答完了 → return              ← 对应 should_continue 返回 __end__

    框架版的操作序列（建图时只声明，不执行）:
      1. graph = StateGraph(AgentState)     → 新建一张图，规定状态格式
      2. graph.add_node("agent", ...)        → 声明图里有一个"客服思考"步骤
      3. graph.add_node("tools", ...)        → 声明图里有一个"执行工具"步骤
      4. graph.set_entry_point("agent")      → 设定入口: 从 agent 开始
      5. graph.add_conditional_edges(...)    → agent 之后去哪？交给 should_continue 判断
      6. graph.add_edge("tools", "agent")    → tools 之后一定回 agent
      7. return graph.compile()              → 把蓝图"编译"成可执行对象

    【重要概念解释】

      graph.add_node("名字", 函数)
        意思: "图里注册一个步骤，叫'名字'，执行时调用'函数'"
        类比: 乐高说明书里的一步"把红色积木放在底座上"

      graph.add_edge("A", "B")
        意思: "A 步骤完成后，必定去 B 步骤"
        类比: 第1步做完一定做第2步，没有例外

      graph.add_conditional_edges("A", 判断函数, {"结果1": "B", "结果2": "C"})
        意思: "A 步骤完成后，调用判断函数，根据返回值决定去哪"
        类比: 第1步做完，如果天气好去公园，天气不好去商场

      graph.compile()
        意思: 把蓝图"冻结"，变成一个能用的对象
        类比: 乐高图纸 → 实际拼好的模型
        之后每次 graph.invoke(...) 就是"让模型运转一次"
    """
    # ———— 第1步: 创建空图 ————
    # StateGraph(AgentState) 的意思是:
    #   "我要建一张图，图中流转的数据格式是 AgentState"
    graph = StateGraph(AgentState)

    # ———— 第2步: 往图里加节点(步骤) ————
    # add_node("名称", 函数)
    #   名称: 随便起，但后面 add_edge 要用这个名字
    #   函数: 当流程走到这个节点时，调用这个函数
    graph.add_node("agent", agent_node)   # 节点1: 客服思考
    graph.add_node("tools", tools_node)   # 节点2: 执行工具

    # ———— 第3步: 设定起点 ————
    # 用户输入后，第一个去哪个节点？→ agent
    graph.set_entry_point("agent")

    # ———— 第4步: 连边(路线) ————
    # add_conditional_edges("从哪来", 判断函数, {返回值: 去哪})
    #
    # 翻译成人话:
    #   从 agent 节点出来之后
    #   调用 should_continue 函数判断下一步
    #   如果 should_continue 返回 "tools"  → 去 tools 节点
    #   如果 should_continue 返回 "__end__" → 结束
    graph.add_conditional_edges(
        "agent",            # 从 agent 节点出来
        should_continue,    # 用这个函数判断去哪
        {
            "tools": "tools",   # 返回 "tools" → 跳到 tools 节点
            "__end__": END,     # 返回 "__end__" → 结束 (END 是 LangGraph 的常量)
        }
    )

    # ———— 第5步: 把 tools 连回 agent（形成循环！） ————
    # 工具执行完了 → LLM 要根据结果重新思考
    # 这就是 ReAct 的 "A"(Act 执行) → 回到 "R"(Reason 思考)
    graph.add_edge("tools", "agent")

    # ———— 第6步: 编译图 + 装记忆 ————
    # compile(checkpointer=MemorySaver()) 做了什么:
    #   MemorySaver 就是一个"对话记忆本"，把每次 invoke 后的 messages 存下来
    #   下次 invoke 时，同一个 thread_id 的消息自动恢复，不需要手动传历史
    #   对应手搓版的 ConversationMemory，但不用手动 append 了
    return graph.compile(checkpointer=MemorySaver())
