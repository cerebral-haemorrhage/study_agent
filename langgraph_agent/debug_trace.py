"""
================================================================================
  debug_trace.py — 调试追踪器
================================================================================

这个文件的作用只有一个: 让框架版运行时，打印出和手搓版 verbose 一样的逐步输出。

【为什么要单独一个文件？】
  框架版的 graph.invoke() 是个黑盒 —— 你丢消息进去，它吐出最终答案。
  中间 agent 和 tools 之间怎么循环的、LLM 每次说了什么、工具返回了什么，
  默认是不可见的。

  这个文件用一个 "包装器" 模式，在 graph 外面包一层，
  手动控制每一步，每一步都打印出来。

  就像你拆开一个自动咖啡机的外壳，看清楚它先磨豆 → 再压粉 → 再冲水的过程。

【包装器模式是什么？】
  class DebugWrapper:
      def invoke(self, state):
          # 手动调 agent_node → 打印结果
          # 手动判断要不要调工具
          # 手动调 tools_node → 打印结果
          # 循环...
          return state

  对外看起来和普通 graph 一样（都有 invoke 方法），
  但内部手动控制每一步并打印日志。
"""
from langchain_core.messages import AIMessage, ToolMessage
from .config import MAX_TURNS


def build_debug_graph():
    """
    工厂函数: 拿到编译好的 graph，包一层 DebugWrapper 再返回。

    调用者(main.py)不知道内部是 DebugWrapper 还是原始 graph，
    它只知道"这玩意儿有个 invoke 方法，能用"。
    这就是"面向接口编程"的好处。
    """
    from .graph import build_graph
    graph = build_graph()
    return DebugWrapper(graph)


class DebugWrapper:
    """
    调试包装器。

    用法和 graph 一样:
      result = debug_wrapper.invoke({"messages": [...]}, config={...})

    区别是: 它会打印每一步的详细信息。
    """

    def __init__(self, compiled_graph):
        """
        compiled_graph: 编译好的图（build_graph() 的返回值）
        虽然这里没用到 compiled_graph 对象本身，
        但保留它是为了方便以后扩展（比如直接用它跑然后截取中间状态）
        """
        self.graph = compiled_graph
        self.turn = 0   # 当前是第几轮

    def invoke(self, state: dict, config: dict = None) -> dict:
        """
        手动模拟 graph.invoke() 的完整流程，同时打印调试信息。

        流程就是手搓版 react_loop() 做的事:
          1. 调 agent_node: LLM 思考
          2. 调 should_continue: 判断是否需要工具
          3. 如果需要 → 调 tools_node: 执行工具 → 回到1
          4. 如果不需要 → 结束

        state 格式: {"messages": [消息列表]}
        config 格式: {"configurable": {"thread_id": "xxx"}}
        """
        self.turn = 0
        config = config or {}

        while True:
            self.turn += 1

            # 在调 agent 之前先"快照"一下当前状态
            # 这样 agent_node 失败时不会污染原始 state
            current_state = {"messages": list(state["messages"])}

            # ———— 防死循环: 超过 MAX_TURNS 就停 ————
            if self.turn > MAX_TURNS:
                print(f"\n⚠️ 已达最大轮次 {MAX_TURNS}，强制结束")
                break

            # ———— 打印轮次标题 ————
            print(f"\n{'='*50}")
            print(f"🔄 Round {self.turn}/{MAX_TURNS}")
            print(f"{'='*50}")

            # =============================================
            # 步骤1: 调 agent_node — LLM 思考
            # =============================================
            print("📍 进入 [agent] 节点 → 调用 LLM 思考...")

            # 动态导入（避免循环导入）
            from .graph import agent_node

            try:
                result = agent_node(current_state)
            except Exception as e:
                print(f"❌ agent 节点出错: {e}")
                raise

            # agent_node 返回 {"messages": [AIMessage对象]}
            # 把返回的消息追加到 state 里
            agent_msgs = result.get("messages", [])
            for msg in agent_msgs:
                state["messages"].append(msg)

            # ———— 打印 LLM 返回了什么 ————
            for msg in agent_msgs:
                if isinstance(msg, AIMessage):
                    # 打印 LLM 的自然语言回复（如果有的话）
                    content_preview = (msg.content or "")[:200]
                    if content_preview:
                        print(f"📝 LLM 思考: {content_preview}")

                    # 打印 LLM 想调用的工具（如果有的话）
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"🔧 LLM 想调用工具: {tc['name']}({tc['args']})")

            # =============================================
            # 步骤2: 判断下一步 — 继续还是结束？
            # =============================================
            from .graph import should_continue
            decision = should_continue(state)

            if decision == "__end__":
                print(f"✅ LLM 在第 {self.turn} 轮给出了最终答案，结束")
                break

            # =============================================
            # 步骤3: 调 tools_node — 执行工具
            # =============================================
            print("📍 进入 [tools] 节点 → 执行工具...")

            from .graph import tools_node

            try:
                result = tools_node(state)
            except Exception as e:
                print(f"❌ tools 节点出错: {e}")
                raise

            # tools_node 返回 {"messages": [ToolMessage对象]}
            tool_msgs = result.get("messages", [])
            for msg in tool_msgs:
                state["messages"].append(msg)

            # ———— 打印工具返回了什么 ————
            for msg in tool_msgs:
                if isinstance(msg, ToolMessage):
                    content_preview = (msg.content or "")[:200]
                    print(f"📊 工具结果: {content_preview}")

            # ← 循环回到 agent，带上工具结果重新思考

        # ———— 循环结束，提取最终回答 ————
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.content:
            print(f"\n🤖 最终回答: {last_msg.content[:300]}")

        return state
