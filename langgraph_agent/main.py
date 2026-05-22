"""
================================================================================
  main.py — 程序入口
================================================================================

这就是你运行 python3 -m langgraph_agent 时执行的代码。
它干的事非常简单:

  1. 建好图（调用 build_graph）
  2. 进入死循环: 等用户输入 → 发给 Agent → 打印回答 → 回到1
  3. 用户输入 quit 时退出

其实就是手搓版 main.py 的框架版等价物。

【两种模式】

  调试模式（默认）:   python3 -m langgraph_agent
    运行时逐步打印每一步（和第1部分一样）
    适合: 学习、调试、看 Agent 内部怎么运作的

  安静模式:           python3 -m langgraph_agent --quiet
    只打印最终答案
    适合: 知道原理后，日常使用
"""
import sys
from langchain_core.messages import HumanMessage
# HumanMessage 就是"用户说的一句话"的封装
# 手搓版里是 {"role": "user", "content": "..."}
# 框架版里是 HumanMessage(content="...")


def main():
    # ———— 1. 根据参数选择模式 ————
    # sys.argv 是命令行的参数列表
    # 比如 python3 -m langgraph_agent --quiet
    # sys.argv = ["langgraph_agent/__main__.py", "--quiet"]
    quiet: bool = "--quiet" in sys.argv

    # ———— 2. 建图 ————
    if quiet:
        # 安静模式：用原始 graph，不打印中间过程
        from .graph import build_graph
        graph = build_graph()
    else:
        # 调试模式：用 DebugWrapper 包一层，会打印每一步
        from .debug_trace import build_debug_graph
        graph = build_debug_graph()

    # ———— 3. 打印欢迎信息 ————
    print("🤖 LangGraph Agent 启动！（输入 quit 退出）")
    print(f"   支持的工具: 城市天气查询 (get_weather)")
    if not quiet:
        print(f"   模式: 调试模式（逐步打印每一步）")
    print()

    # ———— 4. 初始化 ————
    # messages: 当前对话的所有消息
    # 手搓版里对应: ConversationMemory.messages = []
    messages = []

    # thread_id: 会话标识
    # 每次对话用一个独立的 thread_id，消息历史不会混在一起
    # 但现在我们只有单人单会话，所以固定一个就行
    thread_id = "default-conversation"

    # ———— 5. 主循环：等用户输入 → 调 Agent → 打印结果 ————
    while True:
        try:
            # 读用户输入
            user_input = input("\n👤 你: ").strip()

            # 空输入跳过
            if not user_input:
                continue

            # 退出命令
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break

            print("\n🤖 Agent 思考中...")

            # 把用户输入包装成消息，追加到列表
            # 手搓版: memory.add_user_message(user_input)
            messages.append(HumanMessage(content=user_input))

            # ———— 核心: 调 graph.invoke() ————
            # 这就是手搓版 react_loop() 做的事情
            #
            # graph.invoke(
            #     {"messages": messages},     ← 输入: 当前所有消息
            #     config={...}                 ← 配置: 用哪个 thread
            # )
            #
            # 返回值是一个新的 state，里面的 messages 包含了完整对话
            result = graph.invoke(
                {"messages": messages},
                config={"configurable": {"thread_id": thread_id}},
            )

            # ———— 提取最终回答 ————
            # result["messages"][-1] 是最后一条消息
            # 正常情况它是 AIMessage，里面 .content 就是 Agent 的回答
            final_message = result["messages"][-1]
            if hasattr(final_message, "content") and final_message.content:
                answer = final_message.content
            else:
                answer = str(final_message)

            # 安静模式下打印最终答案（调试模式已经在 DebugWrapper 里打印了）
            if quiet:
                print(f"\n🤖 最终回答: {answer}")

            # 把更新后的消息列表保存，下一轮继续用
            messages = result["messages"]

        except KeyboardInterrupt:
            # 用户按了 Ctrl+C
            print("\n👋 再见！")
            break
        except Exception as e:
            # 其他错误
            print(f"\n❌ 出错了: {e}")
            import traceback
            traceback.print_exc()
