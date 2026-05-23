"""入口 —— 多轮对话

MemorySaver 版本：
  每次只传新的用户消息，历史由 checkpointer 自动管理。
"""
from langchain_core.messages import HumanMessage
from .graph import build_graph
from .tools import TOOLS


def print_messages(title: str, messages: list):
    """打印消息列表的辅助函数"""
    print(f"\n{'─'*50}")
    print(f"📋 {title} ({len(messages)} 条)")
    print(f"{'─'*50}")
    if not messages:
        print("  (空)")
        return
    for i, msg in enumerate(messages):
        role = msg.__class__.__name__.replace("Message", "")
        content_preview = str(getattr(msg, "content", "") or "")
        tool_calls = getattr(msg, "tool_calls", None)
        extra = ""
        if tool_calls:
            names = [(tc["name"], tc["args"]) for tc in tool_calls]
            extra = f" → tool_calls: {names}"
        print(f"  [{i}] {role}: {content_preview[:80]}{'...' if len(content_preview)>80 else ''}{extra}")
    print()


def main():
    graph = build_graph()

    print("🤖 LangGraph Agent 启动！（输入 quit 退出）")
    print("   支持的工具:")
    for tool in TOOLS:
        desc = tool.description.strip().split("\n")[0]
        print(f"     - {tool.name}: {desc}")
    print()

    thread_id = "default-conversation"

    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break

            # ———— 调用前：查看 MemorySaver 里存了什么 ————
            config = {"configurable": {"thread_id": thread_id}}
            prev_state = graph.get_state(config)
            if prev_state and prev_state.values:
                print_messages("调用前 — MemorySaver 中的历史", prev_state.values.get("messages", []))
            else:
                print("\n📋 调用前 — MemorySaver: (空，首次对话)")

            # ———— 调 Agent ————
            print("🤖 Agent 思考中...")
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

            # ———— 调用后：打印更新后的完整状态 ————
            print_messages("调用后 — MemorySaver 中的完整状态", result["messages"])

            # ———— 最终回答 ————
            final_message = result["messages"][-1]
            answer = final_message.content if hasattr(final_message, "content") else str(final_message)
            print(f"🤖 最终回答: {answer}")

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            import traceback
            traceback.print_exc()
