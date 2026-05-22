"""主程序入口 —— 多轮对话"""
from .config import MAX_TURNS
from .memory import ConversationMemory
from .react import react_loop


def main():
    print("🤖 手搓 Agent 启动！（输入 quit 退出）")
    print(f"   支持的工具: 城市天气查询 (get_weather)")
    print(f"   ReAct 循环上限: {MAX_TURNS} 轮\n")

    memory = ConversationMemory()

    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break

            print("\n🤖 Agent 思考中...")
            answer = react_loop(user_input, memory, verbose=True)
            print(f"\n🤖 最终回答: {answer}")

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
