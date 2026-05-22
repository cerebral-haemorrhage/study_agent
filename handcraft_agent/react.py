"""ReAct 循环核心逻辑"""
import json
from .config import client, MODEL, MAX_TURNS
from .prompts import SYSTEM_PROMPT
from .tools import get_weather
from .memory import ConversationMemory


def parse_react_output(text: str) -> dict:
    """从 LLM 输出中解析 Thought / Action / Action Input / Final Answer"""
    result = {"thought": "", "action": None, "action_input": None, "final_answer": ""}

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("Thought:"):
            result["thought"] = line[len("Thought:"):].strip()
        elif line.startswith("Action:"):
            action_val = line[len("Action:"):].strip()
            result["action"] = None if action_val.lower() == "none" else action_val
        elif line.startswith("Action Input:"):
            raw = line[len("Action Input:"):].strip()
            if raw.lower() != "none":
                try:
                    result["action_input"] = json.loads(raw)
                except json.JSONDecodeError:
                    result["action_input"] = raw
        elif line.startswith("Final Answer:"):
            result["final_answer"] = line[len("Final Answer:"):].strip()

    return result


def execute_tool(action: str, action_input: dict) -> str:
    """执行工具调用"""
    if action == "get_weather":
        city = action_input.get("city", "") if isinstance(action_input, dict) else str(action_input)
        return get_weather(city)
    return f"未知工具: {action}"


def react_loop(user_query: str, memory: ConversationMemory, verbose: bool = True) -> str:
    """
    ReAct 循环：
    - 最多 MAX_TURNS 轮
    - LLM 输出 Final Answer 就提前退出
    - 每轮 Observation 拼回消息中作为记忆
    """

    # 初始化：system prompt + 历史对话 + 当前问题
    chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    chat_messages.extend(memory.get_messages())
    chat_messages.append({"role": "user", "content": user_query})

    # ReAct 循环内的记忆：逐步拼接 Thought/Action/Observation
    react_memory = ""

    for turn in range(1, MAX_TURNS + 1):
        if verbose:
            print(f"\n{'='*50}")
            print(f"🔄 Round {turn}/{MAX_TURNS}")
            print(f"{'='*50}")

        # 把 ReAct 记忆注入
        current_messages = chat_messages.copy()
        if react_memory:
            current_messages.append({
                "role": "assistant",
                "content": react_memory + "\nThought: 我接下来应该做什么？"
            })

        # 调用 LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=current_messages,
            temperature=0.7,
            max_tokens=1024,
        )
        llm_output = response.choices[0].message.content

        if verbose:
            print(f"📝 LLM 输出:\n{llm_output}")

        # 解析 → 拼接记忆
        parsed = parse_react_output(llm_output)
        react_memory += f"\n{llm_output}"

        # 执行工具
        if parsed["action"]:
            obs = execute_tool(parsed["action"], parsed["action_input"])
            if verbose:
                print(f"🔧 执行工具 [{parsed['action']}]: "
                      f"{json.dumps(parsed['action_input'], ensure_ascii=False)}")
                print(f"📊 Observation: {obs}")
            react_memory += f"\nObservation: {obs}"

        # 提前退出
        if parsed["final_answer"]:
            if verbose:
                print(f"\n✅ LLM 在第 {turn} 轮给出了最终答案，提前退出循环")
            memory.add_user_message(user_query)
            memory.add_assistant_message(parsed["final_answer"])
            return parsed["final_answer"]

    # 超时兜底
    if verbose:
        print(f"\n⚠️ 已达最大轮次 {MAX_TURNS}，要求 LLM 总结")

    current_messages = chat_messages.copy()
    current_messages.append({
        "role": "assistant",
        "content": react_memory + "\nThought: 已达最大轮次，我必须给出最终答案"
    })
    current_messages.append({
        "role": "user",
        "content": "请基于目前已获取的信息，直接给出 Final Answer。"
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=current_messages,
        temperature=0.7,
        max_tokens=512,
    )
    final_output = response.choices[0].message.content

    if verbose:
        print(f"📝 最终输出:\n{final_output}")

    memory.add_user_message(user_query)
    memory.add_assistant_message(final_output)
    return final_output
