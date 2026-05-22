"""对话记忆 —— 简单拼接存储"""


class ConversationMemory:
    """把所有消息按顺序拼接存储"""

    def __init__(self):
        self.messages: list[dict] = []

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def add_system_message(self, content: str):
        self.messages.append({"role": "system", "content": content})

    def get_messages(self) -> list[dict]:
        return self.messages

    def clear(self):
        self.messages = []
