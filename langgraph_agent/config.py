"""
================================================================================
  config.py — 配置中心
================================================================================

这个文件就一件事：创建一个"能跟 DeepSeek 说话的 LLM 对象"。

【超级新手理解】
  比如你想叫外卖，你需要知道三件事：
    1. 外卖 App 叫什么？      → model（模型）
    2. 你的账号密码是多少？     → api_key（密钥）
    3. 外卖 App 的网址是什么？ → base_url（地址）

  这里的 api_key 和 base_url 就是 DeepSeek 的"门牌号和钥匙"。
  拿到 llm 对象之后，你只需要 llm.invoke("你好") 就能跟 AI 对话。

【手搓版对比】
  手搓版: client = OpenAI(api_key=..., base_url=...)
          response = client.chat.completions.create(model=..., messages=...)
  框架版: llm = ChatOpenAI(model=..., api_key=..., base_url=...)
          response = llm.invoke(messages)
"""
import os
from langchain_openai import ChatOpenAI

# ———— 第1步：从环境变量读取 API Key ————
# os.getenv("名字", "默认值") 的意思是：
#   先去系统环境变量里找有没有 DEEPSEEK_API_KEY
#   如果有 → 用它
#   如果没有 → 用 "your-api-key-here" 这个默认值（但这个是假的，用不了）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")

# DeepSeek 的服务器地址（不用改）
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 用哪个模型。deepseek-chat 是 DeepSeek 的通用对话模型（最新的 V3.1）
MODEL = "deepseek-chat"

# 最多让 Agent 循环几轮（防止死循环）
MAX_TURNS = 10

# ———— 第2步：创建 LLM 对象 ————
# ChatOpenAI 是 LangChain 提供的一个"统一接口"
# 虽然名字带 OpenAI，但改一下 base_url 就能接 DeepSeek
# 就像安卓手机的 Type-C 充电口，换个充电头（base_url）就能用
llm = ChatOpenAI(
    model=MODEL,                    # 用什么模型
    api_key=DEEPSEEK_API_KEY,       # API 密钥（身份凭证）
    base_url=DEEPSEEK_BASE_URL,     # 服务器地址
    temperature=0.7,                # "创造力"参数：0=死板, 1=天马行空, 0.7=适中
    max_tokens=1024,                # 每次回复最多多少个 token（约 500-700 个汉字）
)
