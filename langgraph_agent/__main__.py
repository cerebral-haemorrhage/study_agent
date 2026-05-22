"""
当用户执行 python3 -m langgraph_agent 时，
Python 会自动找 __main__.py 并执行。

这里只是简单转发到 main.py 的 main() 函数。
"""
from langgraph_agent.main import main

main()
