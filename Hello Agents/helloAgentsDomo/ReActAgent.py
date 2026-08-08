from idlelib import history

from HelloAgentsLLM import HelloAgentsLLM
from Tools import ToolExecutor
import re
import os
# 定义搜索工具
from serpapi import SerpApiClient
from openai.resources.chat.completions import messages

# ReAct 提示词模板(需要有角色定义、工具清单、格式规约、动态上下文)
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


# 核心逻辑的实现
class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 2):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    # run方法是ReAct智能体的核心逻辑
    def run(self, question: str):
        """
            运行ReAct智能体来回答一个问题。
        """
        self.history = []
        current_step = 0
        while current_step < self.max_steps:
            current_step += 1
            print(f"步骤 {current_step}:")
            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )
            # 2. 调用LLM进行思考
            messages = [
                {"role": "user", "content": prompt}
            ]
            response_text = self.llm_client.think(messages)

            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # 工具调用与执行
            # 解析LLM的输出
            thought, action = self._parse_output(response_text)
            if thought:
                print(f"Thought: {thought}")
            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break
            # 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                # 先进行正则匹配
                match = re.match(r"Finish\[(.*)\]", action)

                # 判断是否匹配成功
                if match:
                    final_answer = match.group(1)
                else:
                    # 如果大模型没有按标准格式输出，你可以选择直接返回原始内容，或者给个默认提示
                    final_answer = action
                    # 或者打印警告方便调试：print(f"警告: 大模型返回格式异常 -> {action}")
                print(f"🎉 最终答案: {final_answer}")
                return final_answer
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                print("警告:未能解析出有效的Action，流程终止。")
                continue
            # 执行工具
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input)
            print(f"👀 观察: {observation}")
            # 将本轮的Action和观察添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
        # 如果超过最大步骤数，则结束
        print("流程终止，已超过最大步骤数。")
        return None

    """_parse_output： 负责从LLM的完整响应中分离出Thought和Action两个主要部分。
                 _parse_action： 负责进一步解析Action字符串，例如从 Search[华为最新手机] 中提取出工具
                 名 Search 和工具输入 华为最新手机"""

    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None

def search(query: str) -> str:
    """
       一个基于SerpApi的实战网页搜索引擎工具。
       它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            raise ValueError("SERPAPI_API_KEY 未定义。")
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn",  # 语言代码
        }
        client = SerpApiClient(params)
        results = client.get_dict()
        # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"

if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)

    clientLLM = HelloAgentsLLM()
    agent = ReActAgent(clientLLM, toolExecutor)
    agent.run("英伟达最新的GPU型号是什么")








