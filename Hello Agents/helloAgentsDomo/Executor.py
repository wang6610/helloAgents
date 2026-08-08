from HelloAgentsLLM import HelloAgentsLLM

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""

class Executor:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        """
            根据计划，逐步执行并解决问题。
        """
        history = ""
        print("开始执行问题...")
        for i, step in enumerate(plan):
            print(f"步骤 {i + 1}: {step}")
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step
            )
            messages = [
                {"role": "user", "content": prompt}
            ]
            response_text = self.llm_client.think(messages=messages) or ""
            history += f"步骤 {i + 1}: {step}\n结果: {response_text}\n\n"
            response_text = self.llm_client.think(messages=messages) or ""
        final_answer = response_text
        return final_answer
