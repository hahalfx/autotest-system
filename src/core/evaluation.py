from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import dashscope
import os
from dotenv import load_dotenv
from typing import Literal

load_dotenv()


class SemanticCorrectness(BaseModel):
    score: float = Field(ge=0, le=1, description="语义正确性评分")
    comment: str = Field(description="评估意见")

class StateChangeConfirmation(BaseModel):
    score: float = Field(ge=0, le=1, description="状态变更确认评分") 
    comment: str = Field(description="评估意见")

class UnambiguousExpression(BaseModel):
    score: float = Field(ge=0, le=1, description="无歧义表述评分")
    comment: str = Field(description="评估意见")

class Assessment(BaseModel):
    semantic_correctness: SemanticCorrectness = Field(description="语义正确性评估")
    state_change_confirmation: StateChangeConfirmation = Field(description="状态变更确认评估")
    unambiguous_expression: UnambiguousExpression = Field(description="无歧义表述评估")
    overall_score: float = Field(ge=0, le=1, description="综合评分")
    valid: bool = Field(description="测试是否通过")
    suggestions: list[str] = Field(description="改进建议")

class EvaluationResult(BaseModel):
    assessment: Assessment = Field(description="完整评估结果")


def create_evaluation_chain(
    llm_provider: Literal[
        "openai", "anthropic", "aliyun_bailian", "openrouter"
    ] = "aliyun_bailian",
):
    """创建包含完整评估逻辑的LangChain流水线"""
    template = """作为车机系统测试专家，请严格评估：
    指令：{instruction}
    响应：{response}
    
    请按以下维度评估并返回严格JSON格式：
    1. semantic_correctness: 评分0-1和评估意见
    2. state_change_confirmation: 评分0-1和评估意见
    3. unambiguous_expression: 评分0-1和评估意见
    4. overall_score: 三个维度的平均分
    5. valid: 测试是否通过
    6. suggestions: 改进建议列表
    
    输出必须严格符合以下JSON结构：
    {{
      "assessment": {{
        "semantic_correctness": {{"score": 0-1, "comment": "..."}},
        "state_change_confirmation": {{"score": 0-1, "comment": "..."}},
        "unambiguous_expression": {{"score": 0-1, "comment": "..."}},
        "overall_score": 0.0-1.0,
        "valid": true/false,
        "suggestions": ["...", "..."]
      }}
    }}"""

    prompt = ChatPromptTemplate.from_template(template)

    if llm_provider == "aliyun_bailian":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        llm = ChatOpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            model="deepseek-v3",
        )
    elif llm_provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model="google/gemini-2.0-flash-001",
        )
    else:
        raise ValueError(f"不支持的LLM提供商: {llm_provider}")

    return prompt | llm | JsonOutputParser(pydantic_object=EvaluationResult)


class LLMEvaluator:
    def __init__(self, llm_provider: str = "openrouter"):
        self.eval_chain = create_evaluation_chain(llm_provider)

    def evaluate(self, instruction: str, response: str) -> EvaluationResult:
        """执行评估并返回结构化结果"""
        try:
            return self.eval_chain.invoke(
                {"instruction": instruction, "response": response}
            )
        except Exception as e:
            raise RuntimeError(f"评估过程中发生错误: {str(e)}") from e
