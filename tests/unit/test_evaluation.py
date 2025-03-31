import os
import pytest
from src.core.evaluation import LLMEvaluator, EvaluationResult
from unittest.mock import patch

class TestLLMEvaluator:
    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """模拟环境变量配置"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
        monkeypatch.setenv("ALIYUN_ACCESS_KEY_ID", "aliyun-test-id")
        monkeypatch.setenv("ALIYUN_ACCESS_KEY_SECRET", "aliyun-test-secret")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-openrouter")

    @pytest.fixture
    def mock_llm_response(self):
        """模拟LLM标准响应"""
        return {
            "valid": True,
            "reason": "测试通过",
            "confidence": 0.9
        }

    def test_openai_evaluation(self, responses, mock_llm_response):
        """测试OpenAI评估流程"""
        responses.post("https://api.openai.com/v1/chat/completions", 
                     json={"choices": [{"message": {"content": str(mock_llm_response)}}]})

        evaluator = LLMEvaluator("openai")
        result = evaluator.evaluate("打开空调", "空调已开启")
        
        assert isinstance(result, EvaluationResult)
        assert result.valid == mock_llm_response["valid"]
        assert result.reason == mock_llm_response["reason"]

    def test_aliyun_bailian_evaluation(self, responses):
        """测试阿里云百炼集成"""
        test_response = {
            "valid": False,
            "reason": "指令不明确",
            "confidence": 0.3
        }
        responses.post("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                     json={"choices": [{"message": {"content": str(test_response)}}]})

        evaluator = LLMEvaluator("aliyun_bailian")
        result = evaluator.evaluate("打开车窗", "请再说一遍")
        
        assert result.valid is False
        assert "不明确" in result.reason

    def test_error_handling(self):
        """测试异常处理"""
        with patch('src.core.evaluation.create_evaluation_chain', side_effect=Exception("API错误")):
            evaluator = LLMEvaluator("openai")
            with pytest.raises(RuntimeError) as excinfo:
                evaluator.evaluate("测试指令", "测试响应")
            
            assert "评估过程中发生错误" in str(excinfo.value)

    def test_evaluation_result_model(self):
        """测试评估结果模型验证"""
        with pytest.raises(ValueError):
            EvaluationResult(valid=True, reason="测试", confidence=1.1)  # 超过置信度上限

        valid_result = EvaluationResult(valid=False, reason="测试", confidence=0.5)
        assert valid_result.dict() == {
            "valid": False,
            "reason": "测试",
            "confidence": 0.5
        }
