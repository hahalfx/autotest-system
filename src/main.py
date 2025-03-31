from core.evaluation import *
def main():
    llm_m = LLMEvaluator("aliyun_bailian");
    print(llm_m.evaluate("打开空调", "我没听清，请再说一遍"))
    print(llm_m.evaluate("打开空调", "空调已打开"))

if __name__ == "__main__":
    main()