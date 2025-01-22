# %load_ext autoreload
# %autoreload 2
# 为简单了解CAPE-FND模型方法的流程而设计的简单代码流程

import sys
import os
import dspy
import openai
import pandas as pd
import json
import re
import time
from dspy import OpenAI
from google.colab import drive  # 对于Colab的用户设置
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
import contextlib

# 环境设置与数据加载
drive.mount('/content/drive')

# 加载假新闻数据集
def load_fnd_data(file_path, sample_size=100):
    df = pd.read_csv(file_path)
    df_sample = df.iloc[:sample_size]
    
    return [dspy.Example(
        article=row['content'],
        label="fake" if row['label'] == 1 else "real"
    ).with_inputs('article') for _, row in df_sample.iterrows()]

# 数据集路径
train_path = "/content/drive/MyDrive/datasets/fnd/train.csv"
test_path = "/content/drive/MyDrive/datasets/fnd/test.csv"

train = load_fnd_data(train_path)
dev = load_fnd_data(test_path)

# 核心检测模块
class CAPEFND(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_constraints = dspy.ChainOfThought("article -> constraints")
        self.generate_background = dspy.ChainOfThought("article -> background")
        self.generate_analogies = dspy.ChainOfThought("article -> analogies")
        self.detect_fake = dspy.ChainOfThought("article, constraints, background, analogies -> veracity")
        self.example_counter = 1

    def get_contexts(self, article):
        """生成三种上下文信息"""
        constraints = self._get_constraints(article)
        background = self._get_background(article)
        analogies = self._get_analogies(article)
        return f"Constraints:\n{constraints}\n\nBackground:\n{background}\n\nAnalogies:\n{analogies}"

    def _get_constraints(self, article):
        prompt = f"""基于以下文章生成验证约束：
        {article}
        请回答：
        1. 该文章是否包含可验证的事实声明？[是/否]
        2. 是否存在明显的情感操纵措辞？[是/否]
        3. 是否引用了可靠的消息来源？[是/否]"""
        return self._gpt_query(prompt)

    def _get_background(self, article):
        prompt = f"""为以下文章生成背景知识：
        {article}
        包含：
        - 关键实体的事实信息
        - 相关领域常识
        - 历史背景"""
        return self._gpt_query(prompt)

    def _get_analogies(self, article):
        prompt = f"""为以下文章生成类比案例：
        {article}
        要求：
        - 3个相似虚假新闻案例
        - 每个案例包含：标题、虚假点分析"""
        return self._gpt_query(prompt)

    def _gpt_query(self, prompt, max_tokens=300):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            time.sleep(1)  # API速率限制
            return response.choices[0].message['content'].strip()
        except Exception as e:
            print(f"API Error: {e}")
            return ""

    def forward(self, article):
        context = self.get_contexts(article)
        prediction = self.detect_fake(
            article=article,
            constraints=context.split("Constraints:")[1].split("Background:")[0].strip(),
            background=context.split("Background:")[1].split("Analogies:")[0].strip(),
            analogies=context.split("Analogies:")[1].strip()
        )
        return dspy.Prediction(context=context, veracity=prediction.veracity)

# 评估指标
def fnd_metric(example, pred, trace=None):
    pred_label = pred.veracity.lower()
    true_label = example.label.lower()
    
    # 处理模型输出的不同表述方式
    if "fake" in pred_label or "false" in pred_label or "misleading" in pred_label:
        pred_label = "fake"
    elif "real" in pred_label or "true" in pred_label or "accurate" in pred_label:
        pred_label = "real"
    return pred_label == true_label

# 模型训练与评估
def main():
    # 初始化配置
    openai.api_base = "https://api.xiaoai.plus/v1"
    openai.api_key = "sk-your-api-key"
    
    # 初始化模型
    fnd_model = CAPEFND()
    
    # 模型编译
    teleprompter = BootstrapFewShot(metric=fnd_metric, max_bootstrapped_demos=4)
    optimized_model = teleprompter.compile(fnd_model, trainset=train)
    
    # 模型评估
    evaluator = Evaluate(devset=dev, metric=fnd_metric, num_threads=4)
    evaluation_result = evaluator(optimized_model)
    
    # 保存结果
    save_results(optimized_model, evaluation_result)

def save_results(model, results):
    results_dir = "/content/drive/MyDrive/FND_Results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存模型
    model.save(os.path.join(results_dir, "optimized_model.json"))
    
    # 保存评估结果
    with open(os.path.join(results_dir, "evaluation.txt"), "w") as f:
        f.write(f"Accuracy: {results * 100:.2f}%")
    
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()

# 辅助功能
def calculate_accuracy(pred_file, true_labels):
    """计算准确率"""
    with open(pred_file) as f:
        preds = [json.loads(line)["veracity"] for line in f]
    
    correct = sum(1 for p, t in zip(preds, true_labels) if p.lower() == t.lower())
    return correct / len(true_labels)

def generate_report(input_file, output_file):
    """生成详细分析报告"""
    analysis = {"fake": 0, "real": 0}
    with open(input_file) as f:
        for line in f:
            data = json.loads(line)
            analysis[data["veracity"].lower()] += 1
    
    with open(output_file, "w") as f:
        f.write("Fake News Detection Report\n")
        f.write("==========================\n")
        f.write(f"Total Samples: {sum(analysis.values())}\n")
        f.write(f"Fake Detected: {analysis['fake']} ({analysis['fake']/sum(analysis.values()):.1%})\n")
        f.write(f"Real Detected: {analysis['real']} ({analysis['real']/sum(analysis.values()):.1%})\n")

# 示例使用
if False:
    test_article = "科学家发现喝绿茶可以治愈新冠肺炎"
    model = CAPEFND()
    prediction = model(test_article)
    print(f"Prediction: {prediction.veracity}")
    print(f"Context Analysis:\n{prediction.context}")