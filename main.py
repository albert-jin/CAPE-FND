import os
import sys
from cape_fnd.detection import CAPEFNDDetector
from cape_fnd.optimization import SelfAdaptiveBootstrapOptimizer
from utils.logger import setup_logger
from utils.file_utils import load_data, save_results

# 设置日志
logger = setup_logger('main')

def main():
    # 加载数据
    train_data = load_data('data/train.json')
    test_data = load_data('data/test.json')

    # 初始化检测器
    detector = CAPEFNDDetector()

    # 进行检测
    results = detector.detect(test_data)

    # 保存结果
    save_results(results, 'results/output.json')

    # 进行优化
    optimizer = SelfAdaptiveBootstrapOptimizer()
    optimized_detector = optimizer.optimize(detector, train_data)

    # 使用优化后的检测器进行检测
    optimized_results = optimized_detector.detect(test_data)

    # 保存优化后的结果
    save_results(optimized_results, 'results/optimized_output.json')

if __name__ == "__main__":
    main()