from dspy.teleprompt import BootstrapFewShot

class SelfAdaptiveBootstrapOptimizer:
    def __init__(self):
        pass

    def optimize(self, detector, train_data):
        # 使用 BootstrapFewShot 进行优化
        teleprompter = BootstrapFewShot(metric=self._evaluate_veracity)
        optimized_detector = teleprompter.compile(detector, trainset=train_data)
        return optimized_detector

    def _evaluate_veracity(self, example, pred, trace=None):
        # 评估函数，用于优化
        return example["result"] == pred["result"]