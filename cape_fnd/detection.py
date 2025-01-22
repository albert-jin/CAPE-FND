from cape_fnd.constraints import get_veracity_constraints
from cape_fnd.background import get_veracity_background
from cape_fnd.analogies import get_veracity_analogies

class CAPEFNDDetector:
    def __init__(self):
        pass

    def detect(self, articles):
        results = []
        for article in articles:
            constraints = get_veracity_constraints(article)
            background = get_veracity_background(article)
            analogies = get_veracity_analogies(article)

            # 综合判断
            result = self._evaluate_veracity(constraints, background, analogies)
            results.append({
                "article": article,
                "constraints": constraints,
                "background": background,
                "analogies": analogies,
                "result": result
            })
        return results

    def _evaluate_veracity(self, constraints, background, analogies):
        # 这里可以根据约束、背景和类比进行综合判断
        # 例如：如果约束和背景都倾向于假新闻，则判定为假
        if "fake" in constraints.lower() and "no evidence" in background.lower():
            return "Fake"
        else:
            return "Real"