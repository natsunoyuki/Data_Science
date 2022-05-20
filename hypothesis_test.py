import numpy as np

class HypothesisTest(object):
    def __init__(self, data):
        self.data = data
        self.MakeModel()
        self.actual = self.TestStatistic(data)

    def PValue(self, iters = 1000):
        self.test_stats = np.zeros(iters)
        for i in range(iters):
            self.test_stats[i] = self.TestStatistic(self.RunModel())
        return np.sum(self.test_stats >= self.actual) / iters

    def TestStatistic(self, data):
        raise NotImplementedError

    def MakeModel(self):
        pass

    def RunModel(self):
        raise NotImplementedError

class DiceTest(HypothesisTest):
    def TestStatistic(self, data):
        n = np.sum(data)
        expected = np.ones(6) * n / 6
        test_stat = np.sum((data - expected) ** 2 / expected)
        return test_stat

    def RunModel(self):
        n = np.sum(self.data)
        values = np.array([1, 2, 3, 4, 5, 6])
        rolls = np.random.choice(values, n, replace = True)
        freqs = np.zeros(len(values))
        for i in range(len(values)):
            freqs[i] = np.sum(rolls == values[i])
        return freqs

class CoinTest(HypothesisTest):
    def TestStatistic(self, data):
        heads, tails = data
        test_stat = abs(heads - tails)
        return test_stat
    
    def RunModel(self):
        heads, tails = self.data
        sample = np.random.choice(2, heads + tails)
        [vals,counts] = np.unique(sample, return_counts = True)
        data = counts[0], counts[1]
        return data