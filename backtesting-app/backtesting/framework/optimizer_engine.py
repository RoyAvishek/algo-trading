import itertools
import pandas as pd

class OptimizerEngine:
    def __init__(self, symbols, data, strategy_class, backtester):
        self.symbols = symbols
        self.data = data
        self.strategy_class = strategy_class
        self.backtester = backtester

    def run_grid_search(self, param_grid):
        results = []
        for params in itertools.product(*param_grid.values()):
            config = dict(zip(param_grid.keys(), params))
            all_metrics = []

            for symbol in self.symbols:
                strategy = self.strategy_class(config)
                result = self.backtester.run(symbol, self.data, strategy)
                all_metrics.append(result['sharpe_ratio'])

            avg_sharpe = sum(all_metrics) / len(all_metrics)
            config['avg_sharpe'] = avg_sharpe
            results.append(config)

        return pd.DataFrame(results)
