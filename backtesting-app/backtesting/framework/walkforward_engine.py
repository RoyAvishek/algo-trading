import pandas as pd

class WalkforwardEngine:
    def __init__(self, data, backtester, strategy_class):
        self.data = data
        self.backtester = backtester
        self.strategy_class = strategy_class

    def run(self, symbol, config, train_years=3, test_years=1):
        stock = self.data[self.data['TckrSymb'] == symbol].copy()
        stock = stock.sort_values('date').reset_index(drop=True)
        stock['year'] = stock['date'].dt.year

        min_year, max_year = stock['year'].min(), stock['year'].max()

        results = []
        for train_start in range(min_year, max_year - train_years - test_years + 1):
            train_end = train_start + train_years - 1
            test_start = train_end + 1
            test_end = test_start + test_years - 1

            test_data = stock[(stock['year'] >= test_start) & (stock['year'] <= test_end)]
            if len(test_data) < 100:
                continue

            strategy = self.strategy_class(config)
            result = self.backtester.run(symbol, test_data, strategy)
            results.append(result)

        return results
