import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from practicalportfolio.optimization import (cov_matrix,
                                             efficency_curve, mean,
                                             plug_in_allocation, returns,
                                             noshort_allocation, noshort_efficency_curve)


stock_list = np.genfromtxt('stocks.csv', dtype=str)

stock_list = stock_list[:50]

stock_list = " ".join(stock_list)

data = yf.download(stock_list, start="2019-09-01", end="2021-02-28",
                   interval='1d', auto_adjust=True).dropna()

stocks = returns(data['Close'].values.transpose())

stock_stds = cov_matrix(stocks).diagonal()**0.5

print(min(stock_stds) + (max(stock_stds) - min(stock_stds))/2)


#efficency_curve(stocks)
noshort_efficency_curve(stocks)

plt.show()
