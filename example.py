import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from practicalportfolio.optimization import mean, plug_in_allocation, returns, allocation

stock_list = 'SPY AAPL MSFT AMZN GOOG GOOGL FB TSLA DIS NVDA PYPL INTC NFLX JNJ JPM'

data = yf.download(stock_list, start="2016-01-01", end="2021-01-30",
                   interval='1d')

stocks = returns(data['Close'].values.transpose())

allocation(stocks, 1)

mu = mean(stocks)
plug_in_c, rets = plug_in_allocation(stocks, 0.01)

print(mu)
print(plug_in_c)
print(rets)

plug_in_c, rets = allocation(stocks, 0.01)

print(plug_in_c)
print(rets)

qq = np.linspace(0.0105, 0.055, 100)

rets = []

for i in qq:
    allocs, ret = plug_in_allocation(stocks, i)
    # allocs, ret = allocation(stocks, i)
    rets.append(ret)

plt.plot(qq, rets)
plt.show()
# st = stock_list.split()
#
#
# for i, stock in enumerate(st):
#     vals = []
#     for j in qq:
#         c = plug_in_allocation(stocks, j)
#         vals.append(c[i])
#     plt.plot(qq, vals, label=stock)
#
# plt.show()
