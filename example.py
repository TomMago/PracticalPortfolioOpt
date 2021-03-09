import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from practicalportfolio.optimization import mean, plug_in_allocation, returns, allocation, cov_matrix

stock_list = 'SPY AAPL MSFT AMZN GOOG GOOGL FB TSLA DIS NVDA MMM ABT ABBV ACN ADBE ' \
              'PYPL INTC NFLX JNJ JPM AES T BLK BA BKNG BSX KO COP DPZ EBAY EA '\
              'EXPE GRMN GS HAS HPQ INTC JNJ JPM MA NKE ORCL PAYX PFE PPG PHM TXN '\
              'TWTR UAA VLO V WMT WDC XRX ZTS'

data = yf.download(stock_list, start="2019-09-01", end="2021-02-28",
                   interval='1h')

stocks = returns(data['Close'].values.transpose())

print(stocks.shape)

dig = cov_matrix(stocks).diagonal()**0.5


print(max(dig))
print(np.argmax(dig))

print(min(dig))
print(np.argmin(dig))

c, r = allocation(stocks, 0.012)
print(c[25])



import sys
sys.exit()

mu = mean(stocks)
plug_in_c, rets = plug_in_allocation(stocks, 0.01)

print(mu)
print(plug_in_c)
print(rets)

plug_in_c, rets = allocation(stocks, 0.01)

print(plug_in_c)
print(rets)

qq = np.linspace(0.0135, 0.095, 500)

rets = []
plug_ins = []

for i in qq:
    allocs, ret = allocation(stocks, i)
    allocs_plug, ret_plug = plug_in_allocation(stocks, i)
    # allocs, ret = allocation(stocks, i)
    rets.append(ret)
    plug_ins.append(ret_plug)

def fit(x, a, b, c):
    return a*x**2 + b*x + c

from scipy.optimize import curve_fit

par, cov = curve_fit(fit, qq, rets)

plt.plot(qq, rets)
plt.plot(qq, plug_ins)
plt.plot(qq, fit(qq, *par))
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
