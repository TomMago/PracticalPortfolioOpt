import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from practicalportfolio.optimization import mean, plug_in_allocation, returns, allocation, cov_matrix, efficency_curve

stock_list = np.genfromtxt('stocks.csv', dtype=str)
stock_list = stock_list[:75]
stock_list = " ".join(stock_list)
stock_list = stock_list.replace("WBC ", "")
stock_list = stock_list.replace("MUV2 ", "")
stock_list = stock_list.replace("RB. ", "")
stock_list = stock_list.replace("BAS ", "")
stock_list = stock_list.replace("RDSB ", "")
stock_list = stock_list.replace("BAM.A ", "")
stock_list = stock_list.replace("REL ", "")
stock_list = stock_list.replace("ZURN ", "")
stock_list = stock_list.replace("ADYEN ", "")
stock_list = stock_list.replace("NOVO ", "")
stock_list = stock_list.replace("PHIA ", "")
stock_list = stock_list.replace("VALE3 ", "")
stock_list = stock_list.replace("CRG ", "")
stock_list = stock_list.replace("UBSG ", "")
stock_list = stock_list.replace("INGA ", "")
stock_list = stock_list.replace("NESN ", "")
stock_list = stock_list.replace("DAI ", "")
stock_list = stock_list.replace("BP. ", "")
stock_list = stock_list.replace("BNP ", "")
stock_list = stock_list.replace("KER ", "")
stock_list = stock_list.replace("SIE ", "")
stock_list = stock_list.replace("XTSLA ", "")
stock_list = stock_list.replace("HSBA ", "")
stock_list = stock_list.replace("IBE ", "")
stock_list = stock_list.replace("BRKB ", "")
stock_list = stock_list.replace("LONN ", "")
stock_list = stock_list.replace("ENEL ", "")
stock_list = stock_list.replace("BATS ", "")
stock_list = stock_list.replace("VOW3 ", "")
stock_list = stock_list.replace("RDSA ", "")
stock_list = stock_list.replace("ANZ ", "")
stock_list = stock_list.replace("ABN ", "")
stock_list = stock_list.replace("FP ", "")
stock_list = stock_list.replace("IFX ", "")
stock_list = stock_list.replace("ULVR ", "")
stock_list = stock_list.replace("ABI ", "")
stock_list = stock_list.replace("NAB ", "")
stock_list = stock_list.replace("CRG ", "")
stock_list = stock_list.replace("ABBN ", "")


stock_list = stock_list.replace("\n", "")

#stock_list = 'SPY AAPL MSFT AMZN GOOG GOOGL FB TSLA DIS NVDA MMM ABT ABBV ACN ADBE ' \
#              'PYPL INTC NFLX JNJ JPM AES T BLK BA BKNG BSX KO COP DPZ EBAY EA '\
#              'EXPE GRMN GS HAS HPQ INTC JNJ JPM MA NKE ORCL PAYX PFE PPG PHM TXN '\
#              'TWTR UAA VLO V WMT WDC XRX ZTS'

data = yf.download(stock_list, start="2019-09-01", end="2021-02-28",
                   interval='1h').dropna()

stocks = returns(data['Close'].values.transpose())

del data

print(stocks.shape)

print(np.isnan(np.sum(stocks)))

efficency_curve(stocks)

efficency_curve(stocks[:20])


#
#print(stocks.shape)
#

#
#
#
#c, r = allocation(stocks, 0.012)
#print(c[25])
#
#

#import sys
#sys.exit()
#
#mu = mean(stocks)
#plug_in_c, rets = plug_in_allocation(stocks, 0.01)
#
#print(mu)
#print(plug_in_c)
#print(rets)
#
#plug_in_c, rets = allocation(stocks, 0.01)
#
#print(plug_in_c)
 # #print(rets)
 #
 # qq = np.linspace(0.0135, 0.095, 500)
 #
 # rets = []
 # plug_ins = []
 #
 # for i in qq:
 #     allocs, ret = allocation(stocks, i)
 #     allocs_plug, ret_plug = plug_in_allocation(stocks, i)
 #     # allocs, ret = allocation(stocks, i)
 #     rets.append(ret)
 #     plug_ins.append(ret_plug)
 #
 # def fit(x, a, b, c):
 #     return a*x**2 + b*x + c
 #
 # from scipy.optimize import curve_fit
 #
 # par, cov = curve_fit(fit, qq, rets)
 #
 # plt.plot(qq, rets)
 # plt.plot(qq, plug_ins)
 # plt.plot(qq, fit(qq, *par))
 # plt.show()
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
