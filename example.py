import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from practicalportfolio.optimization import (efficency_curve,
                                             noshort_allocation,
                                             noshort_efficency_curve, returns,
                                             bootstraped_allocation)

stock_list = np.genfromtxt('stocks.csv', dtype=str)
stock_list = stock_list[:55]
stock_list = " ".join(stock_list)

# Retrive Some sample stocks from yfinance
data = yf.download(stock_list, start="2019-09-01", end="2021-02-28",
                   interval='1d', auto_adjust=True).dropna()
stocks = returns(data['Close'].values.transpose())

# Plot efficency curve
efficency_curve(stocks, plot_points=300)
# Efficency curve without negative weights
# noshort_efficency_curve(stocks, plot_points=150)

# Print optimal allocation for given risk
# print(bootstraped_allocation(stocks, 0.08))

plt.ylabel(r'Optimal Return $R$')
plt.xlabel(r'Risk $\sigma_0$')
plt.show()
