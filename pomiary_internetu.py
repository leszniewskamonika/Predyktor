#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose


df = pd.read_csv('dataset/pomiary_internetu.csv',sep=';')

df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df.resample('W').mean()
df = df.fillna(df.bfill())


y = df['avg_mbps']
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(y,marker='.', linestyle='-', linewidth=0.5, label='Tygodniowe')
ax.plot(y.resample('M').mean(),marker='o', markersize=8, linestyle='-', label='Miesięczne średnie')
ax.set_ylabel('Średnie prędkości pobierania')
ax.legend();

"""#Rozkład danych"""

import statsmodels.api as sm
def seasonal_decompose(y):
  decomposition = sm.tsa.seasonal_decompose(y, model='additive',extrapolate_trend='freq')
  fig = decomposition.plot()
  fig.set_size_inches(14,7)
  plt.show()

seasonal_decompose(y)

"""# Test stacjonarności danych"""

def test_stationary(timeseries, title):
  rolmean = pd.Series(timeseries).rolling(window=12).mean()
  rolstd = pd.Series(timeseries).rolling(window=12).std()

  fig, ax = plt.subplots(figsize=(16,4))
  ax.plot(timeseries, label=title)
  ax.plot(rolmean, label='średnia')
  ax.plot(rolstd, label='odchylenie standardowe')
  ax.legend()

pd.options.display.float_format = '{:.8f}'.format
test_stationary(y, 'dane')

"""# Test Dickeya-Fullera"""

from statsmodels.tsa.stattools import adfuller

def DF_test(timeseries, dataDesc):
    print(' > Czy {} jest stacjonarne ?'.format(dataDesc))
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    print('Test statystyczny = {:.3f}'.format(dftest[0]))
    print('P-value = {:.3f}'.format(dftest[1]))
    print('Wartość krytyczna :')
    for k, v in dftest[4].items():
       print('\t{}: {} - Dane {} są stacjonarne w {}% przedziale ufności'.format(k, v, 'nie' if v<dftest[0] else '', 100-int(k[:-1])))

DF_test(y, 'dane')

"""# Zmiania danych na dane stacjonarne"""

#detrendowanie danych
y_detrend = (y - y.rolling(window=12).mean())/y.rolling(window=12).std()

test_stationary(y_detrend, 'dane de-trendowane')
DF_test(y_detrend, 'dane de-trendowane')

#różnicowanie danych
y_12lag = y - y.shift(12)

test_stationary(y_12lag, 'dane zróżnicowane')
DF_test(y_12lag, 'dane zróżnicowane')

"""# Połączenie dentrendowania oraz różnicowania"""

y_12lag_detrend = y_detrend - y_detrend.shift(12)

test_stationary(y_12lag_detrend, 'dane de-trendowane i różnicowane')

DF_test(y_12lag_detrend, 'dane de-trendowane i różnicowane')

"""# Tworzenie zestawu danych szkoleniwych i testowych"""

y_to_train = y[:'2021-12-01']
y_to_test = y['2022-11-20':]
predict_date = len(y) - len(y[: '2021-12-26'])

"""#Wybór modelu prognozowania szeregów czasowych"""

import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing 

def ses(y, y_to_train,y_to_test,smoothing_level,predict_date):
    y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
    
    fit1 = SimpleExpSmoothing(y_to_train).fit(smoothing_level=smoothing_level,optimized=False)
    fcast1 = fit1.forecast(predict_date).rename(r'$\alpha={}$'.format(smoothing_level))
   
    fcast1.plot(marker='o', color='blue', legend=True)
    fit1.fittedvalues.plot(marker='o',  color='blue')
    mse1 = ((fcast1 - y_to_test) ** 2).mean()
    print('Główny błąd średniokwadratowy naszych prognoz z poziomem wygładzania {} wynosi {}'.format(smoothing_level,round(np.sqrt(mse1), 2)))
    
   
    fit2 = SimpleExpSmoothing(y_to_train).fit()
    fcast2 = fit2.forecast(predict_date).rename(r'$\alpha=%s$'%fit2.model.params['smoothing_level'])
    
    fcast2.plot(marker='o', color='green', legend=True)
    fit2.fittedvalues.plot(marker='o', color='green')
    
    mse2 = ((fcast2 - y_to_test) ** 2).mean()
    print('Główny błąd średniokwadratowy naszych prognoz z automatyczną optymalizacją wynosi {}'.format(round(np.sqrt(mse2), 2)))
    
    plt.show()

ses(y, y_to_train,y_to_test,0.5,predict_date)

from statsmodels.tsa.api import Holt

def holt(y,y_to_train,y_to_test,smoothing_level,smoothing_slope, predict_date):
    y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
    
    fit1 = Holt(y_to_train).fit(smoothing_level, smoothing_slope, optimized=False)
    fcast1 = fit1.forecast(predict_date).rename("Trend liniowy Holta")
    mse1 = ((fcast1 - y_to_test) ** 2).mean()
    print('Główny błąd średniokwadratowy trendu liniowego Holta {}'.format(round(np.sqrt(mse1), 2)))

    fit2 = Holt(y_to_train, exponential=True).fit(smoothing_level, smoothing_slope, optimized=False)
    fcast2 = fit2.forecast(predict_date).rename("Trend wykładniczy")
    mse2 = ((fcast2 - y_to_test) ** 2).mean()
    print('Główny błąd średniokwadratowy trendu wykładniczego Holta {}'.format(round(np.sqrt(mse2), 2)))
    
    fit1.fittedvalues.plot(marker="o", color='blue')
    fcast1.plot(color='blue', marker="o", legend=True)
    fit2.fittedvalues.plot(marker="o", color='red')
    fcast2.plot(color='red', marker="o", legend=True)

    plt.show()

holt(y, y_to_train,y_to_train,0.6,0.2,predict_date)

"""# Model SARIMA"""

import itertools

def sarima_grid_search(y,seasonal_period):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2],seasonal_period) for x in list(itertools.product(p, d, q))]
    
    mini = float('+inf')
    
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                
                if results.aic < mini:
                    mini = results.aic
                    param_mini = param
                    param_seasonal_mini = param_seasonal
                    print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    print('Zestaw parametrów o minimalnym AIC to: SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini, mini))

sarima_grid_search(y,52)

def sarima(y,order,seasonal_order,seasonal_period,pred_date,y_to_test):
   
    mod = sm.tsa.statespace.SARIMAX(y,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

    results = mod.fit()
    print(results.summary().tables[1])
    
    results.plot_diagnostics(figsize=(16, 8))
    plt.show()
    
 
    pred = results.get_prediction(start=pd.to_datetime(pred_date,infer_datetime_format=True), dynamic=False)
    pred_ci = pred.conf_int()
    y_forecasted = pred.predicted_mean
    mse = ((y_forecasted - y_to_test) ** 2).mean()
    print('Średniokwadratowy błąd SARIMA z season_length={} i dynamic = False {}'.format(seasonal_period,round(np.sqrt(mse), 2)))

    ax = y.plot(label='zbiór danych')
    y_forecasted.plot(ax=ax, label='Prognoza o krok do przodu', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel(y.name)
    plt.legend()
    plt.show()


    pred_dynamic = results.get_prediction(start=pd.to_datetime(pred_date, infer_datetime_format=True), dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int()
    y_forecasted_dynamic = pred_dynamic.predicted_mean
    mse_dynamic = ((y_forecasted_dynamic - y_to_test) ** 2).mean()
    print('Pierwiastek średniokwadratowy błędu SARIMA z season_length={} i dynamic = True {}'.format(seasonal_period,round(np.sqrt(mse_dynamic), 2)))

    ax = y.plot(label='zbiór danych')
    y_forecasted_dynamic.plot(label='Prognoza dynamiczna', ax=ax,figsize=(14, 7))
    ax.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel(y.name)

    plt.legend()
    plt.show()
    
    return (results)

model = sarima(y,(0,1,1),(0,1,1,52),52,'2021-12-26',y_to_test)

"""# Tworzenie prognozy dla danych"""

def forecast(model,predict_steps,y):
    
    pred_uc = model.get_forecast(steps=predict_steps)

    #SARIMAXResults.conf_int, może zmienić wartość alfa, domyślna wartość alfa = 0,05 zwraca 95% przedział ufności.
    pred_ci = pred_uc.conf_int()

    ax = y.plot(label='zbiór danych', figsize=(14, 7))
   
    pred_uc.predicted_mean.plot(ax=ax, label='prognoza')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel(y.name)

    plt.legend()
    plt.show()
    
    
    pm = pred_uc.predicted_mean.reset_index()
    pm.columns = ['Date','Przewidywana średnia']
    pci = pred_ci.reset_index()
    pci.columns = ['Date','Dolna granica','Górna granica']
    final_table = pm.join(pci.set_index('Date'), on='Date')
    
    return (final_table)

final_table = forecast(model, 52, y)

final_table