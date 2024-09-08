import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import scipy.signal as sp

df=pd.read_csv('kpdata.csv')

#Converting year, month and day in datetime format

date=pd.to_datetime(df.YYY*10000+df.MM*100+df.DD, format='%Y%m%d')

#Generating the columns for kp and ap.
#We do this by averaging nearby eight columns.
kp=df[['Kp1','Kp2','Kp3','Kp4','Kp5','Kp6','Kp7','Kp8']].mean(axis=1)
ap=df[['ap1','ap2','ap3','ap4','ap5','ap6','ap7','ap8']].mean(axis=1)

#Modifying the dataset
df=df.drop(columns=['YYY','MM','DD'])
df=df.drop(columns=['Kp1','Kp2','Kp3','Kp4','Kp5','Kp6','Kp7','Kp8'])
df=df.drop(columns=['ap1','ap2','ap3','ap4','ap5','ap6','ap7','ap8'])

df.insert(0,'Date',date)
df.insert(5,'Kp',kp)
df.insert(6,'ap',ap)

dataset=df

#Next we have to generate the monthly averaged dataset
#Our date/time column in the dataframe is not in index format.
#Before generating monthly averaged dataset, we have to convert the date column in index format
#dataset_monthly=dataset.set_index('Date').resample('1M').mean()

dataset.set_index('Date',inplace=True)
dataset.index = pd.to_datetime(dataset.index)

dataset_monthly=dataset.resample('1M').mean()
monthly_dataset=dataset_monthly.reset_index()

#Our Monthly averaged dataset is now ready.
#We have to now generate plots.

time=monthly_dataset['Date']
sunspot_number=monthly_dataset['SN']
kpdata= monthly_dataset['Kp']
apdata= monthly_dataset['ap']
F10_RadioFlux= monthly_dataset['F10.7obs']

#We have to also generate 13 month smoothed dataset.
# Calculate moving average with a window size of 13
sunspot_smoothed = sunspot_number.rolling(window=13).mean()
kpdata_smoothed = kpdata.rolling(window=13).mean()
apdata_smoothed = apdata.rolling(window=13).mean()
F10_RadioFlux_smoothed = F10_RadioFlux.rolling(window=13).mean()


#We will now generate the plots

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
#ax1 = plt.subplot(311)
ax1.plot(time,sunspot_number,'-', color="0.6")
ax1.plot(time,sunspot_smoothed,'-k',linewidth=2.0)
ax1.legend(['Monthly','13 month Smoothed'])
#ax2 = plt.subplot(312)
ax2.plot(time,F10_RadioFlux,'-', color="0.6")
ax2.plot(time,F10_RadioFlux_smoothed,'-k',linewidth=2.0)
#ax3 = plt.subplot(313)
ax3.plot(time,kpdata,'-', color="0.6")
ax3.plot(time,kpdata_smoothed,'-k',linewidth=2.0)
ax3 = plt.gca()
ax3.set_xlim([time[216], time[1111]])
ax3.set_xlabel('Time')
ax1.set_ylabel('Sunspot Number')
ax2.set_ylabel('F10.7 Radio Flux')
ax3.set_ylabel('Kp Index')
plt.show()    


#Cross-Correlation Analysis

sunspot=sunspot_number[216:len(sunspot_number)-1]
f10=F10_RadioFlux[216:len(F10_RadioFlux)-1]
kp=kpdata[216:len(kpdata)-1]

#Correlation coefficient
corfs=np.corrcoef(sunspot,f10)
corfp=np.corrcoef(sunspot,kp)

#Lag and correlation
lags = sp.correlation_lags(len(sunspot), len(f10))
crossa=sp.correlate(sunspot,f10, 'full')
crossa /= np.max(crossa)
plt.plot(lags,crossa)
plt.axvline(x = 0, color = 'black', lw = 1)
plt.axhline(y = np.max(crossa), color = 'blue', lw = 1, linestyle='--', label = 'highest +/- correlation')
plt.axhline(y = np.min(crossa), color = 'blue', lw = 1, linestyle='--')
plt.xlabel('Time Lag', weight='bold', fontsize = 12)
plt.ylabel('Correlation Coefficients', weight='bold',fontsize = 12)
plt.show()

lagsb = sp.correlation_lags(len(sunspot), len(kp))
crossb=sp.correlate(sunspot,kp, 'full')
crossb /= np.max(crossb)
plt.plot(lagsb,crossb)
plt.axvline(x = 0, color = 'black', lw = 1)
plt.axhline(y = np.max(crossb), color = 'blue', lw = 1, linestyle='--', label = 'highest +/- correlation')
plt.axhline(y = np.min(crossb), color = 'blue', lw = 1, linestyle='--')
plt.xlabel('Time Lag', weight='bold', fontsize = 12)
plt.ylabel('Correlation Coefficients', weight='bold',fontsize = 12)

plt.show()


def ccf_values(series1, series2):
    p = series1
    q = series2
    p = (p - np.mean(p)) / (np.std(p) * len(p))
    q = (q - np.mean(q)) / (np.std(q))
    c = np.correlate(p, q, 'full')
    return c


crossf=ccf_values(sunspot,f10)
plt.plot(lags,crossf)
plt.axvline(x = 0, color = 'black', lw = 1)
plt.axhline(y = np.max(crossf), color = 'blue', lw = 1, linestyle='--', label = 'highest +/- correlation')
plt.axhline(y = np.min(crossf), color = 'blue', lw = 1, linestyle='--')
plt.xlabel('Time Lag', weight='bold', fontsize = 12)
plt.ylabel('Correlation Coefficients', weight='bold',fontsize = 12)
plt.show()


