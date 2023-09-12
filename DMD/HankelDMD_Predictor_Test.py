import HankelDMD_Predictor
import numpy as np
import matplotlib.pylab as plt
plt.rcParams['figure.dpi']=400

n = 0
k = 5
d = 400
n_ts = 500
hdmdp = HankelDMD_Predictor.HankelDMD_Predictor(d,k)
SV = np.load('../Mouse Data/svdTemporalComponents_corr.npy')[:30000,:500].T
SVn = SV[:5,:10000].T/np.linalg.norm(SV[:5,:n_ts])
hdmdp.fit(SVn[n*n_ts:(n+1)*n_ts,:5])
rec = hdmdp.predict(70,True)
fig,ax = plt.subplots(2,sharex=True,sharey=True)
ax[0].plot(SVn[:570])
ax[0].axvline(500)
ax[0].set_ylabel("True Data")
ax[1].plot(rec)
ax[1].axvline(500)
ax[1].set_ylabel("Reconstruction+Forecast")
rec = hdmdp.predict(70,False)
fig,ax = plt.subplots(2,sharex=True,sharey=True)
ax[0].plot(SVn[500:570])
ax[0].set_ylabel("True Data")
ax[1].plot(rec)
ax[1].set_ylabel("Forecast")
plt.show()
