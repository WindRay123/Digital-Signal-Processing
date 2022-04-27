import streamlit as st
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import scipy.io as spio
import pylops

# %% User Inputs
st.markdown("<h1 style='text-align: center; color: black;'>DFT Filtering of an ECG Signal : Design DFT Filtering for removing high frequency components having frequency above 30 Hz . Assume ECG signal with a sampling rate of 360 Hz</h1>",
            unsafe_allow_html=True)
st.sidebar.markdown("Submitted By : ")
st.sidebar.markdown(" **_Asif M.S._** ")
st.sidebar.markdown(" **_121901007_**")

option = st.selectbox(
     'Please Choose the ECG Signal :',
     ('ecg_rand_signal_fs_360_10seconds','ecg_with_powerline_fs_360_10seconds_p05 - Copy'))


#Importing the ECG signal from the MATLAB data file
if option=='ecg_rand_signal_fs_360_10seconds':
    ecg = spio.loadmat(option)['ecg_rang_signal'][0]
if option=='ecg_with_powerline_fs_360_10seconds_p05 - Copy':
    ecg = spio.loadmat(option)['ecg_01'][0]

#To plot the available ECG data
fig1 = plt.figure(figsize=(16,9))
x_time = [i/(360) for i in range (len(ecg))]
plt.plot(x_time,ecg)
plt.title("Original ECG Signal")
plt.xlabel("Time in seconds")
plt.ylabel("Amplitude")
st.pyplot(fig1)

N = len(ecg)

f_s = 360 #Sampling rate in Hz

x_freq = [i*(f_s/N) for i in range(int(N/2)+1)]
x_freq_1 = [i*(f_s/N) for i in range(int(N))]


f_s = 360 #Sampling frequency in Hz

ecg_fft = fft(ecg)
P2 = np.abs(ecg_fft)
phase = np.angle(ecg_fft)
fig8 = plt.figure()
plt.plot(x_freq_1,P2)

#To obtain single sided spectrum to know the frequency content
P1 = P2[:int(len(ecg)/2) + 1]
P1[1:] = 2*P1[1:]

fig2 = plt.figure()
plt.plot(x_freq,P1)
plt.title("Frequency Content of the Original ECG Signal ")
plt.xlabel("Frequency in Hz")
plt.ylabel("Amplitude")
st.pyplot(fig2)

f_c = st.sidebar.slider('Cut off frequency ', min_value=1, max_value=100, value=30,step=1) #Cut off frequency in Hz


x_freq = np.array(x_freq)
pos = np.where((x_freq)>=(f_c))
P1[pos] = 0

fig3 = plt.figure()
plt.plot(x_freq,P1)
plt.title("Frequency content of the DFT filtered signal ")
plt.xlabel("Frequency in Hz")
plt.ylabel("Amplitude")
st.pyplot(fig3)

#To recreate the double sided spectrum so that idft can be taken
sf = P1
l = 2*len(sf)-1
n = len(sf)
r1 = sf
r1[1:] = r1[1:]/2
Fop = pylops.Flip(n)
y = Fop*r1
fig5 = plt.figure()
plt.plot(x_freq,y)
res = []
for i in range(n-1):
    res.append(0)
for i in range(n-1):
    res.append(y[i])
x_freq = [i*(f_s/N) for i in range(int(N))]

for i in range(n-1):
    res[i] = r1[i]
x_freq = [i*(f_s/N) for i in range(int(N))]

for i in range(len(phase)):
    res[i] = complex(res[i]*np.cos(phase[i]),res[i]*np.sin(phase[i])) # After this loop , res stores the double sided spectrum for the filtered signal
reconstruct_signal = ifft(res) #Takes idft to get the filtered ECG signal back
t = [i*(1/360) for i in range(len(ecg))]
fig4 = plt.figure(figsize=(16,9))
plt.plot(t,reconstruct_signal)
plt.title("Reconstructed ECG Signal After Filtering ")
plt.xlabel("Time in seconds")
plt.ylabel("Amplitude")
st.pyplot(fig4)




