import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import pylops

def W(k,N): #Twiddle factor raised to k required for computing FFT ( Fast Fourier Transform )
    return complex(np.cos((-2*np.pi/N)*k), np.sin((-2*np.pi/N)*k))

def W_neg_pow(k,N): #Twiddle factor raised to -k required for computing IFFT ( Inverse Fast Fourier Transform )
    return complex(np.cos((2 * np.pi / N) * k ), np.sin((2 * np.pi / N) * k ))

def fft_implement(x): #This function computes the Fast Fourier Transform ( FFT ) of a sequence x. It follows radix 2 algorithm .Hence x should be padded with necessary zeroes to make its length a multiple of 2
    N = len(x)
    if N==1: #If the number of elements in the sequence is 1 , then the element itself is its fft
        return x
    # Recursive implementation
    f1_ = fft_implement(x[0:N+1:2])  #even sequence fft
    f2_ = fft_implement(x[1:N+1:2])  #odd sequence fft
    X = [0]*N
    for i in range(int(N/2)):
        X[i] = f1_[i] + W(i,N)*f2_[i] #Butterfly operation
        X[i + int(N/2)] = f1_[i] - W(i,N)*f2_[i]
    return X


def ifft_implement_1(X):
    N = len(X)
    if N == 1:  # If the number of elements in the sequence is 1 , then the element itself is its ifft
        return X
    f1_ = ifft_implement(X[0:N + 1:2]) #even sequence ifft
    f2_ = ifft_implement(X[1:N + 1:2]) #odd sequence ifft
    x = [0] * N
    for i in range(int(N / 2)):
        x[i] = f1_[i] + W_neg_pow(i, N) * f2_[i] #butterfly operation
        x[i + int(N / 2)] = f1_[i] - W_neg_pow(i, N) * f2_[i]
    return x #Here, it is not divided by the sequence length , hence the computation of ifft is not complete here

def ifft_implement(X):#Function to divide the output of the previous function by sequence length to make the computation of ifft complete
    N = len(X)
    x = [i/N for i in ifft_implement_1(X)]
    return x

def round_to_power_2(x):#THis function finds the next power of 2 . This is useful for padding the input sequence with zeroes until its length is a power of 2 .
    x = x - 1
    while x & x - 1: #The number x in binary form and (x-1) in binary form are multiplied in binary with each right bit turning zero at each iteration
        x = x & x - 1  #At last , x will be of the form 10,100,100,1000 (in binary) but it will be less than the original number
    return x << 1 #Then it is multiplied by 2 to get the next highest multiple of 2 (left shifting by 1 bit in binary is equivalent to multuplication by 2 )

# %% User Inputs
st.markdown("<h1 style='text-align: center; color: black;'>DFT Filtering of an ECG Signal : Design DFT Filtering for removing high frequency components having frequency above 30 Hz . Assume ECG signal with a sampling rate of 360 Hz</h1>",
            unsafe_allow_html=True)
st.sidebar.markdown("Submitted By : ")
st.sidebar.markdown(" **_Asif M.S._** ")
st.sidebar.markdown(" **_121901007_**")

option = st.selectbox(
     'Please Choose the ECG Signal ?',
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
N_org = N
ecg = ecg.tolist()
#Padding with zeroes if necessary to make the length of the ecg signal a power of 2
N1 = round_to_power_2(N)
while(N!=N1):#Keeps on appending zeroes to the ecg signal until its length is a multiple of  2
    ecg.append(0)
    N = N + 1
ecg = np.array(ecg)
N = len(ecg)

f_s = 360 #Sampling rate in Hz

x_freq = [i*(f_s/N) for i in range(int(N/2)+1)] #Computes the array of frequencies using the frequency resolution concept

f_s = 360 #Sampling frequency in Hz

ecg_fft = fft_implement(ecg)#Computes the fft of the ecg signal
P2 = np.abs(ecg_fft) #Takes the absolute value of the fft coefficients
phase = np.angle(ecg_fft)# Computes the phase of the array of fft co-efficients . It will be useful for reconstructing the filtered ECG signal back at a later stage .

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
pos = np.where((x_freq)>=(f_c)) #This condition checks for frequencies above the cut off and stores those indices inside pos variable
P1[pos] = 0  #All the DFT co-efficients in the indices inside pos variable are made zero.

fig3 = plt.figure()
plt.plot(x_freq,P1)
plt.title("Frequency content of the DFT filtered signal ")
plt.xlabel("Frequency in Hz")
plt.ylabel("Amplitude")
st.pyplot(fig3)

#To recreate the double sided spectrum so that idft can be taken
n = len(P1)
r1 = P1
r1[1:] = r1[1:]/2 #Earlier we multiplied by 2 to get the single sided spectrum . Here we are dividing by 2 as a step to recreating the double sided spectrum back
Fop = pylops.Flip(n) #Required for flipping the spectrum
y = Fop*r1 #Flipped modified one-sided spectrum
res = []#To store the recreated double sided spectrum
for i in range(n-1):
    res.append(0)
for i in range(n-1):
    res.append(y[i])

for i in range(n-1):
    res[i] = r1[i] #Now res stores the magnitude of the double sided spectrum

for i in range(len(phase)):
    res[i] = complex(res[i]*np.cos(phase[i]),res[i]*np.sin(phase[i])) # After this loop , res stores the double sided spectrum for the filtered signal
reconstruct_signal = ifft_implement(res) #Takes ifft to get the filtered ECG signal back
t = [i*(1/360) for i in range(len(ecg))]
fig4 = plt.figure(figsize=(16,9))
plt.plot(t[:N_org],reconstruct_signal[:N_org])
plt.title("Reconstructed ECG Signal After Filtering ")
plt.xlabel("Time in seconds")
plt.ylabel("Amplitude")
st.pyplot(fig4)



