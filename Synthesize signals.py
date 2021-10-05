# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 10:19:59 2021

@author: KlasRydhmer
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.signal import welch

# This could preferrably be changed
main_path = os.getcwd() + '/'
path_to_data = main_path + 'Data/'

if os.path.isdir(main_path) is False:
    os.mkdir(main_path)
    
if os.path.isdir(path_to_data) is False:
    os.mkdir(path_to_data)

def gen_ts(length=0.1, f0=150, body_size=1000, wing_size=100,
           n_harmonics=5, f0_noise=5, noise_amp=10, fs=20800):
    """Generates a synthesized time signal to mimic an optically recorded
        insect observation.

    Inputs
    ----------
    length : float
        Length of signal in seconds

    f0 : float
        Fundamental wingbeat frequency
    
    body_size : float
        maximum amplitude of body contribution

    wing_size : tensor N, bottleneck size
        Amplitude of wing contributions
    
    n_harmonics : int (> 1)
        Number of harmonics in wing signal
        

    f0_noise : float
        Variation from f0 in Hz for each overtone
        
    noise_amp : float
        Amplitude of measurement noise

    fs : int
        Sample frequency in Hz
        
    Outputs:
    ----------
    time_signal : Numpy array
        Generated time signal
    """

    # Number of samples
    N = int(fs*length)
    # Sample points
    t = np.linspace(0, length, N)
    
    # Create body signal
    body_signal = body_size*np.sin(2*np.pi*10*t - np.pi/3)
    body_signal[body_signal < 0] = 0

    # Create wing signal
    wing_signal = []
    # Wing signal frequencies (fundamental tone, overtone 1, overtone 2 etc.)
    ws = 2*np.pi*np.linspace(f0, f0*n_harmonics, n_harmonics)
    for i in range(n_harmonics):
        # Add some noise to frequencies
        w = ws[i] + np.random.randn(N)*f0_noise
        
        # Add som noise to the amplitude
        A = (wing_size + np.random.randn()*wing_size/3)

        # Simple model to account for decaying intensity
        A = A*np.exp(-i/3)
        
        # Add small pertubations to the phase of the signal
        phi = np.random.random_sample()*np.pi*2 + np.random.randn(N)*0.1
        
        # Generate the wing signal
        sub_signal = A*np.sin(w*t + phi)
        wing_signal.append(sub_signal)

    # Scale wing signal with body signal
    wing_signal = np.array(wing_signal).sum(axis=0)*(body_signal/np.max(body_signal))

    # Assemble signal
    time_signal = body_signal + wing_signal
    
    # Add noise
    noise = np.random.randn(N)
    time_signal += noise*noise_amp
    return time_signal


def generate_labelled_signals(N, f0, body_size, wing_size,
                              f0_var=None, body_var=None, wing_var=None):
    """Generates a collection of similar signals, corresponding to the data
    collected from a single species.

    Inputs
    ----------
    N : int
        Number of generated signals

    f0 : float
        Fundamental wingbeat frequency
    
    body_size : float
        Amplitude of body contribution

    wing_size : tensor N, bottleneck size
        Amplitude of wing contributions
    
    f0_var : int (> 1)
        Variance of f0 within the species, typically ca 30%
        
    body_var : float
        Variance of body size within the species
        
    wing_var : float
        Variance of wing size within the species
        
    Outputs:
    ----------
    time_signals : list 
        Generated time signal, length N
    """

    if f0_var == None:
        f0_var = f0*0.1

    if body_var == None:
        body_var = body_size*0.1

    if wing_var == None:
        wing_var = wing_size*0.1
    
    time_signals = []
    for i in range(N):
        length = np.abs(np.random.randn()*0.2 + 0.1)
        time_signal = gen_ts(f0=f0 + np.random.randn()*f0_var,
                             body_size=body_size+np.random.randn()*body_var,
                             wing_size=wing_size+np.random.randn()*wing_var)
        time_signals.append(time_signal)

    return time_signals


def prepare_data(time_sigs, filename):
    """Calculates the power spectra of synthesized or recorded signals and 
        prepares them for the VAE.
    
        Inputs
        ----------
        time_sigs : list
            list of time signals
    
        filename : string
            filename
        
    """
    global path_to_data

    # Calculate power spectra
    print('Formatting data')
    fft_list = []
    for i in range(len(time_sigs)):
        try:
            ts = np.expand_dims(time_sigs[i], axis=0)

            # Calculate power spectra
            ftx, pws = welch(ts, fs=20800, return_onesided=True,
                             nperseg=1000, noverlap=800, nfft=2000)

            # Use all data below 2kHz
            ind = np.where(ftx < 2000)[0]
            event_data = pws[:, ind]

            fft_list.append(event_data)

        except Exception as E:
            print(path_to_data, i, str(E))

        # Print update
        if i % 500 == 0:
            frac = (i/len(time_sigs))*100
            print('Calculating power spectra...', str(np.round(frac, 3)))

    with open(path_to_data + filename, 'wb') as handle:
        pickle.dump(fft_list, handle)

    return fft_list


# %% Generate proxies for labelled data
plt.close('all')
species = [80, 120, 180, 130, 100, 5,
           500, 90, 130, 160, 200, 25]

body_sizes = [500, 100, 3000, 500, 1000, 500,
              800, 300, 800, 850, 400, 10000]

wing_sizes = body_sizes - body_sizes*np.random.randn(len(species))*0.5
colors = cm.rainbow(np.linspace(0, 1, len(species)))

for i in range(len(species)):
    time_signals = generate_labelled_signals(1000, species[i],
                                             body_sizes[i], wing_sizes[i])
    
    fft_list = prepare_data(time_signals, str(species[i]) + 'Hz')


    # Plot average spectra for species
    data = np.log10(np.array(fft_list))

    # Calculate average spectra
    avg_spectra = np.median(data.mean(axis=1), axis=0)
    ftx = np.linspace(0, 2000, 193)

    # Inter quantile length
    y1 = np.quantile(data[:, 0, :], 0.25, axis=0)
    y2 = np.quantile(data[:, 0, :], 0.75, axis=0)

    fig, ax = plt.subplots(1)
    ax.fill_between(ftx, y1, y2, color=colors[i], alpha=0.15)
    ax.plot(np.linspace(0, 2000, 193), avg_spectra, color=colors[i])
    ax.set_title(str(species[i]) + 'Hz')
    
# %% Generate proxies for unlabelled field data
time_signals = []
for i in range(5000):
    f0 = np.random.random_integers(50, 1000)
    body_size = np.random.randint(50, 10000)
    wing_size = np.random.randint(50, 5000)
    time_signals.append(gen_ts(f0=f0, body_size=body_size,
                               wing_size=wing_size))
    
    if i % 1000 == 0:
        print('Generating unlabelled data', i)
    
prepare_data(time_signals, 'Unlabelled')
