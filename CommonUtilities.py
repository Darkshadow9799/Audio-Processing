import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

FRAME_SIZE = 1024
HOP_SIZE = 512

debussy_file='audio\debussy.wav'
redhot_file='audio\\redhot.wav'
duke_file='audio\duke.wav'

debussy, _ = librosa.load(debussy_file)
redhot, _ = librosa.load(redhot_file)
duke, _ = librosa.load(duke_file)

def plot_music(signal1, signal2, signal3):
    plt.figure(figsize=(15,17))

    plt.subplot(3,1,1)
    librosa.display.waveplot(signal1)
    plt.title("Signal1")
    plt.ylim((-1,1))

    plt.subplot(3,1,2)
    librosa.display.waveplot(signal2)
    plt.title("Signal2")
    plt.ylim((-1,1))

    plt.subplot(3,1,3)
    librosa.display.waveplot(signal3)
    plt.title("Signal3")
    plt.ylim((-1,1))

    plt.show()

def plot_music_ae(t, signal1, s1, signal2, s2, signal3, s3):
    plt.figure(figsize=(15,17))

    plt.subplot(3,1,1)
    librosa.display.waveplot(signal1, alpha=0.5)
    plt.plot(t, s1, color='r')
    #plt.title("Signal1")
    plt.ylim((-1,1))

    plt.subplot(3,1,2)
    librosa.display.waveplot(signal2, alpha=0.5)
    plt.plot(t, s2, color='r')
    #plt.title("Signal2")
    plt.ylim((-1,1))

    plt.subplot(3,1,3)
    librosa.display.waveplot(signal3, alpha=0.5)
    plt.plot(t, s3, color='r')
    #plt.title("Signal3")
    plt.ylim((-1,1))

    plt.show()
