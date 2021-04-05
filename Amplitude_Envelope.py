import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

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

def plot_music_ae(signal1, s1, signal2, s2, signal3, s3):
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


plot_music(debussy, redhot, duke)

FRAME_SIZE = 1024
HOP_SIZE = 512

def aplitude_envelope(signal,frame_size, hop_size):
    return np.array([max(signal[i: i + frame_size]) for i in range(0, signal.size, hop_size)])



ae_debussy = aplitude_envelope(debussy, FRAME_SIZE, HOP_SIZE)
ae_redhot = aplitude_envelope(redhot, FRAME_SIZE, HOP_SIZE)
ae_duke = aplitude_envelope(duke, FRAME_SIZE, HOP_SIZE)

frames = range(0, ae_debussy.size)
t = librosa.frames_to_time(frames, hop_length=HOP_SIZE)   

plot_music_ae(debussy, ae_debussy, redhot, ae_redhot, duke, ae_duke)

