from CommonUtilities import *

plot_music(debussy, redhot, duke)

# Amplitude Envelope
def aplitude_envelope(signal,frame_size, hop_size):
    return np.array([max(signal[i: i + frame_size]) for i in range(0, signal.size, hop_size)])

ae_debussy = aplitude_envelope(debussy, FRAME_SIZE, HOP_SIZE)
ae_redhot = aplitude_envelope(redhot, FRAME_SIZE, HOP_SIZE)
ae_duke = aplitude_envelope(duke, FRAME_SIZE, HOP_SIZE)

frames = range(0, ae_debussy.size)
t = librosa.frames_to_time(frames, hop_length=HOP_SIZE)   

plot_music_ae(t, debussy, ae_debussy, redhot, ae_redhot, duke, ae_duke)

# RMS Energy
def rms_energy(signal, frame_size, hop_size):
    return np.array([np.sqrt(np.sum(signal[i:i+frame_size] ** 2) / frame_size) for i in range(0, len(signal), hop_size)])

rms_debussy = rms_energy(debussy, FRAME_SIZE, HOP_SIZE)
rms_redhot = rms_energy(redhot, FRAME_SIZE, HOP_SIZE)
rms_duke = rms_energy(duke, FRAME_SIZE, HOP_SIZE)

frames = range(0, rms_debussy.size)
t = librosa.frames_to_time(frames, hop_length=HOP_SIZE)   

plot_music_ae(t, debussy, rms_debussy, redhot, rms_redhot, duke, rms_duke)

# Zero-Crossing
zrc_debussy = librosa.feature.zero_crossing_rate(debussy, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)[0]
zrc_redhot = librosa.feature.zero_crossing_rate(redhot, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)[0]
zrc_duke = librosa.feature.zero_crossing_rate(duke, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)[0]

frames = range(0, zrc_debussy.size)
t = librosa.frames_to_time(frames, hop_length=HOP_SIZE)   

plot_music_ae(t, debussy, zrc_debussy, redhot, zrc_redhot, duke, zrc_duke)