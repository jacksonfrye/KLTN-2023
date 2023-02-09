from librosa import midi_to_hz, note_to_midi

HOP_LENGTH = 512
N_FFT = 1024
WIN_LENGTH = None
SAMPLE_RATE = 44100

N_MELS = 88
N_CLASS = 88

MIDI_START = note_to_midi('C1')
MIDI_END = MIDI_START + 88

F_MIN = midi_to_hz(MIDI_START)