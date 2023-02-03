import numpy as np
import matplotlib.pyplot as plt
import wave

def slow_down(wav_file, factor):
    """
    wav_file: string, path to wav file
    factor: float, factor to slow down by
    """
    # read in wav file
    #wav = wave.open('/home/yakir/vioce_pitch_ws/hipitch_sound.wav', 'r')
    wav = wave.open('/home/yakir/vioce_pitch_ws/original.wav', 'r')

    
    # get params
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    # read frames
    frames = wav.readframes(nframes * nchannels)
    # close file
    wav.close()
    # convert to numpy array
    out = np.fromstring(frames, dtype=np.int16)
    # reshape
    out = out.reshape(-1, nchannels)
    # slow down
    out = out[::factor]
    # convert back to string
    out = out.astype(np.int16).tostring()
    # write to file
    wav_out = wave.open(wav_file[:-4] + '_slow.wav', 'w')
    wav_out.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
    wav_out.writeframes(out)
    wav_out.close()

if __name__ == '__main__':
    slow_down('/home/yakir/vioce_pitch_ws/test.wav', 2)
