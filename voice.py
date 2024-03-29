
import sounddevice as sd
from tkinter import *
import queue
import soundfile as sf
import threading
from tkinter import messagebox

from PIL import Image, ImageTk
import matplotlib.pyplot as plt

import time
from pydub import AudioSegment

import scipy
from scipy.io import wavfile
from scipy.fftpack import fft

import numpy as np #pip3 install numpy==1.19.5
import subprocess

import os

import librosa_logic

class Recorder:
    def __init__(self):
     
        self.octaves = 0.0
        self.q = queue.Queue()    
        
        self.project_path = '/home/yoni/voice_final/voice_pitch/'

        # Define the user interface for Voice Recorder using Python
        self.voice_rec = Tk()

        self.voice_rec.protocol("WM_DELETE_WINDOW", self.close)

        self.voice_rec.geometry("1000x500")
        self.voice_rec.title("  YONI - תוכנת עיוות קול")
        self.voice_rec.config(bg="white")
        self.voice_rec.resizable(False, False)
        # Create a queue to contain the audio data
        q = queue.Queue()
        # Declare variables and initialise them
        recording = False
        file_exists = False  

        ##### OCTAVE ##########################
        octave_label = Label(self.voice_rec, text="octave", bg="white").grid(
            row=40, column=0, pady=4, padx=4)
        octave_label_s = Scale(self.voice_rec, from_=-5, to=15,
                            orient=HORIZONTAL, length=300, bg="white", command=self.setOctave)
        octave_label_s.set(0)
        octave_label_s.grid(row=40, column=1, pady=4, padx=4)

        # Button to record audio
        im_re = Image.open(self.project_path+'record.jpg')
        ph_re= ImageTk.PhotoImage(im_re)
        record_btn = Button(self.voice_rec, text="RECORD", image=ph_re, 
                            command=lambda m=1: self.threading_rec(m))
        # Stop button
        im_st = Image.open(self.project_path+'stop.jpg')
        ph_st= ImageTk.PhotoImage(im_st)
        stop_btn = Button(self.voice_rec, text="STOP RECORDING", image=ph_st, 
                        command=lambda m=2: self.threading_rec(m))
        # Play button
        im_pl = Image.open(self.project_path+'play.jpg')
        ph_pl= ImageTk.PhotoImage(im_pl)    
        play_btn = Button(self.voice_rec, text="PLAY RECORDING", image=ph_pl, 
                        command=lambda m=3: self.threading_rec(m))
        
        self.rec_img = ImageTk.PhotoImage(Image.open(self.project_path+"record_b.png"))
        self.idle_img = ImageTk.PhotoImage(Image.open(self.project_path+"idle_b.png"))
       
        # Position buttons
        record_btn.grid(row=1, column=1)
        stop_btn.grid(row=1, column=0)
        play_btn.grid(row=1, column=2)

        self.label_img = Label(self.voice_rec, image = self.idle_img)
        self.label_img.grid(row=50, column=1)

        self.voice_rec.mainloop()

    def close(self):
        print('close')

        os.remove(self.project_path+'original.wav')
        os.remove(self.project_path+'hipitch_sound.wav')
        os.remove(self.project_path+'slow_hipitch_sound.wav')
        os.remove(self.project_path+'plt.png')
        
        

        self.voice_rec.destroy()
        self.voice_rec.quit()
        exit(1)

    def setOctave(self, octaves):
        self.octaves = float(octaves) / 10.0   

   
    def change_octave_and_play(self):
        voice_file = self.project_path+'original.wav'
        sound = AudioSegment.from_file(voice_file, format="wav")

        # shift the pitch up by half an octave (speed will increase proportionally)
        new_sample_rate = int(sound.frame_rate * (2.0 ** self.octaves))

        # keep the same samples but tell the computer they ought to be played at the 
        # new, higher sample rate. This file sounds like a chipmunk but has a weird sample rate.
        hipitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})

        # now we just convert it to a common sample rate (44.1k - standard audio CD) to 
        # make sure it works in regular audio players. Other than potentially losing audio quality (if
        # you set it too low - 44.1k is plenty) this should now noticeable change how the audio sounds.
        hipitch_sound = hipitch_sound.set_frame_rate(int(44100))        
        hipitch_sound.export(self.project_path+"hipitch_sound.wav", format="wav")
        original, fs = librosa_logic.yakr_load(self.project_path+"hipitch_sound.wav")
        
        if (self.octaves > 0):
            slow_recording = librosa_logic.time_stretch(original, 0.5)  
            scipy.io.wavfile.write(self.project_path+"slow_hipitch_sound.wav", fs, slow_recording) 
            subprocess.run(["aplay", self.project_path+"slow_hipitch_sound.wav"])
        elif (self.octaves < 0):
            slow_recording = librosa_logic.time_stretch(original, 1.3)  
            scipy.io.wavfile.write(self.project_path+"slow_hipitch_sound.wav", fs, slow_recording) 
            subprocess.run(["aplay", self.project_path+"slow_hipitch_sound.wav"])    
        else:
            subprocess.run(["aplay", self.project_path+"hipitch_sound.wav"])
            
    # Functions to play, stop and record audio in Python voice recorder
    # The recording is done as a thread to prevent it being the main process

    def threading_rec(self, command_index):
        if command_index == 1:

            self.label_img.configure(image = self.rec_img)           

            # If recording is selected, then the thread is activated
            t1 = threading.Thread(target=self.record_audio)
            t1.start()
        elif command_index == 2:
            # To stop, set the flag to false
            print('stop the recording')

            self.label_img.configure(image = self.idle_img)

            global recording
            recording = False
            messagebox.showinfo(message="Recording finished")
        elif command_index == 3:
           
            if file_exists:
                # Read the recording if it exists and play it

                print('play the recording')

                sample_rate, original_voice = wavfile.read(self.project_path+'original.wav')

                self.fft_plot(original_voice, sample_rate)                 
                self.label_img.configure(image = self.plt_img)

                t2 = threading.Thread(target=self.change_octave_and_play)
                t2.start()

            else:
                # Display and error if none is found
                messagebox.showerror(message="Record something to play")

    # Fit data into queue
    def callback(self, indata, frames, time, status):
        self.q.put(indata.copy())

    # Recording function
    def record_audio(self):
        # Declare global variables
        global recording
        # Set to True to record
        recording = True
        global file_exists
        # Create a file to save the audio
        messagebox.showinfo(message="Recording Audio. Speak into the mic")
        with sf.SoundFile(self.project_path+"original.wav", mode='w', samplerate=44100,
                            channels=2) as file:
            # Create an input stream to record audio without a preset time
            with sd.InputStream(samplerate=44100, channels=2, callback=self.callback):
                while recording == True:
                    # Set the variable to True to allow playing the audio later
                    file_exists = True
                    # write into file
                    file.write(self.q.get())

    def fft_plot(self, audio, sample_rate):
        N = len(audio)    # Number of samples
        y_freq = fft(audio)
        domain = len(y_freq) // 2
        x_freq = np.linspace(0, sample_rate//2, N//2)
        
        plt.plot(x_freq, y_freq[:domain])
       

        plt.savefig(self.project_path+'plt.png')
        time.sleep(1)
        self.plt_img = ImageTk.PhotoImage(Image.open(self.project_path+"plt.png"))

  
def main():
   

    recorder = Recorder() 

if __name__ == '__main__':
    main()
   

   

   

   
