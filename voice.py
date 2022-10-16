

import parselmouth
from parselmouth.praat import call
from IPython.display import Audio
import pygame
import ipywidgets
import glob
from playsound import playsound

import sounddevice as sd
from tkinter import *
import queue
import soundfile as sf
import threading
from tkinter import messagebox


# x1 = 0.01
# x2 =100
# x3 =900

class Recorder:
    def __init__(self):

        self.x1 = 0.01
        self.x2 = 75
        self.x3 = 600
        self.factor = 3
        self.q = queue.Queue()

    def setX1(self, x1):
        self.x1 = float(x1) / float(100)


    def setX2(self, x2):
        self.x2 = int(x2)

    def setX3(self, x3):
        self.x3 = int(x3)

    def setfactor(self, factor):
        self.factor = int(factor)

    def change_pitch(self, sound, factor):
        print('change_pitch ' + str(self.x1) + ', ' + str(self.x2) + ', ' + str(self.x3) + ', ' + str(self.factor))
        manipulation = call(sound, "To Manipulation",
                            self.x1, self.x2, self.x3)

        pitch_tier = call(manipulation, "Extract pitch tier")

        call(pitch_tier, "Multiply frequencies",
                sound.xmin, sound.xmax, self.factor)

        call([pitch_tier, manipulation], "Replace pitch tier")
        return call(manipulation, "Get resynthesis (overlap-add)")

    def interactive_change_pitch(self, audio_file):
        sound = parselmouth.Sound(audio_file)
        sound_changed_pitch = self.change_pitch(sound, self.factor)
        return Audio(data=sound_changed_pitch.values, rate=sound_changed_pitch.sampling_frequency)

    # Functions to play, stop and record audio in Python voice recorder
    # The recording is done as a thread to prevent it being the main process

    def threading_rec(self, command_index):
        if command_index == 1:
            # If recording is selected, then the thread is activated
            t1 = threading.Thread(target=self.record_audio)
            t1.start()
        elif command_index == 2:
            # To stop, set the flag to false
            print('stop the recording')
            global recording
            recording = False
            messagebox.showinfo(message="Recording finished")
        elif command_index == 3:
            # To play a recording, it must exist.
            if file_exists:
                # Read the recording if it exists and play it

                print('play the recording')
                sound = parselmouth.Sound("voice.wav")
                bla = self.interactive_change_pitch(sound)

                with open('change_pitchvoice.wav', 'wb') as f:
                    f.write(bla.data)

                data, fs = sf.read("change_pitch_voice.wav", dtype='float32')
                sd.play(data, fs)
                sd.wait()

            else:
                # Display and error if none is found
                messagebox.showerror(message="Record something to play")

    # Fit data into queue
    def callback(self, indata, frames, time, status):
        self.q.put(indata.copy())

    # Recording function
    def record_audio(self):
        print('record voice ...')
        # Declare global variables
        global recording
        # Set to True to record
        recording = True
        global file_exists
        # Create a file to save the audio
        messagebox.showinfo(message="Recording Audio. Speak into the mic")
        with sf.SoundFile("voice.wav", mode='w', samplerate=44100,
                            channels=2) as file:
            # Create an input stream to record audio without a preset time
            with sd.InputStream(samplerate=44100, channels=2, callback=self.callback):
                while recording == True:
                    # Set the variable to True to allow playing the audio later
                    file_exists = True
                    # write into file
                    file.write(self.q.get())


    # def show_values(value=None):
    #     print(value)

def main():

    recorder = Recorder()
    # Define the user interface for Voice Recorder using Python
    voice_rec = Tk()
    voice_rec.geometry("600x300")
    voice_rec.title("תוכנת עיוות קול")
    voice_rec.config(bg="#cccccc")
    voice_rec.resizable(False, False)
    # Create a queue to contain the audio data
    q = queue.Queue()
    # Declare variables and initialise them
    recording = False
    file_exists = False

    ##### x1 ##########################
    x1_label = Label(voice_rec, text="x1").grid(
        row=2, column=0, pady=4, padx=4)
    x1_label_s = Scale(voice_rec, from_=0, to=100,
                         orient=HORIZONTAL, length=300, command=recorder.setX1)
    x1_label_s.set(0.01)
    x1_label_s.grid(row=2, column=1, pady=4, padx=4)
    ##### x2 ##########################
    x2_label = Label(voice_rec, text="x2").grid(
        row=10, column=0, pady=4, padx=4)
    x2_label_s = Scale(voice_rec, from_=0, to=100,
                         orient=HORIZONTAL, length=300, command=recorder.setX2)
    x2_label_s.set(75)
    x2_label_s.grid(row=10, column=1, pady=4, padx=4)
    ##### x3 ##########################
    x3_label = Label(voice_rec, text="x3").grid(
        row=20, column=0, pady=4, padx=4)
    x3_label_s = Scale(voice_rec, from_=100, to=1000,
                         orient=HORIZONTAL, length=300, command=recorder.setX3)
    x3_label_s.set(600)
    x3_label_s.grid(row=20, column=1, pady=4, padx=4)
    ##### FACTOR ##########################
    factor_label = Label(voice_rec, text="factor").grid(
        row=30, column=0, pady=4, padx=4)
    factor_label_s = Scale(voice_rec, from_=1, to=5,
                         orient=HORIZONTAL, length=300, command=recorder.setfactor)
    factor_label_s.set(3)
    factor_label_s.grid(row=30, column=1, pady=4, padx=4)

    # Button to record audio
    record_btn = Button(voice_rec, text="RECORD",
                        command=lambda m=1: recorder.threading_rec(m))
    # Stop button
    stop_btn = Button(voice_rec, text="STOP RECORDING",
                      command=lambda m=2: recorder.threading_rec(m))
    # Play button
    play_btn = Button(voice_rec, text="PLAY RECORDING",
                      command=lambda m=3: recorder.threading_rec(m))
    # Position buttons
    record_btn.grid(row=1, column=1)
    stop_btn.grid(row=1, column=0)
    play_btn.grid(row=1, column=2)
    voice_rec.mainloop()


if __name__ == '__main__':
    main()

