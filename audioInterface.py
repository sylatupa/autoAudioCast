#https://benchodroff.com/2017/02/18/using-a-raspberry-pi-with-a-microphone-to-hear-an-audio-alarm-using-fft-in-python/
#!/usr/bin/env python
import pyaudio
from numpy import zeros,linspace,short,fromstring,hstack,transpose,log
from time import sleep
import wave
from ctypes import *
#from pydub import AudioSegment
import pydub
class audioInterface():
    def __init__(self):
        print "starting audioInterface.py, pygame, mixer"
#Volume Sensitivity, 0.05: Extremely Sensitive, may give false alarms#             0.1: Probably Ideal volume#             1: Poorly sensitive, will only go off for relatively loud
        self.CHANNELS = 1
        self.frequencyoutput=True
        self.freqNow = 1.0
        self.freqPast = 1.0
#Set up audio sampler - 
        self.NUM_SAMPLES = 2048
        self.SAMPLING_RATE = 48000 #make sure this matches the sampling rate of your mic!
        """
        self._stream = self.pa.open(format=pyaudio.paInt16,
                  channels=1, rate=self.SAMPLING_RATE,
                  input=True,
                  frames_per_buffer=self.NUM_SAMPLES)
        """
        print("finished audio init")

    '''sound is a pydub.AudioSegment\silence_threshold in dB\chunk_size in ms\iterate over chunks until you find the first one with sound'''
    def detect_leading_silence(self,filename, silence_threshold=-24.0, chunk_size=200):
        sound = pydub.AudioSegment.from_file(filename, format="wav")
        duration = len(sound)    
        trimmed_sound = pydub.effects.strip_silence(sound, 1000, silence_threshold, chunk_size)
        print 'remaining duration: ', len(trimmed_sound) - (duration - len(trimmed_sound))
        trimmed_sound.export(filename, format="wav")
        return trimmed_sound

    def pydub_split_on_silence(self,filename, min_silence_len=200, silence_thresh=-24, keep_silence=100, seek_step=1):
        sound = pydub.AudioSegment.from_file(filename, format="wav")    
        trimmed_sound = pydub.effects.split_on_silence(sound, min_silence_len, silence_thresh,keep_silence)
        #(audio_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=100,seek_step=1):
        count = 0
        file_names = []
        filename = filename[0:-4]
        for snd in trimmed_sound:
            length = len(snd)
            print length
            snd = snd.fade(from_gain=-30.0, start=0, duration=length/6)
            snd = snd.fade(to_gain=-40.0, end=0, duration=length/5)
            snd.export(filename+'_'+str(count)+'.wav', format="wav")
            file_names.append(filename+'_'+str(count)+'.wav')
            count = count + 1
        return file_names 
    """
from pydub import AudioSegment
sound1 = AudioSegment.from_file("sound1.wav")
sound2 = AudioSegment.from_file("sound2.wav")

played_togther = sound1.overlay(sound2)

sound2_starts_after_delay = sound1.overlay(sound2, position=5000)

volume_of_sound1_reduced_during_overlay = sound1.overlay(sound2, gain_during_overlay=-8)

sound2_repeats_until_sound1_ends = sound1.overlay(sound2, loop=true)

sound2_plays_twice = sound1.overlay(sound2, times=2)

# assume sound1 is 30 sec long and sound2 is 5 sec long:
sound2_plays_a_lot = sound1.overlay(sound2, times=10000)
len(sound1) == len(sound2_plays_a_lot)
"""

    def remove_all_silence(self,filename, silence_threshold=-15.0, chunk_size=500):
        print 'silence removal below: ', silence_threshold
        sound = pydub.AudioSegment.from_file(filename, format="wav")
        duration = len(sound)    
        trim_ms = 0 # ms
        trimmed_sound = pydub.AudioSegment.empty()
        chunk_prev = 0
        for chunk in range(1,len(sound),chunk_size):
            print  chunk, '    ' ,sound[chunk_prev:chunk].dBFS , ' ' , sound[chunk_prev:chunk].dBFS < silence_threshold
            if sound[chunk_prev:chunk].dBFS > silence_threshold: 
                trimmed_sound += sound[chunk_prev:chunk]
            chunk_prev = chunk
        trimmed_sound.export(filename, format="wav")
        return trimmed_sound
        
    def getSampleStream(self):
        self.pa = pyaudio.PyAudio()
        self._stream = self.pa.open(format=pyaudio.paInt16, channels=self.CHANNELS, rate=self.SAMPLING_RATE, input=True, frames_per_buffer=self.NUM_SAMPLES)
        while self._stream.get_read_available()< self.NUM_SAMPLES: sleep(0.01) 
        audio_data  = fromstring(self._stream.read(self._stream.get_read_available()), dtype=short)[-self.NUM_SAMPLES:]
        self._stream.stop_stream()
        self._stream.close()
        self.pa.terminate() 
        return audio_data  # Each data point is a signed 16 bit number, so we can normalize by dividing 32*1024
    def getAudioFile(self, name,length):
        WAVE_OUTPUT_FILENAME = name 
        self.pa = pyaudio.PyAudio()
        stream = self.pa.open(format=pyaudio.paInt16, channels=self.CHANNELS, rate=self.SAMPLING_RATE, input=True, frames_per_buffer=self.NUM_SAMPLES)
        frames = []
        for i in range(0, int(self.SAMPLING_RATE / self.NUM_SAMPLES * length)):   #self.RECORD_SECONDS#
            data = stream.read(self.NUM_SAMPLES)
            frames.append(data)
        # stop Recording
        stream.stop_stream()
        stream.close()
        self.pa.terminate()
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(self.CHANNELS)
        waveFile.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(self.SAMPLING_RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
    def playAudioFile(self,fileName):
        p = pyaudio.PyAudio()
        chunk = 1024
        wf = wave.open(fileName, 'rb')
        stream = p.open(format =
                        p.get_format_from_width(wf.getsampwidth()),
                        channels = wf.getnchannels(),
                        rate = wf.getframerate(),
                        output = True)
        data = wf.readframes(chunk)

# play stream (looping from beginning of file to the end)
        while data != '':
            stream.write(data)
            data = wf.readframes(chunk)

        stream.close()    
        p.terminate()


