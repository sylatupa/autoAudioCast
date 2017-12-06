#import imutils
import numpy
import subprocess
#from multiprocessing import Pool
#import os
count = 1
#import itertools
import os

class sonifyData(object):
    def __init__(self):
        self.p = ''
        self.pid = ''
        pass
    def playData(self, featureSet, playOptions):
        featureSetProcess =  ['play','-n','synth']

        for note in featureSet:        
            featureSetProcess.append(playOptions['playType'])
            featureSetProcess.append(str(note))            
        print playOptions
        playThis = ''
        if len(featureSet)>1:
            effects = ['fade','h' , '0', '1','.1']
            effects = ['fade','h' , str(playOptions['fade']+.1), str(playOptions['fade']+.2),str(playOptions['fade']+.3)]
            delay = ['delay', '1.3', '1' , '.76', '.54', '.27']
            delay = ['delay', str(playOptions['delay'] *1.3), str(playOptions['delay']*.76) , str(playOptions['delay']*1.13), str(playOptions['delay']*1.24), str(playOptions['delay']*.99)]        
            repeat = ['repeat', str(playOptions['repeat'])]
            #repeat = ['repeat', playOptions['repeat']]
            playThis = featureSetProcess + effects + delay + repeat
        else:
            effects = ['fade','h' , str(playOptions['fade']), str(playOptions['fade'])]
            #effects = ['fade','h' , playOptions['fade'], playOptions['fade'],playOptions['fade']]
            delay = ['delay',  str(playOptions['delay'])]
    #        delay = ['delay', playOptions['delay'] +.1, playOptions['delay']+.2 , playOptions['delay']+.2, playOptions['delay']+.2, playOptions['delay']+.2]        
            repeat = ['repeat', '0']
            #repeat = ['repeat', playOptions['repeat']]
            playThis = featureSetProcess + effects + delay + repeat

        print playThis
                #process =  ['play','-n','synth','.6', 'sine', '333','sine',str(note), 'pl', str(note)]
                #process = ['play','-n','synth','333','sine',str(note), 'sine', str(note), 'fade', '0', '1']
        process_string =''
        for ele in playThis:
            process_string = process_string + ' ' + str(ele)
        print process_string
        try:
            subprocess.check_output(playThis)
        except: 
            print "*********SUB PROCESS FAIL*****************"

#        play -n synth pl G2 pl B2 pl D3 pl G3 pl D4 pl G4 delay 0 .05 .1 .15 .2 .25 remix - fade 0 4 .1 norm -1
          #subprocess.check_output(['play','-n','synth','2','pluck',str(note)])
    """
    def playData(self, featureSet, playOptions):
        print featureSet
        for note in featureSet:
            if note>=28:
                print 'current note ' , note
                subprocess.check_output(['play','-n','synth','.6','sine',str(note)])
          #subprocess.check_output(['play','-n','synth','2','pluck',str(note)])
    """   
    def playDatasss(self):
        featureSet = [333,532,514,255,114,235,135,156]
        featureSetProcess =  ['play','-n','synth']

        for note in featureSet:        
            featureSetProcess.append('sine')
            featureSetProcess.append(str(note))            

        effects = ['fade','h' , '0', '1','.1']
        delay = ['delay', '1.3', '1' , '.76', '.54', '.27']
        repeat = ['repeat', '10']

        playThis = featureSetProcess + effects + delay + repeat

                #process =  ['play','-n','synth','.6', 'sine', '333','sine',str(note), 'pl', str(note)]
                #process = ['play','-n','synth','333','sine',str(note), 'sine', str(note), 'fade', '0', '1']
        process_string =''
        for ele in playThis:
            process_string = process_string + ' ' + ele
        print process_string
        try:
            subprocess.check_output(playThis)
        except: 
            print "*********SUB PROCESS FAIL*****************"

#        play -n synth pl G2 pl B2 pl D3 pl G3 pl D4 pl G4 delay 0 .05 .1 .15 .2 .25 remix - fade 0 4 .1 norm -1
          #subprocess.check_output(['play','-n','synth','2','pluck',str(note)])
    def playAudioFile(self, AudioFile):
        featureSetProcess =  ['play','-n','-G',AudioFile, 'gain','-5']
        featureSetProcess =  ['play',AudioFile, 'gain','-5']

        #effects = ['fade','h' , '0', '1','.1']
        #delay = ['delay', '1.3', '1' , '.76', '.54', '.27']
        #repeat = ['repeat', '10']
        playThis = featureSetProcess #+ effects + delay + repeat
                #process =  ['play','-n','synth','.6', 'sine', '333','sine',str(note), 'pl', str(note)]
                #process = ['play','-n','synth','333','sine',str(note), 'sine', str(note), 'fade', '0', '1']
        #p = subprocess.Popen(playThis, stdout=subprocess.PIPE, shell=True)
        #subprocess.Popen(['ps', '-ef', '|' ,'grep','sox'])
        #ps = subprocess.Popen(('ps', '-ef'), stdout=subprocess.PIPE)
        #output = subprocess.check_output(('grep', 'python'), stdin=ps.stdout)
        #print ps
        #print output
        #ps.wait()
        #
        if self.pid == '' or os.path.lexists('/proc/%s' % self.pid) == False or True:
            print 'here'
            FNULL = open(os.devnull, 'w')
            self.p = subprocess.Popen(playThis, stdout=FNULL, stderr=FNULL, close_fds=True)
            self.pid = self.p.pid
            return self.p
        else:
            pass
            #self.p.kill()
            #FNULL = open(os.devnull, 'w')
            #self.p = subprocess.Popen(playThis, stdout=FNULL, close_fds=True)            
        """
        try:
            subprocess.check_output(playThis)
        except: 
            print "*********SUB PROCESS FAIL*****************"
        """
#        play -n synth pl G2 pl B2 pl D3 pl G3 pl D4 pl G4 delay 0 .05 .1 .15 .2 .25 remix - fade 0 4 .1 norm -1
          #subprocess.check_output(['play','-n','synth','2','pluck',str(note)])
    """def hello_world():
    URL = 'https://traffic.libsyn.com/democracynow/dn%d-%02d%02d-1.mp3'%(x.year,x.month,x.day)
    sonifyData.playPODCast(URL)
t = Timer(secs, hello_world)
t.start()
#https://gist.github.com/alexbw/1187132
#https://stackoverflow.com/questions/24072765/timer-cannot-restart-after-it-is-being-stopped-in-python
"""
    def playPODCast(self, URL):
        featureSetProcess =  ['play','-t','mp3',URL]
        playThis = featureSetProcess #+ effects + delay + repeat
        process_string =''
        for ele in playThis:
            process_string = process_string + ' ' + ele
        print 'subprocess string: ', process_string
        try:
            subprocess.check_output(playThis)
        except: 
            print "*********SUB PROCESS FAIL*****************"

    def tryit(self,fileName):
        try:
            subprocess.check_output(featureSetProcess)
        except: 
            print "*********SUB PROCESS FAIL*****************"

    def test(self):
        major_scale = 2
        scale = major_scale
        featureSet = [333,532,514,255,114,235,135,156]
        featureSet = [67]
        featureSetProcess =  []

        instrument = 'sine'
        repeat = 2
        scale = 2 
        for n in range(0,repeat):
            for note in featureSet:
                for position in range(1, scale+1):
                    featureSetProcess = []
                    featureSetProcess.append('play')
                    featureSetProcess.append('-n')
                    featureSetProcess.append('synth')
                    featureSetProcess.append('2')
                    featureSetProcess.append(instrument)
                    featureSetProcess.append(str(note * major_scale * position))
                    
                    featureSetProcess.append('fade')
                    featureSetProcess.append(str(.8))
                    featureSetProcess.append(str(0))
                    featureSetProcess.append(str(.8))                

                    featureSetProcess.append('tempo')
                    featureSetProcess.append('+3')
                    featureSetProcess.append('gain')                    
                    featureSetProcess.append('-2')
                    print n, ' ' , list(reversed(featureSet)).index(note) , ' ', position
                    #subprocess.check_output(featureSetProcess)
                    subprocess.Popen(featureSetProcess, stdout=subprocess.PIPE)
                    if n != 0  or list(reversed(featureSet)).index(note) != 0 or scale-position != 0 :
                        subprocess.check_output(featureSetProcess, stdin=ps.stdout)
                    
                        #ps.wait()
                    #ps.wait()
                        #featureSetProcess.append('|')
                        #featureSetProcess.append('&&')
        featureSetProcess.append('-t')
        featureSetProcess.append('alsa')        
                #featureSetProcess.append('&&')
                #featureSetProcess.append('fg')
#        effects = ['fade','h' , '0', '1','.1']
#        delay = ['delay', '1.3', '1' , '.76', '.54', '.27']
#        repeat = ['repeat', '10']

        playThis = featureSetProcess #+ effects + delay + repeat

                #process =  ['play','-n','synth','.6', 'sine', '333','sine',str(note), 'pl', str(note)]
                #process = ['play','-n','synth','333','sine',str(note), 'sine', str(note), 'fade', '0', '1']
        process_string =''
        for ele in playThis:
            process_string = process_string + ' ' + ele
        print process_string
        try:
            pass
            #subprocess.check_output(playThis)
        except: 
            print "*********SUB PROCESS FAIL*****************"

#        play -n synth pl G2 pl B2 pl D3 pl G3 pl D4 pl G4 delay 0 .05 .1 .15 .2 .25 remix - fade 0 4 .1 norm -1
          #subprocess.check_output(['play','-n','synth','2','pluck',str(note)])
        
if __name__ == "__main__":
    s = sonifyData()
    s.test()
#https://stackoverflow.com/questions/13332268/python-subprocess-command-with-pipe
#https://stackoverflow.com/questions/12999361/piping-sox-in-python-subprocess-alternative
#https://github.com/rabitt/pysox/blob/master/sox/core.py
