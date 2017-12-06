import OSCInterface
import BuzzerInterface
import audioInterface
import time
import threading
import sys
import random
import mlInterface
import os
import numpy
import sonifyData
OSCInterface = OSCInterface.OSCInterface()
Buzzer = BuzzerInterface.Buzzer()
try:
    OSCInterface.startServer("192.168.43.66",8000)     #cell phone
    Buzzer.register_action('start')
except:
    try:
        OSCInterface.startServer("192.168.0.7",8000)                  #home wifi
        Buzzer.register_action('start')
    except:
        try:
            OSCInterface.startServer("192.168.43.135", 8000)
            Buzzer.register_action('start')
        except:
            try:
                OSCInterface.startServer("192.168.0.4",8000)                  #home wifi
            except:
                print 'osc not connected'
                #Buzzer.register_action('end')
                pass
Buzzer = BuzzerInterface.Buzzer()
sonifyData = sonifyData.sonifyData()
AudioInterface = audioInterface.audioInterface()
OSCInterface.length = 70 # normal is 8 or 10 chant 25
mt_size = 1.0
mt_step = 1.0
st_size = .25
st_step = .25
st_size = .5
st_step = .5

buzz=True
keep_the_sound = False
decibel_treshold = -26#-16 for coffee shop with music
KEEPSpectral_Centroid = .25
LOWSpectral_Centroid = KEEPSpectral_Centroid - .05
HIGHSpectral_Centroid = KEEPSpectral_Centroid + .05
KEEPzcrs = .1 
LOWzcrs = KEEPzcrs - .1
HIGHzcrs = KEEPzcrs + .1
file_play_count = 0
def should_I_keep(st_features):
        keep_the_sound = False
        AVGzcrs = numpy.mean(st_features[:,0])
        AVGenergy = numpy.mean(st_features[:,1])
        AVGentropy = numpy.mean(st_features[:,2])
        AVGSpectral_Centroid = numpy.mean(st_features[:,3])    
        AVGSpectral_Spread = numpy.mean(st_features[:,4])
        space = ''
        for d in range(int(AVGzcrs*200)): space += '.'
        print "ZCR:",space,AVGzcrs
        space = ''
        for d in range(int(AVGenergy*200)): space += '.'
        print "Energy:",space,AVGenergy
        space = ''
        for d in range(int(AVGSpectral_Centroid*200)): space += '.'
        print "Spectral_Centroid:",space,AVGSpectral_Centroid
        space = ''
        for d in range(int(AVGSpectral_Spread*200)): space += '.'
        print "Spectral_Spread:",space,AVGSpectral_Spread
        
        # CONTROL LOGIC PUT ON THE INCOMING SOUND STATISTICS

        if AVGenergy > .65:     
            print 'Energy MAX   -- CONSTANT LOUD NOISE 2 Energy' 
            keep_the_sound = True        
        if AVGentropy < 1.7:    
            print 'Entropy MIN  -- FINGER SNAP also with the moving of the object' 
            keep_the_sound = True
        if AVGzcrs > .65:       
            print 'keeping **** ZCR MAX WHISTLE' 
            keep_the_sound = True
        elif LOWzcrs <= AVGzcrs and AVGzcrs <= HIGHzcrs: #not keeing these sounds now    
            print 'keeping **** ZCR',AVGzcrs   
            keep_the_sound = True
        if  LOWSpectral_Centroid <= AVGSpectral_Centroid and HIGHSpectral_Centroid >= AVGSpectral_Centroid:
            print 'centroid range keeping'
            keep_the_sound = True
        if  AVGSpectral_Centroid > .33:     print 'ambient'
        elif AVGSpectral_Centroid > .23:    print 'birds music or a commercial'  
        elif AVGSpectral_Centroid > .205:   print 'high music or a commericial ' 
        elif AVGSpectral_Centroid > .19:    print 'high '  
        elif AVGSpectral_Centroid >= .14:   print 'Male Rock Voice music or atmospheric sound'  #else:  print '< 14 SC     *^* Below .14 SC *^* Low Spectral Noise noise' 
        if AVGSpectral_Spread > .3:         
            print 'SpectralSpread MAX  .16 violin, .20 birds .22 cars, ' 
            keep_the_sound = True
        elif AVGSpectral_Spread < .1:       print 'SpectralSpread MIN'

        return keep_the_sound

file_loaded_and_no_new_files = False
while True:
    decibel_treshold = -OSCInterface.playOptions['threshold']
    if OSCInterface.playOptions['record'] == True or True: 
        if buzz: Buzzer.register_action('start')
        #keep_the_sound = False
        wavFileOutput = '/home/pi/SOUND/ML_samples/' + time.strftime("%Y%m%d%H%M%S") + ".wav"
        AudioInterface.getAudioFile(wavFileOutput,OSCInterface.length)
        (mt_features, st_features, Fs, x) = mlInterface.mtFeatureExtraction(wavFileOutput, mt_size, mt_step, st_size, st_step) 
        st_features = numpy.transpose(st_features)   
        # AVERAGING
        for d in range(5): print '' #clear screen
#        keep_the_sound = should_I_keep(st_features)
        for d in range(5): print '' #clear screen

        if keep_the_sound or True:
            file_names = AudioInterface.pydub_split_on_silence(wavFileOutput, 250, decibel_treshold, 100, 1)
            for files in file_names:
                #(mt_features, st_features, Fs, x) = mlInterface.mtFeatureExtraction(files, mt_size, mt_step, st_size, st_step) 
                #st_features = numpy.transpose(st_features)  
                #keep_the_sound = should_I_keep(st_features)
                print files
                time.sleep(2.5)
                print '************going to keep*************'
                sonifyData.playAudioFile(files)
                print '************going to keep*************'
                """
                if keep_the_sound == False:
                    os.remove(files)
                else:
                    file_loaded_and_no_new_files = False
                """
            # AudioInterface.detect_leading_silence(wavFileOutput, decibel_treshold, 500)   #AudioInterface.remove_all_silence(wavFileOutput, decibel_treshold, 500)
        #for f in file_names:
        #os.remove(wavFileOutput)
            if buzz: Buzzer.register_action('start')
            time.sleep(15)
    if file_loaded_and_no_new_files == False:
        file_loaded_and_no_new_files = True
        files = os.listdir("/home/pi/SOUND/ML_samples")
        files = [os.path.join("/home/pi/SOUND/ML_samples", f) for f in files] # add path to each file
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    if OSCInterface.playOptions['play'] == True:
        p = sonifyData.playAudioFile(files[file_play_count]) #AudioInterface.playAudioFile(f)
        OSCInterface.playOptions['play'] = False
    if OSCInterface.playOptions['delete'] == True:
        try:
            p.kill()
        except:
            pass
        os.remove(files[file_play_count])
        file_play_count += 1
        OSCInterface.playOptions['delete'] = False
    if OSCInterface.playOptions['keep'] == True:
        try:
            p.kill()
        except:
            pass        
        path, filen  = os.path.split(files[file_play_count])
        print path, '    ' , filen
        os.rename(files[file_play_count], "/home/pi/SOUND/KEEP_samples/"+filen)
        OSCInterface.playOptions['keep'] = False
    if OSCInterface.playOptions['skip'] == True:
        try:
            p.kill()
        except:
            pass        
        file_play_count += 1
        OSCInterface.playOptions['skip'] = False
    if OSCInterface.playOptions['stop'] == True:
        OSCInterface.playOptions['stop'] = False
"""
except Exception as inst:
    Buzzer.register_action('end')
    OSCInterface.stop(True, True,True,True)
    e = sys.exc_info()[0]
    print( "<p>Error: %s</p>" % e )
    print type(inst)     # the exception instance
    print inst.args      # arguments stored in .args
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
"""

