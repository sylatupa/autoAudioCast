import audioInterface
import time
import threading
import sys
import random
import mlInterface
import os
import numpy
import sonifyData
sonifyData = sonifyData.sonifyData()
AudioInterface = audioInterface.audioInterface()
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
        #wavFileOutput = '/home/pi/SOUND/ML_samples/' + time.strftime("%Y%m%d%H%M%S") + ".wav"
        #AudioInterface.getAudioFile(wavFileOutput,OSCInterface.length)
        #(mt_features, st_features, Fs, x) = mlInterface.mtFeatureExtraction(wavFileOutput, mt_size, mt_step, st_size, st_step) 
        #st_features = numpy.transpose(st_features)   
        #file_names = AudioInterface.pydub_split_on_silence(wavFileOutput, 250, decibel_treshold, 100, 1)
        # AudioInterface.detect_leading_silence(wavFileOutput, decibel_treshold, 500)   #AudioInterface.remove_all_silence(wavFileOutput, decibel_treshold, 500)
        #os.remove(wavFileOutput)
    if file_loaded_and_no_new_files == False:
        file_loaded_and_no_new_files = True
        files = os.listdir("/home/pi/SOUND/ML_samples")
        files = [os.path.join("/home/pi/SOUND/ML_samples", f) for f in files] # add path to each file
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        #p = sonifyData.playAudioFile(files[file_play_count]) #AudioInterface.playAudioFile(f)
