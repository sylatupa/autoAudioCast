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
#OSCInterface.startServer("169.254.10.1",8000)     
#OSCInterface.startServer("192.168.0.8",8000)  
"""
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
"""                
Buzzer = BuzzerInterface.Buzzer()
sonifyData = sonifyData.sonifyData()
AudioInterface = audioInterface.audioInterface()
OSCInterface.length = 70# 70 # normal is 8 or 10 chant 25 
mt_size = 1.0
mt_step = 1.0
st_size = .249
st_step = .249
buzz=True
OSCInterface.length = .48# 70 # normal is 8 or 10 chant 25 
st_size = .24
st_step = .12
buzz=True
#if buzz: Buzzer.register_action('start')
wavFileOutput = './MLsamples/' + time.strftime("%Y%m%d%H%M%S") + ".wav"
while True:
    if True: 
        #keep_the_sound = False
        AudioInterface.getAudioFile(wavFileOutput,OSCInterface.length)
        (mt_features, st_features, Fs, x) = mlInterface.mtFeatureExtraction(wavFileOutput, mt_size, mt_step, st_size, st_step) 
        st_features = numpy.around(st_features, decimals=3)
        st_features = numpy.transpose(st_features)
        print 'osc'
        print 'size',st_features.shape
        for feat in st_features:
            OSCInterface.msg.clear()
            OSCInterface.msg.setAddress("/test")
            OSCInterface.msg.append(feat)
            OSCInterface.client.sendto(OSCInterface.msg, ('192.168.0.9', 8100)) # note that the second arg is a tupple and not two arguments
        os.remove(wavFileOutput)
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

