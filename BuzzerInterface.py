import RPi.GPIO as GPIO   #import the GPIO library
import time               #import the time library
import random

class Buzzer(object):
    def __init__(self):
        GPIO.setwarnings(False) 
        GPIO.setmode(GPIO.BCM)
        self.buzzer_pin = 5 #set to GPIO pin 5
        GPIO.setup(self.buzzer_pin, GPIO.IN)
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        print("buzzer ready")

    def __del__(self):
        class_name = self.__class__.__name__
        print (class_name, "finished")

    def buzz(self,pitch, duration):   
        if(pitch==0):
            time.sleep(duration)
            return
        period = 1.0 / pitch     
        delay = period / 2     
        cycles = int(duration * pitch)   

        for i in range(cycles):  
            GPIO.output(self.buzzer_pin, True)   
            time.sleep(delay)    
            GPIO.output(self.buzzer_pin, False) 
            time.sleep(delay)    

    def play(self, tune):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        GPIO.setup(self.buzzer_pin, GPIO.IN)
    def register_action(self, name):
        if name == 'listen_on':
            pitches=[250, 250,250]        
            duration=[0.9,0.9,.9]
        elif name == 'stop_record':
            pitches=[250, 250,250]        
            duration=[0.4,0.4,.4]
        elif name == 'end':
            pitches= list(reversed([333,444,555,666,777]))
            duration=[.01,.02,.03,.04,.09]
            for p in range(0,len(pitches)):
                self.buzz(pitches[p], duration[p])  
                time.sleep(duration[p] *0.5)
        elif name == 'start':
            pitches = [333,444,555,666,777]
            duration=[.01,.02,.03,.04,.09]
            for p in range(0,len(pitches)):
                self.buzz(pitches[p], duration[p])  
                time.sleep(duration[p] *0.5)
        elif name == 'waiting':
            pitches = [ 435 for i in xrange(10)]
            pitchNormal = []
            minPitch = min(pitches)
            maxPitch = max(pitches)
            for element in range(0,len(pitches)):
                pitchNormal.append(1000*((pitches[element] - minPitch)/(maxPitch-minPitch))-0)

            duration = [random.randrange(0,40,1)/100.0 for i in xrange(10)]
            print pitchNormal
            for p in range(0,len(pitchNormal)):
                self.buzz(pitchNormal[p], duration[p])  
                time.sleep(duration[p] *0.5)
    def record(self):
        pitches=[500, 500,800]
        duration=[0.5,0.5,.9]
        for p in range(0,len(pitches)):
            self.buzz(pitches[p], duration[p])  
            time.sleep(duration[p] *0.5)

    def play(self,featureSet, playOptions = dict()):
        duration = [.2 for i in xrange(len(featureSet))] 
        if len(featureSet) == 1:
            duration = [1]
        for p in range(0,len(featureSet)):
            self.buzz(featureSet[p], duration[p])  
            time.sleep(duration[p] *0.5)

if __name__ == "__main__":
    buzzer = Buzzer()
    buzzer.bored()
