from OSC import OSCServer,OSCClient, OSCMessage
import OSC
from time import sleep
import types
import threading
class OSCInterface():
    def __init__(self):
        self.length = 4
        self.playOptions = dict()
        self.playOptions['play'] = False
        self.playOptions['record'] = False
        self.playOptions['skip'] = False
        self.playOptions['keep'] = False        
        self.playOptions['delete'] = False
        self.playOptions['threshold'] = 25
        self.playOptions['stop'] = False
        self.client = OSC.OSCClient()
        self.msg = OSC.OSCMessage()
    def startServer(self, addr, port):
        self.addr = (addr, port) #home wifi
        #super(OSCInterface, self).__init__()   
        self.threadedOSC = OSC.ThreadingOSCServer(self.addr)
        self.threadedOSC.addDefaultHandlers()
        self.threadedOSC.addMsgHandler( "/1/play",self.play)
        self.threadedOSC.addMsgHandler( "/1/record",self.record)
        self.threadedOSC.addMsgHandler( "/1/skip",self.skip)
        self.threadedOSC.addMsgHandler( "/1/keep",self.keep)        
        self.threadedOSC.addMsgHandler( "/1/delete",self.delete)
        self.threadedOSC.addMsgHandler( "/1/threshold",self.threshold)
        self.threadedOSC.addMsgHandler( "/1/stop",self.stop)
        self.st = threading.Thread(target=self.threadedOSC.serve_forever)
        self.st.start()
        print("Starting openOSC on addr and port: " + str(addr) + ":" +str(port))

    

    def play(self,path, tags, args, source): 
        print 'play'
        if self.playOptions['play']: 
            self.playOptions['play'] = False 
        else: 
            self.playOptions['play'] = True
    def record(self,path, tags, args, source):
        print 'record'
        if self.playOptions['record']:
            self.playOptions['record'] = False
        else:
            self.playOptions['record'] = True
    def skip(self,path, tags, args, source):
        print 'skip'
        if self.playOptions['skip']:
            self.playOptions['skip'] = False
        else:
            self.playOptions['skip'] = True
    def keep(self,path, tags, args, source):
        print 'keep'
        if self.playOptions['keep']:
            self.playOptions['keep'] = False
        else:
            self.playOptions['keep'] = True
    def delete(self,path, tags, args, source):
        print 'delete'
        if self.playOptions['delete']:
            self.playOptions['delete'] = False
        else:
            self.playOptions['delete'] = True            
    def threshold(self,path, tags, args, source):
        self.playOptions['threshold'] = args[0]
        print args[0]    
        
    def stop(self,path, tags, args, source):
        print('stop')
        print(path, ' ', tags , ' ' , args , ' ', source)
        self.playOptions['stop'] = False
        self.playOptions['play'] = False
        self.playOptions['record'] = False
        self.threadedOSC.close()
        self.st._Thread__stop()
        self.st._Thread__delete()
        self.st._stop_event.set()
        OSCInterface.listen = True
    def handle_error(self,request,client_address):
        print 'OSC HANDLE ERROR'
        pass
    def handle_request(self):
        #needs to be run in a loop
        self.server.handle_request()

if __name__ == "__main__":
    #self.addr = ("192.168.43.66", 8000) #cell phone
    #self.addr = ("192.168.43.135", 8000)
    addr = "192.168.0.7"  #home wifi
    port = 8000
    OSCInterface = OSCInterface(addr,port)
