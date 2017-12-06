from scipy.io import arff
from cStringIO import StringIO
import BuzzerInterface
import time
import numpy
import csv

# numpy.savetxt(ARFFoutput+"_st.csv", STermFeatures, delimiter=",", fmt='%f')    # store st features to CSV file

class ARFFInterface(object):

    def __init__(self):
        self.dtype={'names': ('o',
        'zeroCrossingRate','energyOfSoS','energy','spectralCentroid',
        'spectralSpread','spectralEntropy','spectralFlux','spectralRolloff',
        '9MFCC','10MFCC','11MFCC','12MFCC','13MFCC','14MFCC','15MFCC','16MFCC',
        '17MFCC','18MFCC','19MFCC','21MFCC','22chroma','23chroma','24chroma','25chroma',
        '26chroma','27chroma','28chroma','29chroma','30chroma','31chroma','32chroma','33chroma','34chromaDeviation','className','fileName'),
        'formats' : (
        'f4','f4','f4','f4','f4','f4','f4','f4',
        'f4','f4','f4','f4','f4','f4','f4','f4',
        'f4','f4','f4','f4','f4','f4','f4','f4',
        'f4','f4','f4','f4','f4','f4','f4','f4','f4',
        'f4','f8','f16',
        )
        }
        self.data = ""
        self.meta = ""
        self.csv = ""        
        self.mfcc = dict()
        self.mfcc['mfcc'] = dict()
        self.files = []


        #print self.data[0:10]
    def loadModels(self, arffFile,csvFile):
        try:
            openFile = open(arffFile,"r")
            f = StringIO(openFile.read())
            self.data, self.meta = arff.loadarff(f)
            print "loading models, ARFF: ", arffFile
        except:
            print 'arff failed to load'
        if True:
            openFile = open(csvFile, "r")
            fCSV = StringIO(openFile.read())
            self.csv = numpy.loadtxt(fCSV, delimiter=',')#,dtype=self.dtype) 
            self.files = self.csv[:,34]
            self.files = numpy.unique(self.files)
            print 'files'
            print self.files
            print "loading models, CSV: ", csvFile
            
        if True:
            print 'csv failed to load'

 



    def writeToModels(self, arffFile,csvFile, row):
        with open(csvFile, "a") as theCSV:
            numpy.savetxt(theCSV, numpy.array(row),  delimiter=',', fmt='%f')
        #with open(arffFile, "a") as theARFF:
        #    numpy.savetxt(theARFF, numpy.array(row),  delimiter=' ', fmt='%f')

    def writeARFFFileFromScratch(modelName,classNames,MidTermFeatures):
        f = open(modelName + ".arff", 'w')
        f.write('@RELATION ' + modelName + '\n')
        for fn in featureNames:
            f.write('@ATTRIBUTE ' + fn + ' NUMERIC\n')
        f.write('@ATTRIBUTE class {')
        for c in range(len(classNames)-1):
            f.write(classNames[c] + ',')
        f.write(classNames[-1] + '}\n\n')
        f.write('@DATA\n')
        for c, fe in enumerate(MidTermFeatures):
            for i in range(fe.shape[0]):
                f.write("{0:f},".format(fe[i]))
            f.write(classNames[0]+"\n")
        f.close()

    def addClass(self, className, rowNumber):
        self.data[rowNumber][len(data)] = className
        print( self.data[rowNumber][len(data)])
        print( self.data[rowNumber])

    def getMFCC(self):
        

        self.mfcc = dict()
        self.mfcc['mfcc'] = dict()
        
        for ele in range(0,len(self.data)):
            mfccRow = []
            mfccClass = self.data[ele][len(self.data[ele])-1]
            
            for col in range(9,21):
                mfccRow.append(self.data[ele][col])
            self.mfcc['mfcc'].setdefault(mfccClass,dict())
            self.mfcc['mfcc'][mfccClass].setdefault('data',dict())
            self.mfcc['mfcc'][mfccClass]['data'][ele] = dict()            
            self.mfcc['mfcc'][mfccClass]['data'][ele]['mfcc'] = mfccRow
            rowmax =  max(mfccRow)
            rowmin =  min(mfccRow)
            self.mfcc['mfcc'][mfccClass]['data'][ele]['max'] = rowmax
            self.mfcc['mfcc'][mfccClass]['data'][ele]['min'] = rowmin
            mfccNormal = []
            self.mfcc['mfcc'][mfccClass].setdefault('normalData',dict())
            for element in range(0,len(mfccRow)):
                mfccNormal.append(1000*((mfccRow[element] - rowmin)/(rowmax-rowmin))-0)
            self.mfcc['mfcc'][mfccClass]['normalData'] = mfccNormal


#ARFF = ARFFInterface()
#ARFF.getMFCC()
#print(ARFF.mfcc['mfcc']['beat']['normalData'])

"""

for ele in range(0,len(data)):
    chromaRow = []
    for col in range(23,33):
        chromaRow.append(data[ele][col])
    chroma.append(chromaRow)
for row in chroma:
    rowmax = max(row)
    rowmin = min (row)
    for ele in row:
        print(ele)
        norm = (1000*((ele - rowmin)/(rowmax-rowmin))-0)
        print norm
        buzz.buzz(norm,.15)
    time.sleep(1)

"""

content = """
@relation foo
@attribute width  numeric
@attribute height numeric
@attribute color  {red,green,blue,yellow,black}
@data
5.0,3.25,blue
4.5,3.75,green
3.0,4.00,red
"""




if __name__ == "__main__":

    mainCSV = "/home/pi/SOUND/ARFF/MLbrain.csv"
    mainARFF = "/home/pi/SOUND/ARFF/MLbrain.arff"
    arffInterface = ARFFInterface.arffInterface()
    arffInterface = arffInterface.loadModels(mainARFF, mainCSV)
    arffInterface.getMFCC()

#    row = 'apples'
#    arffInterface.writeToModels(mainARFF,mainCSV, row)
#    arffInterface.loadModels(mainARFF, mainCSV)

"""
0.06075126677751541, 0.04816611111164093, 3.174607038497925, 0.11591576784849167, 0.15509533882141113, 0.447290301322937, 0.0, 0.08054167032241821, -17.076231002807617, 1.3198065757751465, -0.7380301356315613, -0.3133908212184906, -0.14702309668064117, 0.19170987606048584, 0.15104541182518005, -0.1794986128807068, -0.3093113601207733, -0.4405377209186554, -0.38512930274009705, -0.09594880044460297, 0.06199207156896591, 0.0010971216252073646, 0.0024089247453957796, 0.001426439150236547, 0.004614533856511116, 0.0016191331669688225, 0.0013121123192831874, 0.0009952582186087966, 0.0006701184902340174, 0.0007140204543247819, 0.001281490083783865, 0.0011998607078567147, 0.0010541855590417981)
"""
