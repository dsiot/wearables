import csv
import numpy as np
import matplotlib.pyplot as pp
from scipy.signal import find_peaks_cwt
import scipy

#data processing packages
from dstML.data_fn import *
from dstML.output_fn import *
from scipy.special import expit
import pickle


import os
import time 
from time import sleep
from datetime import datetime
# Simple example of reading the MCP3008 analog input channels and printing
# Import SPI library (for hardware SPI) and MCP3008 library.
import Adafruit_GPIO.SPI as SPI
import Adafruit_MCP3008


# Software SPI configuration:
CLK  = 18
MISO = 23
MOSI = 24
CS   = 25
mcp = Adafruit_MCP3008.MCP3008(clk=CLK, cs=CS, miso=MISO, mosi=MOSI)

# Hardware SPI configuration:
# SPI_PORT   = 0
# SPI_DEVICE = 0
# mcp = Adafruit_MCP3008.MCP3008(spi=SPI.SpiDev(SPI_PORT, SPI_DEVICE))

#Sampling period
PERIOD = 0.001
PERIOD_RR = 0.025

#function to calculate RMSSD
def rmssd(peak_durations):
    peak_duration_diff = peak_durations[1:] - peak_durations[:-1]
    peak_duration_diff = np.square(peak_duration_diff)
    peak_duration_diff = np.mean(peak_duration_diff)
    return peak_duration_diff
#calculate ppNN50 and ppNN20
def getPPNN50(peak_durations):
    peak_duration_diff = 0.025*abs(peak_durations[1:] - peak_durations[:-1])
    totallen = len(peak_duration_diff)
    peak_duration_diff = [data for data in peak_duration_diff if data > 0.05]
    n50 = len(peak_duration_diff)
    print str(n50)
    return n50*100/totallen
def getPPNN20(peak_durations):
    peak_duration_diff = 0.025*abs(peak_durations[1:] - peak_durations[:-1])
    totallen = len(peak_duration_diff)
    peak_duration_diff = [data for data in peak_duration_diff if data > 0.02]
    n50 = len(peak_duration_diff)
    print str(n50)
    return n50*100/totallen
def smooth(x,window_len=6,window='hamming'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

print('Reading MCP3008 values, press Ctrl-C to quit...')
# Print nice channel column headers.
print('| {0:>4} | {1:>4} | {2:>4} | {3:>4} | {4:>4} | {5:>4} | {6:>4} | {7:>4} |'.format(*range(8)))
print('-' * 57)

#keeping track of number of collecting samples
count = 0
countClassify = 0
countTotal = 0
#raw signal collected from sensor
signal = []
#array input to classifer
#inputData = np.array([[0,0,0,0,0,0]],dtype = 'float64')
inputData = np.zeros((1,6))
label = np.zeros((1,2))

#use 'raw.csv' to store time-domain waveform of PPG signal
#file = open('raw.csv','w')

#csv file storing features set for 1 minute segment which is used for classfication
file = open('FeaturesSetDtree.csv','w')

outputWriter = csv.writer(file)
if os.stat("FeaturesSetDtree.csv").st_size == 0:
        outputWriter.writerow(['Time','AVRR','SDRR','RMSSD','ppNN50','ppNN20','mental state'])
# Main program loop.
while (True):
    # Read all the ADC channel values in a list.
    values = [0]*8
    for i in range(8):
        # The read_adc function will get the value of the specified channel (0-7).
        values[i] = mcp.read_adc(i)
    # Print the ADC values.
    signal.append(values[0])
    print('| {0:>4} | {1:>4} | {2:>4} | {3:>4} | {4:>4} | {5:>4} | {6:>4} | {7:>4} |'.format(*values))
    # Pause for half a second.
    time.sleep(PERIOD)
    #extract dataset into csv file
    #now = datetime.now()
    #outputWriter.writerow([now,values[i]])
    #for every 2400 PPG time-domain-signal samples corresponding to 1 minute data segment with sampling period of 0.025s
    #calculate 6 time domain features of Heart Rate Variability 
    #1. AVRR = average time interval between two systolic peaks
    #2. AVHR = average Heart rate
    #3. SDRR = Standard deviation of time intervals set
    #4. RMSSD = root mean square
    #5. ppNN50 = number of difference between two consecutive peak intervals greater than 50ms
    #6. ppNN20 = number of difference between two consecutive peak intervals greater than 20ms
    if(count == 2400):
        signal_np = np.array(signal)
        filtered_np = smooth(signal_np)
        #detect systolic peaks
        indexes = find_peaks_cwt(filtered_np, np.arange(1, 25))
        indexes_np = np.array(indexes)
        print(indexes)
        print(len(indexes))
        #array storing peak interval
        peak_intervals = []
        peak_intervals[:] = indexes_np[1:] - indexes_np[:-1]
        peak_intervals = [data for data in peak_intervals if data > 20 and data < 45]
        peak_intervals = np.array(peak_intervals)
        print(peak_intervals)
        AVRR = np.mean(peak_intervals)
        SDRR = np.std(peak_intervals)
        AVHR = 60/(AVRR*PERIOD_RR)
        print("AVRR ="+str(AVRR))
        #calculate RMSSD
        RMSSD = rmssd(peak_intervals)
        ppNN50 = getPPNN50(peak_intervals)
        ppNN20 = getPPNN20(peak_intervals)
        count = 0
        print "AVRR: " + str(AVRR)+ " "+"SDRR: "+ str(SDRR) +" "+ "RMSSD: " +" "+ str(RMSSD) +" "+ "ppNN20: " +" "+ str(ppNN20) +" "+ "ppNN50:" + str(ppNN50)
        #Uncomment to plot waveform of PPG signal
##        pp.plot(filtered_np)
##        pp.plot(signal_np)
##        pp.show()

        #construct a data sample (instance) for classification
        inputData = np.append(inputData,[[AVHR, AVRR, SDRR, RMSSD, ppNN50, ppNN20]],0)
        #put timestamp into label array
        now = datetime.now()
        label = np.append(label,[[now,0]],0)
        
        print inputData
        

    if(countClassify == 4800):
       
        print(inputData)
        #only use last two elements of 2D inputData array
        data = normalize(inputData[1:,:])
        #data = normalize(inputData)
        print(data)
        
        #********Classify mental state**********
        # get output from DTree
        clf_dtree = pickle.load(open('clf_tree.sav','rb'))
        out_dtree = clf_dtree.predict_proba(data)
        out_dtree = (out_dtree[:,1]-out_dtree[:,0]+1)/2.0

    
        print 'relax = 1, stress = 0'
        print out_dtree
        #transform probability into proper label or class for mental state, then put into label array to store into csv file afterwards
        #relax = 1, stress = 0
        for i in range(2):
            if(out_dtree[i]>0.5):
                #output = relax 1
                label[i,1] = 1
            else:
                #output = stress 0
                label[i,1] = 0

        #Store data into csv file
        featuresTemp1 = np.concatenate((np.array([label[1,0]]),inputData[1,:],np.array([label[1,1]])),axis = 0)
        featuresTemp2 = np.concatenate((np.array([label[2,0]]),inputData[2,:],np.array([label[2,1]])),axis = 0)
        outputWriter.writerow(featuresTemp1)
        outputWriter.writerow(featuresTemp2)
        #reset countClassify for next classification
        countClassify = 0
        #delete the 1th and 2nd element of label array for next classfication
        label = np.delete(label,1,0)
        label = np.delete(label,1,0)
        #delete the 1th and 2nd element of inputData array for next classification
        print inputData
        inputData = np.delete(inputData,1,0)
        inputData = np.delete(inputData,1,0)
        #print clf_svm.decision_function(data)
    #after 24000 samples, stop the program
    #Change to different number for longer or shorter duration
    if(countTotal == 9600):
        break

        
    count = count+1
    countClassify = countClassify + 1
    countTotal = countTotal + 1

        
file.close()
