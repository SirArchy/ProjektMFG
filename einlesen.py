import numpy as np
import scipy.signal as sg
import pandas as pd
import re
import os
#from numpy import genfromtxt  ----> data = genfromtxt('daten.csv', delimiter = ',') um csv einzulesen



def refine_all_data(folder='Runde_2'):
    data = pd.read_csv("all_data.txt")#Liest all data als pandas datenbank ein

    arr = data.values                                       #pandas->numpy transformation für glättung und sowas
    arr = arr.transpose()                                   #transposition vereinfacht das glätten der daten, da danach arr[3] der 3. messreihe entspricht, und nicht wie vorher arr[:][3] die 3. Messreihe wäre
    columns = list(data.columns)                            #nun werden aus der DB die namen der daten ausgelesen
    
    savgol_arr = np.stack([sg.savgol_filter(arr[i],21,5) for i in range(arr.shape[0])])     #mit savitzky golais werden die daten geglättet


    #in dieser Schleife werden die Daten durch leistung und dauer genormt
    #noch nicht getestet!!!
    for i in range(len(columns)):
        if 'mW' in columns[i] and 's_' in columns[i]:
            savgol_arr[i] = savgol_arr[i] / get_mW_and_s(columns[i])
        
    konz = get_concentrations(columns)

    return savgol_arr, columns, konz #columns enthält die Namen der messreichen, so das savgol_arr[i] die daten mit dem Namen aus columns[i] enthält


# liest mW und s aus dem Namen aus
def get_mW_and_s(string):

    x = string.find('mW') -1
    y = 1
    mW = 0
    while string[x] in '0123456789':
        mW += int(string[x]) * y
        y *=10
        x -=1
    
    x = string.find('s_') -1
    y = 1
    s = 0
    while string[x] in '0123456789':
        s += int(string[x]) * y
        y *=10
        x -=1
    
    #print(string, ' mW ',mW,' s ', s)
    return mW * s

def get_concentrations(list):
    konz = []
    for i in list:
        if i == 'x':
            konz.append(-1)
        else:
            konz.append(float(re.findall(r'-?\d+\.?\d*', i)[0]))

    return konz


def get_urea_data(folder = '2022_02_25_RamanMessungen'):

    if folder not in  os.getcwd():   
        os.chdir(folder)

    #path = os.getcwd() + '\\2022_02_25_Messprotokoll.xlsx'
    excel = pd.read_excel(r'C:\\Users\\Paul\\Desktop\\UNI\\MFG\\KI_Spektroskopie_Quellcode\\Current\\Runde 2\\2022_02_25_RamanMessungen\\2022_02_25_Messprotokoll.xlsx')
    print(excel)
