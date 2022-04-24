import numpy as np
import scipy.signal as sg
import pandas as pd
import re
import os
import csv


def refine_all_data(folder='Runde_2', save_data = True):
    '''
        Liest alle Messdaten aus angegebenen Order ein, filtert diese Daten und speichert sie in einem Array
        
        input:  folder (int) 
                save_data (boolean) 
                
        output: savgol_arr (array)
                columns (array)
                konz (array)
    '''
    data = pd.read_csv("all_data.txt")#Liest all data als pandas datenbank ein

    arr = data.values                                       #pandas->numpy
    arr = arr.transpose()                                   #transposition vereinfacht das glätten der daten, da danach arr[3] der 3. messreihe entspricht, und nicht wie vorher arr[:][3] die 3. Messreihe wäre
    columns = list(data.columns)                            #nun werden aus der DB die namen der daten ausgelesen
    
    savgol_arr = np.stack([sg.savgol_filter(arr[i],21,5) for i in range(arr.shape[0])])     #mit savitzky golais werden die daten geglättet


    #in dieser Schleife werden die Daten durch leistung und dauer genormt (24mW 10s --> /240)
    for i in range(len(columns)):
        if 'mW' in columns[i] and 's_' in columns[i]:
            savgol_arr[i] = savgol_arr[i] / get_mW_and_s(columns[i])
        
    konz = get_concentrations(columns)                      #die konzentrationnen werden ausgelesen, die Reihenfolge ist konstant  (z.B. savgol_arr[5],columns[5],konz[5] beschreiben selbe stelle)

    #Daten werden abgespeichert
    if save_data:
        np.savetxt("All_data.csv", savgol_arr, delimiter=",")
        np.savetxt('All_data_columns.txt', columns, delimiter=",", fmt='%s')
        np.savetxt('All_data_konzentrationen.txt', konz, delimiter=",", fmt='%s')

    return savgol_arr, columns, konz #im grunde nicht mehr benötigt


# liest mW und s aus dem Namen aus
def get_mW_and_s(string):
    '''
        Laserleistung und Belichtungszeit werden ausgelesen
        
        input:  string (str) 
                
        output: mW * s
    '''

    #sucht nach mW und geht von dort an nach links, bis keine zahl mehr aufgerufen wird
    x = string.find('mW') -1
    y = 1
    mW = 0
    while string[x] in '0123456789':
        mW += int(string[x]) * y
        y *=10
        x -=1
    
    #Der unterstrich ist wichtig, da sonst auch die s in wasser gefunden werden könnten
    x = string.find('s_') -1
    y = 1
    s = 0
    while string[x] in '0123456789':
        s += int(string[x]) * y
        y *=10
        x -=1
    
    
    return mW * s #Multiplikation um die Energie in millijoule zu bekommen

def get_concentrations(list): #In der Funktion wird davon ausgegangen, dass die Konzentration die erste zahl im dateinamen ist  WICHTIG: . statt ,
    '''
        Konzentrationen werden von Liste in Array umgewandelt
        
        input:  list (list) 
                
        output: konz (array)
    '''
    
    konz = []
    for i in list:
        if i == 'x':  #Eine Exception für die x-achse da ohne Zahl sonst ein Fehler auftreten würde. hier wird für weitere verarbeitung ein -1 Platzhalter gewählt. 
            konz.append(-1)
        elif '%' in i:
            konz.append(float(re.findall(r'-?\d+\.?\d*', i)[0])*10)
        else:
            konz.append(float(re.findall(r'-?\d+\.?\d*', i)[0]))

    return konz


def get_urea_data(folder = 'Urea_Messungen', save_data = True):   #Im Übergeordneten Ordner müssen die einzelnen Dateien hier vom Typen 'Urea (20mg, 25.02.2022) 24mW 5s'
    '''
        Urea Daten im Ordner werden eingelesen und in einen Numpy Array umgewandelt
        
        input:  folder (str) 
                save_data (boolean)
                
        output: x (array)
                columns (array)
                konz (array) 
    '''
    
    if folder not in  os.getcwd():              #Der ordner wird geöffnet
        os.chdir(folder)

    files = os.listdir()                        #Die Files werden als strings eingelesen
    data = []
    columns = []


    for i in files:                             #Für alle Files wird die Funktion readout file ausgeführt
        a,b  =readout_file(i)                   #a enthält die daten und wird in einer gewöhnliche Python liste abgespeichert. b enthält die Namen der Arrays
        data.append(a)
        columns +=b


    #np.concatenate()
    x = np.vstack([data[i] for i in range(len(data))])   #Hier wird von einer Python liste zu einem Numpy Array gewechselt
    konz = get_concentrations(columns)
    os.chdir("..")
    if save_data:	                                       #Das verzeichnis wird zurückgesetzt und die Daten abgespeichert(reihenfolge gut oder wechseln?)
        np.savetxt("Urea_data.csv", x, delimiter=",")
        np.savetxt('Urea_Columns.txt', columns, delimiter=",", fmt='%s')
        np.savetxt('Urea_konzentrationen.txt', konz, delimiter=",", fmt='%s')
    return x, columns, konz                                   #Durch das abspeichern im grunde redundand


def readout_file(file_name):
    '''
        Einzelne Datei wird eingelesen um diesen zu normen
        
        input:  file_name (str) 

        output: arr_glatt (array) 
                name_array (array)
    '''
    
    array = np.loadtxt(file_name)                        #Eine einzelne Datei wird eingelesen

    array = array.transpose()                            #Durch diesen Befehl hat das array die Form a[22][36**] wodurch es leichter zu glätten ist
    #print(type(array))
    x_free_array = np.stack([array[i] for i in range(1, array.shape[0], 2)])        #Hier werden die x-achsen entfernt, welche an allen geraden stellen im Array liegen entfernt(a[0],a[2] sind x-achsen)

    element_name = file_name.replace('.txt','')                                     #Das .txt ist komplett irrelevant und wird entfernt
    arr_glatt = np.stack([sg.savgol_filter(x_free_array[i],21,5) for i in range(x_free_array.shape[0])])        #Die daten werden mit savitzky golais geglättet


    name_array = []
    
    for i in range(x_free_array.shape[0]):
        name_array.append(element_name + '_' + str(i+1))            #Die dateinamen werden von 'Urea (20mg, 25.02.2022) 24mW 5s.txt' zu 'Urea (20mg, 25.02.2022) 24mW 5s_1' transformiert. (Der unterstrich ist für das finden der sekunden wichtig)
        arr_glatt[i] = arr_glatt[i] / get_mW_and_s(name_array[i])


    return arr_glatt, name_array

def get_combined_data(folder = 'Urea_Messungen', save_data = True):
    '''
        Speichert alle Messdaten von Urea in einem Array und filter diese in 3 verschiedene Dateien.
        
        input:  folder (str) 
                save_data (boolean) 
                
        output: None
    '''
    
    Urea_arr, Urea_columns, Urea_konz = get_urea_data(folder, False)
    allData_arr, all_Data_columns, allData_konz = refine_all_data(save_data=False)

    Complete_arr = np.vstack([allData_arr, Urea_arr])
    complete_columns = all_Data_columns + Urea_columns
    complete_konz = allData_konz + Urea_konz

    if save_data:
        np.savetxt("Complete_Data.csv", Complete_arr, delimiter=",")
        np.savetxt('Complete_Data_columns.txt', complete_columns, delimiter=",", fmt='%s')
        np.savetxt('Complete_konzentrationen.txt', complete_konz, delimiter=",", fmt='%s')
