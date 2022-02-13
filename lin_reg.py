import numpy as np # type: ignore
import os
#import scipy.signal as sg
import re
#import csv
import random
#from numpy import genfromtxt  ----> data = genfromtxt('daten.csv', delimiter = ',') um csv einzulesen
def lineare_regression(anzahl_versuche, folder='Messdaten'):
    ''' 
        führt komplette lineare Regression durch

        input: anzahl_versuche (),
               folder (str), Name des Ordners in dem sich die Messdaten befinden 
                
        output: betas (int), Betas der linearen Regression
                abweichung_min (int), Minimale Abweichung
                
    '''
    total_data_array, concentrations = create_data_arrays(folder)
    text_file_length = len(total_data_array[0][0])
    
    abweichung_min = float('inf')
    betas = []
    for i in range(len(concentrations)):
        concentrations[i] = float(concentrations[i])

    for x in range(anzahl_versuche):
        b1, b2, b3, b4, b5, b6, b7, b8, b9 = get_random_betas(text_file_length)

        
        abweichung_ges = 0

        for i in range(len(total_data_array)):

            for j in range(len(total_data_array[i])):
                x = get_array_sum(total_data_array[i][j], b1, b2)
                y = get_array_sum(total_data_array[i][j], b3, b4)
                #z = get_array_sum(total_data_array[i][j], b8, b9)

                #abweichung_ges += abs((((x**b5) / (y**b6)+ z) * b7 - float(concentrations[i])))
                #abweichung_ges += abs((x**b5) * b7 - float(concentrations[i]))
                abweichung_ges += abs((x**b5 -y) * b7 - concentrations[i])
                
        if abweichung_ges < abweichung_min:
            betas = [b1,b2,b3,b4,b5,b6,b7,b8,b9]
            abweichung_min = abweichung_ges

    return betas,abweichung_min
        

        
        
def get_array_sum(array, b1, b2):
    ''' 
        errechnet Summe aus dem Array

        input: array (list), 
               b1 (int), 
               b2 (int),
                
        output: sum (int), Errechnete Summe
                
    '''
    sum = 0
    for i in range(b1, b2):
        sum += array[i][1]

    return sum


        
def get_random_betas(len):
    ''' erstellt zufällige Betas für Regression

        input: len(int),   
                
        output: beta 1-9(int), zufällig gewählte betas
                
    '''
    x1 = random.randint(709,756)
    x2 = random.randint(709,756)
    beta1 = min(x1,x2)
    beta2 = max(x1,x2)
    if beta1 == beta2:
        beta2 +=1
    x1 = random.randint(756,1000)
    x2 = random.randint(756,1000)
    beta3 = min(x1,x2)
    beta4 = max(x1,x2)
    if beta3 == beta4:
        beta4 +=1

    beta5 = random.random() * 3
    beta6 = random.random() * 3

    beta7 =  random.randint(1,1000000)

    x1 = random.randint(0,len-1)
    x2 = random.randint(0,len-1)
    beta8 = min(x1,x2)
    beta9 = max(x1,x2)



    return beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8, beta9


def get_sum_arrays(folder='Messdaten'):
    ''' 
    
        input: folder (str), Name des Ordners in dem sich die Messdaten befinden 
                
        output: compact_array (list),
                
    '''
    
    total_data_array, concentrations = create_data_arrays(folder)
    compact_array = []
    x_axis = get_row(total_data_array[0][0],0)
    text_file_length = len(total_data_array[0][0])
    diff_concentrations = len(concentrations)
    
    for i in range(diff_concentrations):

        number_of_messes = len(total_data_array[i])          #gibt die anzahl der messungen bei aktueller konz. an. (ist ein joke ich bin nicht dumm)
        concentration = []
        for j in range(text_file_length):
            x = 0
            for k in range(number_of_messes):
                #print(total_data_array[i][k][j][1],i,k,j, len(total_data_array[i][k][j]))
                x += total_data_array[i][k][j][1]
            
            concentration.append(x)
        compact_array.append(concentration)

    x_axis_np = np.array(x_axis)
    compact_array_np = np.array(compact_array)
    concentrations_np = np.array(concentrations)
    #np.rot90(compact_array_np)
    np.savetxt("x_achse_np.csv", x_axis_np, delimiter=",")
    np.savetxt("daten.csv", compact_array_np, delimiter=",")
    #np.savetxt('konzentrationen.csv', concentrations_np, delimiter=",")
    np.savetxt('konzentrationen.txt', concentrations_np, delimiter=",", fmt='%s')

    return compact_array
            





def get_row(array, row):
    ''' 
        errechnet Reihe für 

        input: array (list), 
               row (int), Reihennummer   
                
        output: erg (list), errechnete Reihe
                
    '''
    erg=[]
    for i in array:
        erg.append(i[row])

    return erg


def create_data_arrays(folder='Messdaten'):
    ''' 
        Das hier erstellte Array total_data_array hat folgende Form:
        Die Oberste Schicht der arrays entspricht den verschiedenen Kontentrationen(Bsp. 50mg 20mg etc.)
        Diese sind in concentrations abgespeichert um diese später zuordnen zu können

        In dem unterarray total_data_array[x] sind nun alle Messdaten mit der entsprechenden Konzentration abgespeichert,
        wobei die messwerte für konformität durch die verwendete Leistung (zeit*energie) geteilt wurden.

        input:  folder(str), Ordner in dem sich befinden
                
        output: total_data_array (list), 
                concentrations (list),
                
    '''
    if folder not in  os.getcwd():   
        os.chdir(folder)
    folder = os.getcwd()
    file_names = [file for file in os.listdir() if file[0] != '.' and '_intensity' not in file and '.pdf' not in file and '_graphs' not in file and '.csv' not in file and '.txt' not in file]
    total_data_array = []
    concentrations = []
    for i in file_names:
        #array, konzentrationen, data_ges = load_data(folder +'\\'+ i)
        total_energy = re.findall(r'-?\d+\.?\d*', i)
        x = 1
        for j in total_energy:      #total energy/x gibt das produkt von dauer und leistung des bestrahlens wieder
            x *= float(j)


        #sub_file_names = os.listdir(i)
        sub_file_names = [file for file in os.listdir(i) if '.csv' not in file]
        for j in sub_file_names:
            
            konzentration = re.findall(r'-?\d+\.?\d*', j)[0]
            array = messdaten_einlesen_und_normen(i+'\\'+j, x)

            if konzentration not in concentrations:
                concentrations.append(konzentration)
                total_data_array.append([array])
            else:
                for k in range(len(concentrations)):
                    if concentrations[k] == konzentration:
                        total_data_array[k].append(array)
    return total_data_array, concentrations


def messdaten_einlesen_und_normen(folder, total_energy):
    ''' 
        liest Messdaten ein und normt diese

        input: folder (str), Name des Ordners in dem sich die Messdaten befinden  
               total_energy (int),  
                
        output: array (list), 
                
    '''
    array = np.loadtxt(folder)

    for i in range(len(array)):
        array[i][1] /= total_energy

    return array

