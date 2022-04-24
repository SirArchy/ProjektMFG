import numpy as np
#import os
#import scipy.signal as sg
#import re
#import cs
import random
#from numpy import genfromtxt  ----> data = genfromtxt('daten.csv', delimiter = ',') um csv einzulesen
def lineare_regression(anzahl_versuche,stoff='Pentan'):
    '''
        
        input:  anzahl_versuche (int), 
                stoff (str), 
                
        output: None (bzw. es werden verschiedene Graphen geplottet, und/oder Daten abgespeichert)
    '''

    #Daten Werden als Numpy Arrays eingelesen Data enthält die Messwerte, Columns wird verwendet um den Stoff auszulesen und Konz um die zu erwartende Konzentration zu berechnen
    Data    = np.loadtxt('Complete_Data.csv', delimiter=",")
    Columns = np.genfromtxt('Complete_Data_columns.txt',delimiter='\n', dtype=str)
    Konz    = np.loadtxt('Complete_konzentrationen.txt')

    peak = 0
    if stoff == 'Pentan':
        peak = 730
    if stoff == 'Urea':
        peak = 329
    versuche_str = str(anzahl_versuche)
    abweichung_min = float('inf')  
    betas_min = []
    data_length = len(Konz)
    genutzte_messreihen = 0
    messung_betrag_gesamt = []
    for x in Columns:
        if stoff in x or 'Wasser' in x:
            genutzte_messreihen += 1
    
    teta = 0
    zeta_jones = 0
    
    for i in range(data_length):
        messung_betrag_gesamt.append(np.sum(Data[i]) / len(Konz))
        if 'Wasser' in Columns[i]:
            teta+=1
            zeta_jones += Data[i][peak] / messung_betrag_gesamt[i]
    baseline_höhe = zeta_jones / teta

    for x in range(anzahl_versuche):

        if x %1000 ==0:
            print('Fortschritt: ' + str(x) + ' / ' + versuche_str)


        betas = get_random_betas(len(Data[0]),stoff)#Jeden Durchlauf werden zufällig bestimmte Betas generiert
        abweichung_ges = 0

        for i in range(data_length):  #Die meißten Messwerte sind wasser --> mögliches Problem durch zu stark gewichtete Messdaten  Vorschlag: Messungen, welche den Stoff enthalten stärker gewichten

            if stoff in Columns[i]:
                abweichung_ges += calculate_abw(Data[i], betas, Konz[i], messung_betrag_gesamt[i], baseline_höhe)  #Die daten mit dem gesuchten Stuff

            elif 'Wasser' in Columns[i]:  #Alternativ statt 'Wasser' geht auch (stoff not in Columns[i] and Konz[i] > 0) das 2. argument ist hauptsächlich wegen der x-achse wichtig
                abweichung_ges += calculate_abw(Data[i], betas, 0, messung_betrag_gesamt[i], baseline_höhe)

    #Checkt ob die aktuellen betas besser als die vorherigen besten sind und speichert diese ggf. ab         
        if abweichung_ges < abweichung_min:
            betas_min = betas
            abweichung_min = abweichung_ges

    abweichung_average = (abweichung_min/ genutzte_messreihen)
    Write_results_to_txt(betas_min, abweichung_average, stoff)



def calculate_abw(array, betas, konz, summe, baseline): #Hier wird die abweichund zwischen durch lin-reg erwarteter konz. und dem ergebnis ausgerechner und *aktuell* quadriert(ist das smart?)
    '''

        
        input:  array (array), 
                betas (int), 
                konz (int),  
                summe (int),
                baseline (int), 
                
        output: beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8 (int)
    '''
    '''
    Currently best for Penthan
    step_1 = abs(array[730] -array[betas[2]])
    step_2 = step_1 **betas[4]
    step_3 = step_2 * betas[6]
    step_4 = abs(step_3 - konz)
    return step_4
    '''
    step_1 = (array[730]) / summe -baseline*betas[5]
    if step_1 < 0:
        step_1 = 0
    step_2 = step_1 **betas[4]
    step_3 = step_2 * betas[6]
    step_4 = abs(step_3 - konz)
    return step_4

def get_random_betas(len, stoff): #die Funktion generiert betas und geht für die betas, welche Flächen abdecken sollen sicher, das diese aufsteigend sortiert sind
    ''' 

        
        input:  len (int),
                stoff (str), 
                
        output: beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8 (int)
    '''
    
    beta1 = random.randint(709,756)#Beta 1 und 2 sollen den Pentan peak einfangen
    beta2 = random.randint(720,730)

    beta3 = random.randint(1,len-1)#Beta 3 und 4 sollen den H2o peak einfangen
    beta4 = random.randint(756,1000)

    beta5 = random.random() * 2
    if beta5 < 0.001:
        beta5 = 1
    beta6 = random.random() * 3

    beta7 =  random.randint(1,1000)# beta 7 soll die niedrigen Messwerte anheben, kein plan hat sich richtig angefühlt
    beta8 =  random.randint(1,100000)




    return [beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8]



def get_array_sum(array, b1, b2): #berechnet die summe der messwerte zwichen zwei Punkten, ist bestimmt irgendwie sinnvoll
    ''' 
        Lädt die Daten aus den .txt Datein, sliced auf den angegebenen Wellenlängenbereich, glättet die Funktion, 
        erzeugt neue Daten aus den übergebenen Datenmatrizen, visualisiert die Spektren, berechnet die Intensitätswerte,
        plottet Intensitätswerte über Konzentration.
        Optional: Speichert die erzeugten Daten und die berechneten Intensitätswerte als Arrays ab.
        
        input:  array (array), 
                b1 (int),
                b2 (int), 
                
        output: sum (int), 
    '''
    
    sum = 0
    
    for i in range(b1, b2):
        sum += array[i]
    return sum


def Write_results_to_txt(betas, abweichung, stoff):
    ''' 

        
        input:  betas (array), 
                abweichung (int), 
                stoff (str), 
                
        output: None
    '''
    with open('results.txt', 'a') as file:
        for i in betas:
            file.write(str(i)+' ')
        file.write('die Berechnete durchschnittliche Abweichung beträgt: ' + str(abweichung) + ' für den Stoff: ' + stoff)
        file.write('\n')
        #file.write('\n'.join(str(item) for item in betas))


def backwards(betas, stoff = 'Pentan'):
    ''' 

        input:  betas (array), 
                stoff (str), 
                
        output: konzentrationen (array),
                Konz_abw (array),
    '''
    Data    = np.loadtxt('Complete_Data.csv', delimiter=",")
    Columns = np.genfromtxt('Complete_Data_columns.txt',delimiter='\n', dtype=str)
    Konz    = np.loadtxt('Complete_konzentrationen.txt')
    messung_betrag_gesamt = []
    data_length = len(Konz)
    peak = 0
    if stoff == 'Pentan':
        peak = 730
    if stoff == 'Urea':
        peak = 329

    teta = 0
    zeta_jones = 0
#    peak_heights = []
    for i in range(data_length):
        messung_betrag_gesamt.append(np.sum(Data[i]) / len(Konz))
#        peak_heights.append(np.argmax(Data[i][peak-10:peak+10]))

        if 'Wasser' in Columns[i]:
            teta+=1
            zeta_jones += Data[i][peak] / messung_betrag_gesamt[i]
    baseline_höhe = zeta_jones / teta

    Konz_abw = []
    konzentrationen=[]
    erg = []


    for i in range(len(Data)):
        if stoff in Columns[i]:
            Konz_and_mws = [Konz[i], get_mW_and_s(Columns[i])]
            #print(Konz_and_mws)
            #print(Konz_and_mws not in konzentrationen)
            #print(konzentrationen)
            if Konz_and_mws not in konzentrationen:
                konzentrationen.append(Konz_and_mws)
                Konz_abw.append(calculate_abw(Data[i], betas, Konz[i], messung_betrag_gesamt[i], baseline_höhe))
            else:
                for j in range(len(konzentrationen)):
                    if konzentrationen[j] == Konz_and_mws:
                        Konz_abw[j] += calculate_abw(Data[i], betas, Konz[i], messung_betrag_gesamt[i], baseline_höhe)
    #for i in range(len(konzentrationen)):
    #    erg.append()

    return konzentrationen, Konz_abw

def get_mW_and_s(string):
    ''' 

        input:  string (str), 
                
        output: mW * s (int),  
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
