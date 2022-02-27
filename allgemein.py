import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import os
import scipy.signal as sg # type: ignore
import re


def mein(folder='Messdaten', lambda_1=600, lambda_2=660, window_length=21, poly_order=5, save_phs=False, show_plot=False):
    if folder not in  os.getcwd():   
        os.chdir(folder)
    folder = os.getcwd()
    file_names = [file for file in os.listdir() if file[0] != '.' and '_intensity' not in file and '.pdf' not in file and '_graphs' not in file]

    for i in file_names:
        daten_aufbereiten(folder +'\\'+ i, lambda_1, lambda_2, window_length, poly_order, save_phs, show_plot)

def daten_aufbereiten(folder, lambda_1=600, lambda_2=660, window_length=21, poly_order=5, save_phs=False, show_plot=False):    
    ''' Lädt die Daten aus den .txt Datein, sliced auf den angegebenen Wellenlängenbereich, glättet die Funktion, 
        erzeugt neue Daten aus den übergebenen Datenmatrizen, visualisiert die Spektren, berechnet die Intensitätswerte,
        plottet Intensitätswerte über Konzentration.
        Optional: Speichert die erzeugten Daten und die berechneten Intensitätswerte als Arrays ab.
        
        input:  Verzeichnis (str), Ordner, aus dem die Daten geladen und bearbeitet werden
                lambda_1, lambda_2, (int, int), Start- und Endwert des angegebenen Wellenlängenbereichs (int)(angegebener Defaultwert)
                window_length (int) (muss positiv und ungerade sein) Parameter für den Savitzky-Golay Filter
                poly_order (int) (muss kleiner sein als window_length) Parameter für den Savitzky-Golay Filter
                
        output: None 
                (bzw. es werden verschiedene Graphen geplottet, und/oder Daten abgespeichert)
    '''
    ## Daten laden
    data, konzentration, data_ges = load_data(folder)
    
    ## Daten slicen
    sliced_data = []
    for i in data:
        sliced_data.append(slice_data(i, lambda_1, lambda_2))
    sliced_data_ges = slice_data(data_ges, lambda_1, lambda_2)
    ## Daten filtern
    clean_data = []
    for i in sliced_data:
        clean_data.append(filter_data(i, window_length, poly_order))
    clean_data_ges = filter_data(sliced_data_ges, window_length, poly_order)
    ## Daten erzeugen
    #extra_data_20 = create_data(folder, clean_data_20, '20')
    #extra_data_50 = create_data(folder, clean_data_50, '50')
    #extra_data_100 = create_data(folder, clean_data_100, '100')
    #extra_data = np.concatenate([extra_data_20, extra_data_50, extra_data_100], axis=-1)
    
    #if save_d:
    #    ## Erzeugte 'Extra'-Daten speichern
    #    save_extra_data(extra_data_20, folder, '20')
    #    save_extra_data(extra_data_50, folder, '50')
    #    save_extra_data(extra_data_100, folder, '100')
    #    save_extra_data(extra_data, folder, 'all')
    
    ## Plots der Spektren erstellen
    for i in range(len(konzentration)):
        plot_spectra(clean_data[i], folder, konzentration[i],show_plot)
    # plot_spectra(clean_data_20, folder, '20')
    # plot_spectra(clean_data_50, folder, '50')
    # plot_spectra(clean_data_100, folder, '100')
    # plot_spectra(clean_data, folder)
    # plot_spectra(extra_data, folder, extra=True) # Visualisierung der künstlich erzeugten Daten

    ## Intensitäten berechnen
    peak_heights_data = peak_heights_per_set(clean_data_ges)   
    #peak_heights_extra = peak_heights_per_set(extra_data)
    
    ## Intensität zu Konzentration plotten
    #plot_peak_heights(folder, peak_heights_data)
    
    if save_phs:
        ## Intensitätsarrays speichern:
        for i in range(len(konzentration)):
            save_intensity(clean_data[i], folder, konzentration[i])
        #save_intensity(clean_data_20, folder, '20')
        #save_intensity(clean_data_50, folder, '50')
        #save_intensity(clean_data_100, folder, '100')
        #save_intensity(clean_data, folder, 'all')

        #save_intensity(extra_data_20, folder, '20_extra')
        #save_intensity(extra_data_50, folder, '50_extra')
        #save_intensity(extra_data_100, folder, '100_extra')
        #save_intensity(extra_data, folder, 'all_extra')

    return

def load_data(folder):
    ''' Liest die .txt Datein im übergebenen Ordner ein und gibt die Daten als numpy arrays zurück.
    
        input:  folder (str), Name des Verzeichnis in dem die Daten als .txt Datein liegen
        output: data, data_20, data_50, data_100, (np arrays), (Dimension: (# Wertepaare, 2, # Spektren)), die eingelesenen Daten
    ''' 

    # Wechsel des Arbeitsverzeichnisses ins angegebene Verzeichnis:
    if folder not in  os.getcwd():   
        os.chdir(folder)

    # Liste aller relevanten Dateinamen in dem aktuellen Verzeichnis (ohne versteckte Datein wie z.B. '.checkpoints'):
    file_names = [file for file in os.listdir() if file[0] != '.'] 
    # sortieren der Dateinamen in der Liste nach aufsteigender Konzentration:
    def sorting_key(file_name): # gibt die Konzentration aus dem Dateinamen zurück
        
        x = kon_Auslesen(file_name)

        return float(x[0])

    def kon_Auslesen(name):

        return re.findall(r'-?\d+\.?\d*', name)

    def kon_arr_erstellen(filenames):

        arr = []

        for i in filenames:
            j = kon_Auslesen(i)
            if j[0] not in arr and j[0] != []:
                arr.append(j[0])
        return arr

    a = []
    konzentrationen = kon_arr_erstellen(file_names)
    for i in konzentrationen:
        #a.append([np.loadtxt(file) for file in file_names if i in file])
        DONDAbetterthanCLB = [np.loadtxt(file) for file in file_names if i in file] #Interessante Variablen Namen xD
        KanyeForPresident = np.stack(DONDAbetterthanCLB,axis=-1)
        a.append(KanyeForPresident)

    #if [] in a:    
    #    a.remove([])
    
    file_names = sorted(file_names, key=sorting_key)               

    # Daten aller Messungen bei gleichen Rahmenbedingungen in eine Matrix:
    files = [np.loadtxt(file) for file in file_names]
    data_ges = np.stack(files, axis=-1)     
    
    # Daten der 5 gleichen Messungen in jeweils eine Matrix:
    #files_20 = [np.loadtxt(file) for file in file_names if '20' in file]
    #data_20 = np.stack(files_20,axis=-1)
    #files_50 = [np.loadtxt(file) for file in file_names if '50' in file]
    #data_50 = np.stack(files_50,axis=-1)
    #files_100 = [np.loadtxt(file) for file in file_names if '100' in file]
    #data_100 = np.stack(files_100,axis=-1)

    # Arbeitsverzeichnis wieder zurücksetzten so wie vor der Funktion
    os.chdir("..")    
    
    return a, konzentrationen, data_ges

def slice_data(data_matrix, lambda_1, lambda_2):
    ''' Geht durch alle Spektren in der übergebenen Datenmatrix und ruft für jedes Spektrum slice_spectrum() auf.
    
        input:  data_matrix (np array), Datenmatrix, dessen kompletter Inhalt gesliced wird.
                lambda_1 (int), linker Rand des betrachteten Fensters
                lambda_2 (int), rechter Rand des betrachteten Fensters
                
        output: new_data_matrix (np array), Datenmatrix mit reduzierten Spektren.
    '''
    
    new_data_matrix = np.stack([slice_spektrum(data_matrix[...,i], lambda_1, lambda_2) for i in range(data_matrix.shape[-1])], axis=-1)

    return new_data_matrix

def slice_spektrum(frame, lambda_1, lambda_2):
    ''' Reduziert ein Spektrum auf den Bereich zwischen lambda_1 und lambda_2.
    
        input:  frame (np array), das Spektrum was gesliced wird
                lambda_1 (int), linker Rand des betrachteten Fensters
                lambda_2 (int), rechter Rand des betrachteten Fensters
                
        output: new_frame (np array), auf den angegebenen Wellenlängenbereich reduziertes Spektrum
    '''
    
    id_1 = np.searchsorted(frame[:,0], lambda_1)
    id_2 = np.searchsorted(frame[:,0], lambda_2)
    
    new_frame = frame[id_1:id_2]
    
    return new_frame


def filter_data(data_matrix, window_length, poly_order):
    ''' Glättet mit dem Savitzky-Golay-Filter aus scipy.signal alle Signale in der übergebenen Datenmatrix. 
        Die inhaltliche Zusammensetzung der data_matrix ist irrelevant, da jedes enthaltene Spektrum einzeln gefiltert wird.

        input:  data_matrix (np array), der zuglättenden Spektren.
                window_length (int), positive, ungerade Zahl (siehe scipy.signal.savgol() documentation für mehr Infos)
                poly_order (int), muss kleiner sein als window_length (siehe scipy.signal.savgol() documentation für mehr Infos)
             
        output: new_data_matrix (np array), neue data_matrix die diesselben Dimensionen hat und die geglätteten Signale enthält       
    '''
    new_data_matrix = np.stack([filter_spectrum(data_matrix[...,i], window_length, poly_order) for i in range(data_matrix.shape[-1])], axis=-1)
      
    return new_data_matrix

def filter_spectrum(frame, window_length, poly_order):
    new_frame = frame.copy()
    new_frame[:,1] = sg.savgol_filter(frame[:,1] , window_length, poly_order)
    
    return new_frame


def plot_spectra(data_matrix, folder, konz='all', extra=False, show_plot = False):
    ''' Erzeugt ein Plot mit allen Spektren, die in der übergebenen data_matrix sind.
    
        input:  data_matrix (np array), die Datenmatrix dessen Inhalt in ein figure geplottet werden soll.
                folder (str), Name des Ordners dessen Inhalt geplottet wird, wird als Titel übernommen.
                konz (str),  optionale Konzentrationsbeschreibung (falls nur ein set geplottet wird) die in den Titel des plots kommt, default: 'all'
                extra (bool), Flag für künstlich erzeugte Daten, die dann in schwarz geplottet werden
                
        output: None 
                (bzw. es werden die Spektren geplottet)
    '''
    # Figure erstellen
    plt.figure(figsize=(20,3))
    
    # Titel bestimmen & Achsenbeschriftung
    if konz == 'all':
        title = folder
    else: 
        title = folder + ": " + konz + ' mg/ml'
        
    plt.title(title)
    plt.xlabel("Wellenlänge [nm]")
    plt.ylabel("Intensität")
    plt.grid(True)
    
    # Schleife geht die einzelnen Scheiben der data_matrix durch und plottet diese
    for i in range(data_matrix.shape[-1]):
        if extra:
            plt.plot(data_matrix[:,0,i], data_matrix[:,1,i], lw=0.4, c='k') 
        else:
            plt.plot(data_matrix[:,0,i], data_matrix[:,1,i], lw=0.4)
    
    a  =''
    for i in os.listdir():
        if i in str(folder):
            a = i

    new_folderx = a + '_graphs' #wenn _graph entfernt wird muss auch mein angepasst werden, da diese datei sonst bei wiederholtem aufführen mit datensatz verwechselt wird
    
    # gegebenenfalls Ordner erstellen
    if new_folderx not in os.listdir():
        os.mkdir(new_folderx)
  
    # wechsel ins neue Verzeichnis
    #os.chdir(new_folderx) 
    
    #Plot speichern und gegebenenfalls anzeigen(Pausiert das Programm also bei großer Anzahl daten nicht zu empfehlen)
    plt.savefig(str(folder)+'_graphs\\'+konz+'mg_pro_ml_plot.pdf', bbox_inches='tight')
    #plt.savefig(str(folder) + konz+'mg_pro_ml_plot.pdf', bbox_inches='tight')

    if show_plot:
        plt.show()    
    
        
    return

def peak_heights_per_set(data_matrix):
    ''' Bestimmt die Peakhöhen für jedes Spektrum in der übergebenen Datenmatrix und gibt diese als numpy-Array zuruück.
    
        input:  complete_data_matrix (np array), Datenmatrix mit Spektren aller Konzentrationen
        output: intens (np array), Array mit folgendem Inhalt: (peak_height_1, peak_height_2) und sovielen Zeilen wie die Datenmatrix
    '''
    
    peak_heights = np.array([peak_heights_per_spektrum(data_matrix[...,i]) for i in range(data_matrix.shape[-1])])
    
    return peak_heights


def peak_heights_per_spektrum(frame, window_size=5, peak_1_left=610, peak_1_right=620, peak_2_left=620, peak_2_right=650):
    ''' Bestimmt die Peakhöhen eines einzelnen Spektrums.
    
        input:  frame (np array), ein einzelnes Spektrum
                window_size (int), ungerade Zahl, die die Anzahl Werte angibt, über die der Durchschnitt gebildet wird, welcher als Maximum angenommen wird
                peak_1_left (int), gibt die Wellenlänge am linken Rand des Fensters des ersten Peaks an
                peak_1_right (int), gibt die Wellenlänge am rechten Rand des Fensters des ersten Peaks an
                peak_2_left (int), gibt die Wellenlänge am linken Rand des Fensters des zweiten Peaks an
                peak_2_right (int), gibt die Wellenlänge am rechten Rand des Fensters des zweiten Peaks an
                
        output: result (np array), Array mit folgendem Inhalt: (peak_height_1, peak_height_2)
    '''
    id_1 = np.searchsorted(frame[:,0], peak_1_left)
    id_2 = np.searchsorted(frame[:,0], peak_1_right)

    max_id = np.argmax(frame[id_1:id_2,1], axis=0)
    max_value_1 = np.mean(frame[id_1+max_id-(window_size//2):id_1+max_id+(window_size//2), 1])

    id_1 = np.searchsorted(frame[:,0], peak_2_left)
    id_2 = np.searchsorted(frame[:,0], peak_2_right)

    max_id = np.argmax(abs(frame[id_1:id_2,1]))
    max_value_2 = np.mean(frame[id_1+max_id-(window_size//2):id_1+max_id+(window_size//2), 1])

    result = np.array([max_value_1, max_value_2])

    return result

def plot_peak_heights(folder, peak_heights, peak_heights_extra=None): # erstellt ein Plot der Intensität über der Konzentration
    ''' Plottet die Peakhöhen über der Konzentration. Erzeugte Daten sind Punkte, gemessene Daten sind Kreuze.
    
        input:  inten_data (np array), Array mit den Peakhöhen aller Spektren des Ordners
        output: intens (np array), Array mit folgendem Inhalt: (peak_height_1, peak_height_2) und sovielen Zeilen wie die Datenmatrix
    '''
    plt.figure(figsize=(10,5))
    
    if peak_heights_extra is not None: # Falls es extra Daten gibt
        
        konzentrationen = np.array([20,20,20,20,20,20,20,20,20,20,50,50,50,50,50,50,50,50,50,50,100,100,100,100,100,100,100,100,100,100])
        
        if peak_heights_extra.shape[1] == 2 : 
            plt.plot(konzentrationen, peak_heights_extra[:,0], marker='o', linestyle='', label='Extra Peak_1', c='blue', ms=4, alpha=0.5)
            plt.plot(konzentrationen, peak_heights_extra[:,1], marker='o', linestyle='', label='Extra Peak_2', c='tab:red', ms=4, alpha=0.5)
        else:
            plt.plot(konzentrationen, peak_heights_extra, marker='x', linestyle='')
    
    konzentrationen = np.array([20,20,20,20,20,50,50,50,50,50,100,100,100,100,100]) 
    
    if peak_heights.shape[1] == 2 : # intens_data.shape = (15,2)
        plt.plot(konzentrationen, peak_heights[:,0], marker='x', linestyle='', label='Peak_1', c='blue')
        plt.plot(konzentrationen, peak_heights[:,1], marker='x', linestyle='', label='Peak_2', c='tab:red')
    else:
        plt.plot(konzentrationen, peak_heights, marker='x', linestyle='')
 
    plt.grid(True)
    plt.xlabel("Konzentration [mg/ml]")
    plt.ylabel("Peakhöhe")
    plt.legend()
    plt.title(folder)
    plt.savefig(folder + '_ph.png', bbox_inches='tight')
    plt.show() 
    
    return


def save_intensity(data_matrix, folder, konz):
    ''' Speichert die berechneten Intensitätswerte als Konzentration-beschrifteten Arrays in einem Ordner pro Rahmenbedingung.

        input:  data_matrix (np array), dessen Intensitätswerte berechnet und abgespeichert werden sollen
                folder (str), Name des Ordners in den gespeichert wird, der dann zusatzlich den suffix "_intensity" bekommt 
                konz (str), Name der .npy Datei, deshalb eigentlich auch eine Konzentrationsbeschreibung um die Daten identifizieren zu können
             
        output: None 
                (bzw. es werden die Arrays abgespeichert)
    '''
    # neuen Dateinamen erstellen
    a  =''
    for i in os.listdir():
        if i in str(folder):
            a = i

    new_folderx = a + '_intensity' #wenn intensity entfernt wird muss auch mein angepasst werden, da diese datei sonst bei wiederholtem aufführen mit datensatz verwechselt wird

    # gegebenenfalls Ordner erstellen
    if new_folderx not in os.listdir():
        os.mkdir(new_folderx)
  
    # wechsel ins neue Verzeichnis
    os.chdir(str(folder) + '_intensity')    
    # falls es das Array noch nicht gibt, wird es hier berechnet und abgespeichert
    intens = peak_heights_per_set(data_matrix)    # Intensität berechnen
    np.save(konz, arr=intens)    # Intensität speichern
    
    # Arbeitsverzeichnis wieder zurücksetzten so wie vor der Funktion
    os.chdir("..")
        
    return #fehlt hier eine Return Variable???