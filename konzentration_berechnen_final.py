import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
import scipy.signal as sg # type: ignore


def konzentration(file_path, beta_folder, i=None):  # file_path = z.B. 'Pentan_10sec_24mW_new/Pentan_100mg_w0.txt'
    ''' Berechnet die Konzentration des übergebenen Spektrums.

     input:  file_path (str), der Datenpfad bis zu einem genauen Spektrum
             beta_folder (str), Name des Ordners aus dem die betas ausgelesen werden
             i (int), Angabe welches Spektrum ausgewertet werden soll, wenn der file_path zu einer Datenmatrix führt und nicht zu einem Spektrum, default: 'None'

     output: k (float), berechnete Konzentration

 '''
    test_spektrum = prepare_data(file_path,
                                 i)  # bearbeitet das Spektrum, sodass es in einheitlicher sauberer Form vorliegt

    peak_height_1, peak_height_2 = find_peak_heights(test_spektrum)  # Bestimmung der peak_höhen

    # betas
    betas = betas_aktualisieren(beta_folder)
    beta_key = find_beta_key(file_path)
    betas_xsec_xmW = betas[beta_key]

    k = konzentration_berechnen(peak_height_1, peak_height_2, betas_xsec_xmW)

    return k


def prepare_data(file_path, i, lambda_1=600, lambda_2=660, window_length=21, poly_order=5):
    if 'new' in file_path or 'tr_1' in file_path:
        test_spektrum = new_spektrum_laden(file_path, i)

    else:
        test_spektrum = mess_spektrum_laden(file_path)

    sliced_test_spektrum = slice_data(test_spektrum, lambda_1, lambda_2)
    clean_test_spektrum = clean_data(sliced_test_spektrum, window_length, poly_order)

    return clean_test_spektrum


def mess_spektrum_laden(file_path):  # name eines spektrums als .txt oder .npy
    return np.loadtxt(file_path)


def new_spektrum_laden(file_path, i):  # name eines spektrums als .txt oder .npy z.B Pentan_10sec_24mW_new/100.npy

    test_matrix = np.load(file_path)
    test_spektrum = test_matrix[..., i]

    return test_spektrum


def slice_data(test_spektrum, lambda_1, lambda_2):
    test_spektrum_x = test_spektrum[:, 0]

    start_id = np.searchsorted(test_spektrum_x, lambda_1)
    end_id = np.searchsorted(test_spektrum_x, lambda_2, side='right')

    new_spektrum = test_spektrum[start_id:end_id].copy()

    return new_spektrum


def clean_data(test_spektrum, window_length, poly_order):
    new_spektrum = test_spektrum.copy()
    new_spektrum[:, 1] = sg.savgol_filter(test_spektrum[:, 1], window_length, poly_order)

    return new_spektrum


def betas_aktualisieren(beta_folder):
    betas = {'5sec_24mW': np.load(beta_folder + "/betas_5sec_24mW.npy"),
             '5sec_34mW': np.load(beta_folder + "/betas_5sec_34mW.npy"),
             '10sec_24mW': np.load(beta_folder + "/betas_10sec_24mW.npy"),
             '10sec_34mW': np.load(beta_folder + "/betas_10sec_34mW.npy")}

    return betas


def find_beta_key(file_path):
    if 'tr_1' in file_path:
        if '100' in file_path:
            beta_key = file_path[22:-8]
        else:
            beta_key = file_path[22:-7]
    elif 'tr_2' in file_path:
        if '101' in file_path:
            beta_key = file_path[22:-20]
        elif '18.9' in file_path:
            beta_key = file_path[22:-21]
        else:
            beta_key = file_path[22:-19]

    return beta_key


def find_peak_heights(test_spektrum, window_size=5, peak_1_left=610, peak_1_right=620, peak_2_left=620,
                      peak_2_right=650):  # muss evt. an die Form des Spektrums angepasst werden

    test_spektrum_x = test_spektrum[:, 0]

    id_1 = np.searchsorted(test_spektrum_x, peak_1_left)
    id_2 = np.searchsorted(test_spektrum_x, peak_1_right)

    max_id = np.argmax(test_spektrum[id_1:id_2, 1], axis=0)
    max_value_1 = np.mean(test_spektrum[id_1 + max_id - (window_size // 2):id_1 + max_id + (window_size // 2), 1])

    id_1 = np.searchsorted(test_spektrum_x, peak_2_left)
    id_2 = np.searchsorted(test_spektrum_x, peak_2_right)

    max_id = np.argmax(abs(test_spektrum[id_1:id_2, 1]))
    max_value_2 = np.mean(test_spektrum[id_1 + max_id - (window_size // 2):id_1 + max_id + (window_size // 2), 1])

    return (max_value_1, max_value_2)


def konzentration_berechnen(ph_1, ph_2, betas_xsec_xmW):  # betas_xsec_xmW ist ein numpy array
    b = betas_xsec_xmW

    # Wenn man ein anderes Modell nimmt, muss man hier die Berechnungsvorschrift ändern:
    # k = b[0] + b[1]*ph_1 + b[2]*ph_2 + b[3]*np.sqrt(ph_1) + b[4]*(ph_2 **2)
    k = b[0] + b[1] * ph_1 + b[2] * np.sqrt(ph_1)

    return k


def fehler_spektrum(file_path, beta_folder, i=None, pr=False):
    # tatsächliche Konzentration bestimmen
    if 'new' in file_path:
        file_name = file_path[34:]
    else:
        file_name = file_path[32:]

    k_echt = konzentration_aus_namen(file_name)

    # Konzentration berechnen
    k = konzentration(file_path, beta_folder, i)

    # Zeile erstellen und zurückgeben, um daraus eine Matrix bauen zu können
    fehler_matrix_zeile = np.array([k, k_echt, k - k_echt])

    # hilfreiche Ausgaben
    if pr:
        print(f"Berechnete Konzentration:  {k:.2f} mg/ml")
        print(f"Eigentliche Konzentration: {k_echt}    mg/ml")
        print(f"Differenz: {fehler_matrix_zeile[2]:.2f}\n")

    return fehler_matrix_zeile


def konzentration_aus_namen(file_name):
    if '20' in file_name:
        k_echt = 20
    elif '50' in file_name:
        k_echt = 50
    elif '100' in file_name:
        k_echt = 100
    elif '18.9' in file_name:
        k_echt = 18.9
    elif '56' in file_name:
        k_echt = 56
    elif '101' in file_name:
        k_echt = 101
    else:
        print('Hier passiert ein Fehler.')

    return k_echt


def fehler_per_set(folder, beta_folder, numpy_array=False):
    file_list = [file for file in os.listdir(folder) if file[0] != '.']

    fehler_matrix_20_list, fehler_matrix_50_list, fehler_matrix_100_list, fehler_matrix_all_list = [], [], [], []

    if 'tr_2' in folder:
        n = 6
    else:
        n = 5

    for file_name in file_list:  # entweder [100.npy, 50.npy, 20.npy] oder [Pentan_100mg_w0.txt, Pentan_100mg_w0.txt, ...]

        if numpy_array:
            if '20' in file_name:
                for j in range(n):
                    fehler_matrix_20_list.append(fehler_spektrum(folder + '/' + file_name, beta_folder, j))
            elif '50' in file_name:
                for j in range(n):
                    fehler_matrix_50_list.append(fehler_spektrum(folder + '/' + file_name, beta_folder, j))
            elif '100' in file_name:
                for j in range(n):
                    fehler_matrix_100_list.append(fehler_spektrum(folder + '/' + file_name, beta_folder, j))

        else:
            if '18.9' in file_name:
                fehler_matrix_20_list.append(fehler_spektrum(folder + '/' + file_name, beta_folder, pr=False))
            elif '56' in file_name:
                fehler_matrix_50_list.append(fehler_spektrum(folder + '/' + file_name, beta_folder, pr=False))
            elif '101' in file_name:
                fehler_matrix_100_list.append(fehler_spektrum(folder + '/' + file_name, beta_folder, pr=False))

    fehler_matrix_20 = np.array(fehler_matrix_20_list)
    fehler_matrix_50 = np.array(fehler_matrix_50_list)
    fehler_matrix_100 = np.array(fehler_matrix_100_list)

    fehler_20 = rmse(fehler_matrix_20[:, 2])
    fehler_50 = rmse(fehler_matrix_50[:, 2])
    fehler_100 = rmse(fehler_matrix_100[:, 2])
    if n == 5:
        print(f"Durchschnittlicher Fehler von Konzentration 20:   {fehler_20}")
        print(f"Durchschnittlicher Fehler von Konzentration 50:   {fehler_50}")
        print(f"Durchschnittlicher Fehler von Konzentration 100:  {fehler_100}")
    else:
        print(f"Durchschnittlicher Fehler von Konzentration 18.9:   {fehler_20}")
        print(f"Durchschnittlicher Fehler von Konzentration 56:     {fehler_50}")
        print(f"Durchschnittlicher Fehler von Konzentration 101:    {fehler_100}")
    return


def rmse(array):
    f = np.sqrt((1 / array.shape[0]) * (np.sum(array ** 2)))
    return f


'''
betas = 'betas_tr_2'
fehler_per_set('test_data_tr_2/Pentan_5sec_34mW', betas)
fehler_per_set('test_data_tr_2/Pentan_5sec_24mW', betas)
fehler_per_set('test_data_tr_2/Pentan_10sec_34mW', betas)
fehler_per_set('test_data_tr_2/Pentan_10sec_24mW', betas)
betas = 'betas_tr_4'
fehler_per_set('test_data_tr_2/Pentan_5sec_34mW', betas)
fehler_per_set('test_data_tr_2/Pentan_5sec_24mW', betas)
fehler_per_set('test_data_tr_2/Pentan_10sec_34mW', betas)
fehler_per_set('test_data_tr_2/Pentan_10sec_24mW', betas)

k = konzentration('test_data_tr_2/Pentan_5sec_34mW/Pentan_101mg_w5.txt', 'betas_tr_2')
print(f"Die berechnete Konzentration für das obige Spektrum beträgt {k} mg/ml.")
'''
