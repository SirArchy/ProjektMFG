import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import numpy as np
from mpl_toolkits import mplot3d

data = np.zeros((180, 5))  # Initialisierung der leeren Matrix
row_id = 0


def extract_konz(file_name):  # gibt die Konzentration aus dem Dateinamen zurück
    i = 0
    number = ''
    while file_name[i].isdigit():
        number += file_name[i]
        i += 1
    return int(number)


file_list = [file for file in os.listdir() if
             file[-9:] == 'intensity']  # sucht sich alle relevanten Datein aus dem Verzeichnis

for file in file_list:
    file_strings = file.split('_')  # es entsteht eine Liste folgenden Formats: ["pentan", "10sec", "24mW", "intensity"]
    int_time = int(file_strings[1][:-3])
    leistung = int(file_strings[2][:-2])

    os.chdir(file)

    intens_files = [array for array in os.listdir() if
                    (array[0:3] != 'all' and array[0] != '.')]  # sucht sich die individuellen Konzentrationsarrays
    intens_files = sorted(intens_files, key=extract_konz)  # sortiert diese Dateien aufsteigend nach Konzentration

    for array in intens_files:
        measured_pairs = np.load(array)  # alle berechneten Peakpaare

        for pair in measured_pairs:
            data[row_id, 0] = int_time
            data[row_id, 1] = leistung
            data[row_id, 2] = pair[0]
            data[row_id, 3] = pair[1]
            data[row_id, 4] = extract_konz(array)

            row_id += 1
    os.chdir('..')

k_tilde = data[0:45, 4]
k_tilde = k_tilde.reshape(45, 1)
k_tilde_train = k_tilde

# k_tilde_train = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 50, 50, 50, 50, 50, 50, 50, 50, 50, 100, 100, 100,
# 100, 100, 100, 100, 100, 100]) k_tilde_train = k_tilde_train.reshape(27,1)

# Trainingsdaten
i = 6  # Index zum durchlaufen der Datenmatrix
j = 0  # Index in der neu erstellten Matrix
training_data = np.zeros((108, 5))
while i <= data.shape[0]:
    training_data[j:j + 9] = data[i:i + 9]
    i += 15  # Es gibt immer 15 'gleiche' Spektren in der feature matrix
    j += 9  # wir wollen 9 davon in die Trainingsmatrix tun

# Validierungsdaten
i = 3
j = 0
validation_data = np.zeros((36, 5))
while i <= data.shape[0]:
    validation_data[j:j + 3] = data[i:i + 3]
    i += 15
    j += 3

# Testdaten
i = 0
j = 0
test_data = np.zeros((36, 5))
while i <= data.shape[0]:
    test_data[j:j + 3] = data[i:i + 3]
    i += 15
    j += 3


def create_feature_matrices(data, op_list):
    """ Erstellt für ein Modell vier verschiedene Matrizen für die verschiedenen Rahmenbedingungen.

        input:  data (np array), die Datenmatrix, die in vier verschiedene Datenmatrizen aufgeteilt werden soll
                op_list (list),  List mit den Funktionsbeschreibungen des jeweiligen Modells als Strings

        output: 4x X_i_5s_24mW (np array), die zum Modell i gehörenden nach Rahmenbedingungen aufgeteilten feature matrices
    """

    quarter_id = data.shape[0] // 4  # dynamisch gestaltet um verschieden große Datenmatrizen zu ermöglichen.

    X_i_5s_24mW = create_feature_matrix(data[0:quarter_id], op_list)
    X_i_5s_34mW = create_feature_matrix(data[quarter_id:quarter_id * 2], op_list)
    X_i_10s_24mW = create_feature_matrix(data[quarter_id * 2:quarter_id * 3], op_list)
    X_i_10s_34mW = create_feature_matrix(data[quarter_id * 3:quarter_id * 4], op_list)

    return X_i_5s_24mW, X_i_5s_34mW, X_i_10s_24mW, X_i_10s_34mW


def create_feature_matrix(data_xsec_xmW, op_list):  # nur die kleinen data matrizen übergeben, nicht die gesamte
    """ Erstellt für ein Modell vier verschiedene Matrizen für die verschiedenen Rahmenbedingungen.

        input:  data_xsec_xmW (np array), die Datenmatrix, für die die feature matrix berechnet werden soll
                op_list (list),  List mit den Funktionsbeschreibungen des jeweiligen Modells als Strings

        output: feature_matrix (np array), die mit der übergebenen op_list berechnete feature Matrix
    """

    feature_amount = len(op_list)
    feature_matrix = np.ones((data_xsec_xmW.shape[0],
                              feature_amount))  # falls man keinen Konstanten Term haben will muss man hier zeros nehmen

    # Definitionen der Hilfsfunktionen
    linear = lambda x: x
    quadrat = lambda x: x * x
    wurzel = lambda x: np.sqrt(x)

    for i in range(feature_amount):
        # Abfrage auf welchem Peak die Funktion angewendet werden soll
        if 'ph1' in op_list[i]:
            variable = data_xsec_xmW[:, 2]
            # print(variable)
        elif 'ph2' in op_list[i]:
            variable = data_xsec_xmW[:, 3]

        # Abfrage welche Funktion angewendet werden soll
        if 'konst' in op_list[i]:
            feature_matrix[:, i] = linear(1)
        elif 'linear' in op_list[i]:
            feature_matrix[:, i] = linear(variable)
        elif 'quadrat' in op_list[i]:
            feature_matrix[:, i] = quadrat(variable)
        elif 'wurzel' in op_list[i]:
            feature_matrix[:, i] = wurzel(variable)
        else:
            print("Die übergebene Funktion konnte nicht richtig umgesetzt werden.")

    return feature_matrix


'''
tr_1: Für den Fall wo alle urprünglichen Daten (erste messung + erzeugte Daten) als Trainingsdaten verwendet werden, wir hier mit data aufgerufen
tr_2: Für den Fall dass nur 9 der erzeugten Daten für das training benutzt werden, muss man hier mit training_data aufrufen
'''

# X_1_5s_24mW, X_1_5s_34mW, X_1_10s_24mW, X_1_10s_34mW = create_feature_matrices(training_data, op_list=['konst(1)', 'linear(ph1)', 'linear(ph2)'])
# X_2_5s_24mW, X_2_5s_34mW, X_2_10s_24mW, X_2_10s_34mW = create_feature_matrices(training_data, op_list=['konst(1)', 'linear(ph1)', 'linear(ph2)', 'quadrat(ph1)', 'quadrat(ph2)'])
X_3_5s_24mW, X_3_5s_34mW, X_3_10s_24mW, X_3_10s_34mW = create_feature_matrices(data, op_list=['konst(1)', 'linear(ph1)',
                                                                                              'linear(ph2)',
                                                                                              'wurzel(ph1)',
                                                                                              'quadrat(ph2)'])

# Creating figure
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

# Creating plot
# data = training_data
q_id = training_data.shape[0] // 4

ax.scatter3D(data[0:q_id, 2], data[0:q_id, 3], data[0:q_id, 4], color="green", label='10sec 24mW')
ax.scatter3D(data[q_id:q_id * 2, 2], data[q_id:q_id * 2, 3], data[q_id:q_id * 2, 4], color="blue", label='10sec 34mW')
ax.scatter3D(data[q_id * 2:q_id * 3, 2], data[q_id * 2:q_id * 3, 3], data[q_id * 2:q_id * 3, 4], color="red",
             label='5sec 24mW')
ax.scatter3D(data[q_id * 3:q_id * 4, 2], data[q_id * 3:q_id * 4, 3], data[q_id * 3:q_id * 4, 4], color="orange",
             label='5sec 34mW')

ax.set_xlabel('peak_height_1')
ax.set_ylabel('peak_height_2')
ax.set_zlabel('Konzentration')
ax.legend()

ax.elev = 15
ax.azim = 90
plt.savefig('3d_plot_view_2.png')


def loss(beta, X_i, k_tilde):  # Fehlerberechnung mit MSE

    k = np.dot(X_i, beta)  # f berechnen
    k = k.reshape(len(X_i), 1)

    quad_fehler = (k - k_tilde_train) ** 2  # elementweise quadratischer unterschied

    avg_fehler = (1 / len(k)) * np.sum(quad_fehler)  # Formel anwenden

    return avg_fehler


def multiple_regression(X_i):
    ''' Führt für eine Feature Matrix die Regression durch.

        input:  X_i (np array), die Featurematrix für das Modell i

        output: result (result Objekt), Siehe Dokumentation für minimize() Funktion für weitere Attribute (enthält optimale Gewichte)
    '''
    # Findet die Gewichte, bei der die loss()-Funktion minimal ist
    result = minimize(loss, np.ones((X_i.shape[1], 1)), args=(X_i, k_tilde_train), method='POWELL', tol=1e-5)

    print(f'Anzahl Iterationen: {result.nit}')  # Ausgabe der gefragten Größen

    # for i in range(len(result.x)):
    #    print(f'b_{i} = {result.x[i]:.3e}') # Einzelausgabe der Koeffizienten

    print(f'Fehler (Kosten) = {loss(result.x, X_i, k_tilde):.3e}')

    return result


def multiple_regression_per_set(X_i_5s_24mW, X_i_5s_34mW, X_i_10s_24mW, X_i_10s_34mW):
    ''' Führt für alle vier Rahmenbedingungen eine Regression durch, um unterschiedliche Ergebnisse (und damit unterschiedliche Gewichte) zu bekommen.

        input:  4 x X_i_xs_xmW (np array), die Featurematrizen für das Modell i, für die verschiedenen Rahmenbedingungen

        output: 4 x result_i (result Objekte), Siehe Dokumentation für minimize() Funktion für weitere Attribute
    '''
    result_1 = multiple_regression(X_i_5s_24mW)
    print("\n")
    result_2 = multiple_regression(X_i_5s_34mW)
    print("\n")
    result_3 = multiple_regression(X_i_10s_24mW)
    print("\n")
    result_4 = multiple_regression(X_i_10s_34mW)

    return (result_1, result_2, result_3, result_4)


result_3_1, result_3_2, result_3_3, result_3_4 = multiple_regression_per_set(X_3_5s_24mW, X_3_5s_34mW, X_3_10s_24mW,
                                                                             X_3_10s_34mW)
betas_3 = [result_3_1.x, result_3_2.x, result_3_3.x, result_3_4.x]

# beta_tr_1 sind die Betas, wenn nur ein Teil der ursprünglichen Messungen zum training benutzt wird
# beta_tr_2 sind die Betas, wenn alle ursprünglichen Messungen zum training benutzt werden, und die Testdaten die neuen Messungen sind

if "betas_tr_3" not in os.listdir():
    os.mkdir("betas_tr_3")
os.chdir("betas_tr_3")
np.save("betas_5sec_24mW", betas_3[0])
np.save("betas_5sec_34mW", betas_3[1])
np.save("betas_10sec_24mW", betas_3[2])
np.save("betas_10sec_34mW", betas_3[3])
os.chdir("..")
