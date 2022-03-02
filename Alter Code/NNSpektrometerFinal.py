"""
!pip install -q -U keras-tuner  #Installieren von Tuner Modul von Keras
%load_ext tensorboard
""" 

import tensorflow as tf   # type: ignore         #Hier sind alle entsprechenden Imports
import sklearn # type: ignore
import keras_tuner # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras import datasets, layers, models, optimizers # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.layers.experimental import preprocessing # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from tensorflow.keras.callbacks import TensorBoard # type: ignore
from keras.wrappers.scikit_learn import KerasClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from tensorflow.keras.metrics import RootMeanSquaredError # type: ignore
from tensorflow.keras.optimizers import Adam, SGD # type: ignore
from keras_tuner.tuners import RandomSearch # type: ignore
from sklearn.model_selection import KFold # type: ignore

import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore

## Hier wird unser prepariertes Datenset als CSV Datei geladen
dataset = pd.read_csv('ki_data_set.csv',
                 names = ["Konzentration", "Integrationszeit", "Leistung", "Peak 1", "Peak 2"])              #Es werden den einzelnen Features bzw. Spalten des Dataframe Namen verteilt
dataset = dataset.sample(frac = 1, random_state=27)        #Das Datenset wird geshuffled und ausgegeben
dataset
#%%writefile CSVDataSet.ipynb

dataset_test = pd.read_csv('ki_data_set_test2.csv',      #Gleiches Prinzip wie in obiger Zelle, es wurden nur die neuen Testdaten zur Evaluierung genommen
                 names = ["Konzentration", "Integrationszeit", "Leistung", "Peak 1", "Peak 2"])
dataset_test = dataset_test.sample(frac = 1, random_state=27)
print(dataset_test.head())

X = dataset.iloc[:, 1:]    # alle Features in X reinpacken / Indizierung über .loc, da dataframe
Y = dataset.iloc[:,0]    # alle labels in Y reinpacken
train_samples, test_samples, train_labels, test_labels = train_test_split(
    X, Y, test_size=0.2, random_state=27)    #  random_state wie seed zur Reproduzierbarkeit, es wurde in trainings- und Testdaten gesplittet
# #dataset.describe()

X_test = dataset_test.iloc[:, 1:]    #gleiches Vorgehen mit den Testdaten, diese sind schon geshuffled
Y_test = dataset_test.iloc[:,0]

def plot_single_feature(feature, x = None, y = None):      #Diese Zelle ist nicht wichtig für weiteres vorgehen, lediglich zur Veranschaulichung von Abhängigkeiten einzelner Features
  plt.figure(figsize = (14, 6))
  plt.scatter(train_samples[feature], train_labels, label = 'Intensität', marker = 'x', s = 15)
  if x is not None and y is not None:
    plt.plot(x, y, coler = 'k', label = 'Predictions')
  plt.xlabel(feature)
  plt.ylabel('Konzentration')

  # plot_single_feature("Peak 1")

  scaler = MinMaxScaler(
      feature_range=(-1, 1))  # Es wird MinMaxScaler zum normalisieren der Daten zwischen -1:1 genommen
  scaled_train_samples = scaler.fit_transform(train_samples)
  scaled_test_samples = scaler.fit_transform(test_samples)

  scaler_test = MinMaxScaler(
      feature_range=(-1, 1))  # Hier kann der Skalierungsbereich nochmal explizit angegeben werden
  scaled_test_test_samples = scaler_test.fit_transform(X_test)


# def create_model(hp):                     #Dieser Abschnitt wurde komplett auskommentiert, da er nur einmal ausgeführt werden musste. Es wird versucht hier ein optimales Model für unseren Anwendungsfall zu finden
#     model = keras.Sequential()
#     for i in range(hp.Int('num_layers', 2, 5)):      #iteriert von 2-5 durch
#         model.add(layers.Dense(units = hp.Int('units_' + str(i),      #Fügt dann entsprechend Layer hinzu und erstellt Neuronen
#                                             min_value = 8,
#                                             max_value = 128,
#                                             step = 8),             #Hier wählt man die Schrittweite der Iteration der Neuronenanzahl
#                                activation = 'relu'))               #Aktivierungsfunktion festlegen
#     model.add(layers.Dense(1))                  #Ein Outputlayer, da Regressionsproblem
#     model.compile(                     #Das Model wird kompiliert
#         optimizer = keras.optimizers.Adam(
#             hp.Choice('learning_rate', [1e-1, 1e-2, 1e-3, 1e-4])),             #Es wird nocheinmal die Lernrate Variiert, da diese ein weiterer entscheidener Hyperparameter ist
#         loss = 'mean_squared_error',
#         metrics = ['RootMeanSquaredError'])
#     return model

# #https://keras.io/keras_tuner/   Quelle für unser Hyperparameter tuning

# hp_optimizer = RandomSearch(        #Nun wird die Funktion random Search über das Modell in der obigen Zelle ausgeführt
#     create_model,
#     objective=keras_tuner.Objective("val_root_mean_squared_error", direction="min"),
#     max_trials=15,           #Maximale Anzahl der Versuche wird festgelegt
#     executions_per_trial=8,        #Anzahl einzelner Ausführungen pro Versuch
#     directory='project',
#     project_name='Pentandiol15')


# hp_optimizer.search_space_summary()      #führe diese Funktion aus für Überblick über die Ergebnisse vom Objekt hp_optimizer

# hp_optimizer.search(scaled_train_samples, train_labels,        #Hier arbeitet der hp_optimizer nochmal gezielt mit unseren Daten
#              epochs=5,
#              validation_data=(scaled_test_samples, test_labels))

# hp_optimizer.results_summary()               'Es wird das Ergebnis der besten Performance eines Modells ausgegeben

complex_model = Sequential([         #Das beste Modell, sprich mit dem geringsten RME, wird hier aufgebaut
    Dense(units=64, input_shape=(4,), activation='relu'),         #Inputshape sind unsere Features, Dichte Layer, Aktivierungsfunktion ist Rectified linear unit, da wir nur Werte >= 0 haben
    Dense(units=64, activation='relu'),    #Weitere Layer werden hinzugefügt
    Dense(units=48, activation='relu'),
    Dense(units=88, activation='relu'),
    Dense(units=1)    # , activation='sigmoid' softmax alternativ, ohne Aktivierungsfunktion als Output hat sich jedoch bewehrt
])
complex_model.summary()     #Gibt Eigenschaften des Modells wieder

complex_model.compile(optimizer=Adam(learning_rate=0.01),    #Es wird der Optimizer und die Lernrate festgelegt.
              loss='MeanSquaredError', metrics=['RootMeanSquaredError'])    # , metrics=['accuracy'] -> bei regressionen nicht sinnvoll , sondern nur bei Klassifikation, deshalb hier beides auf MSE


root_logdir = os.path.join(os.curdir, "my_logs")             #erstellt einen neuen Ordner, indem die runs alle separat gespeichert werden
def get_run_logdir():
  import time
  run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")      #Speichert die runs mit entsprechenden Namen ab
  return os.path.join(root_logdir, run_id)             #wechselt wieder das Abreitsverzeichnis zurück

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)      #Runs werden mit callbacks verknüpft

n_split = 5
# , shuffle = True, random_state = 27
for train_index, test_index in KFold(n_split).split(
        X):  # Hier wird das K-fold Crossvalidation verfahren verwendet, um Overfitting zu vermeiden. Es werden 5 folds genommen
    train_samples_kfold, test_samples_kfold = X.iloc[train_index], X.iloc[test_index]
    train_labels_kfold, test_labels_kfold = Y.iloc[train_index], Y.iloc[test_index]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_train_samples_kfold = scaler.fit_transform(train_samples_kfold)
    # scaled_test_samples_kfold = scaler.fit_transform(test_samples_kfold)
    # model=create_model()
    # print(train_index, 'und', test_index)
    # print(test_samples_kfold)
    complex_model.fit(scaled_train_samples_kfold, train_labels_kfold, epochs=140, validation_split=0.2, batch_size=12,
                      callbacks=[
                          tensorboard_cb])  # Hier wird das Modell trainiert. epochen pro durchlauf, batchsize und Validationsplit können eingestellt werden

# print('Model evaluation ', vorlauf_model.evaluate(x_test,y_test))

#%tensorboard --logdir my_logs    #Ergebnisse des Trainings in Tensorboard

predictions_complex = complex_model.predict(x=scaled_test_samples, batch_size=32, verbose=0)  #Predictions des trainierten Modells
print('Gegenüberstellung der vorhergesagten und tatsächlichen Konzentrationen\n')
pred_list_complex = []      #Erstellt Liste für Predictions
for i in range(len(predictions_complex)):      #Vergleicht Vorhergesagte Werte mit eigentlichen Werten (Labels)
  true_value = test_labels.iloc[i]
  pred_value = predictions_complex[i]
  print(f'vorhergesagt:\t{np.around(pred_value, 1)}\t tatsächlich:\t{true_value}')
  pred_list_complex.append(pred_value)

mean_error1 = np.around(np.mean([abs(predictions_complex[i] - test_labels.iloc[i]) / test_labels.iloc[i] for i in range(len(test_labels))])*100, 2)       #Gibt den durchschnittlichen, relativen Fehler aus
print(f'Durchschnittliche/r Abweichung/Fehler der gesamten Vorhersagen:\t{mean_error1}%\nBzw. durchschnittliche Modellgenauigkeit liegt bei:\t\t{100-mean_error1}%')


vorlauf_model = Sequential([                           #Erstellt neben dem komplexen Modell ein simples Modell für den Vergleich
    Dense(units=64, input_shape=(4,), activation='relu'),    # input_shape: Anzahl an features (hier: test_samples.shape[1]=4)
    Dense(units=1)    # , activation='sigmoid' softmax alternativ
])
vorlauf_model.summary()

vorlauf_model.compile(optimizer=Adam(learning_rate=0.01),    #optimizer Adam und Lernrate 0.01 wird gewählt
              loss='MeanSquaredError', metrics=['RootMeanSquaredError'])    # , metrics=['accuracy'] -> bei regressionen nicht sinnvoll bzw. für classification


root_logdir = os.path.join(os.curdir, "my_logs")             #erstellt einen neuen Ordner, indem die run alle separat gespeichert werden
def get_run_logdir():
  import time
  run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
  return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)      #Runs werden mit callbacks verknüpft


root_logdir = os.path.join(os.curdir, "my_logs")             #erstellt einen neuen Ordner, indem die run alle separat gespeichert werden
def get_run_logdir():
 import time
 run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
 return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)      #Runs werden mit callbacks verknüpft

n_split = 5  # Gleiches Vorgehen wie für das andere Modell

for train_index, test_index in KFold(n_split).split(X):  # , shuffle = True, random_state = 27
    train_samples_kfold, test_samples_kfold = X.iloc[train_index], X.iloc[test_index]
    train_labels_kfold, test_labels_kfold = Y.iloc[train_index], Y.iloc[test_index]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_train_samples_kfold = scaler.fit_transform(train_samples_kfold)
    vorlauf_model.fit(scaled_train_samples_kfold, train_labels_kfold, epochs=140, validation_split=0.2, batch_size=12,
                      callbacks=[tensorboard_cb])

# print('Model evaluation ', vorlauf_model.evaluate(x_test,y_test))


 #!rm -rf my_logs    #hier können alte Logfiles in my_log gelöscht werden

#% tensorboard - -logdir

my_logs  # speichert log files vom Tensorboard

# def plot_loss(history):
#     plt.plot(history.history['loss'], label = 'train_loss')
#     plt.plot(history.history['val_loss'], label='val_loss')
#     plt.ylim([0, 2])
#     plt.xlabel('Epoch')
#     plt.ylabel('Error [Konz]')
#     plt.legend()
#     plt.grid(True)
# plot_loss(history)


predictions = vorlauf_model.predict(x=scaled_test_samples, batch_size=32, verbose=0)    #Es werden Labels mit Hilfe des einfachen Modells hervorgesagt
print('Gegenüberstellung der vorhergesagten und tatsächlichen Konzentrationen\n')
pred_list = []
for i in range(len(predictions)):
  true_value = test_labels.iloc[i]                 #Es werden alle richtigen Werte gespeichert
  pred_value = predictions[i]                      #Es werden alle vorhergesagten Werte gespeichert
  print(f'vorhergesagt:\t{np.around(pred_value, 1)}\t tatsächlich:\t{true_value}')     #Gibt die entsprechende Gegenüberstellung aus
  pred_list.append(pred_value)


mean_error0 = np.around(np.mean([abs(predictions[i] - test_labels.iloc[i]) / test_labels.iloc[i] for i in range(len(test_labels))])*100, 2)    #berechnet den durchschnittlichen relativen Fehler
print(f'Durchschnittliche/r Abweichung/Fehler der gesamten Vorhersagen:\t{mean_error0}%\nBzw. durchschnittliche Modellgenauigkeit liegt bei:\t\t{100-mean_error0}%')        #gibt den Fehler aus


predictions_test = complex_model.predict(x=scaled_test_test_samples, batch_size=12, verbose=0)    #Hier werden die eigentlichen Testdaten benutzt und das komplexe Modell getestet
print('Gegenüberstellung der vorhergesagten und tatsächlichen Konzentrationen\n')                 #Es wird analog zu dem einfachen Modell vorgegangen
pred_list_test = []
for i in range(len(predictions_test)):
  true_value_test = Y_test.iloc[i]
  pred_value_test = predictions_test[i]
  print(f'vorhergesagt:\t{np.around(pred_value_test, 1)}\t tatsächlich:\t{true_value_test}')
  pred_list_test.append(pred_value_test)


predictions_test2 = vorlauf_model.predict(x=scaled_test_test_samples, batch_size=12, verbose=0)          #Hier wird das einfache Modell anhand der Testdaten getestet, Vorgehen analog
print('Gegenüberstellung der vorhergesagten und tatsächlichen Konzentrationen\n')
pred_list_test2 = []
for i in range(len(predictions_test2)):
  true_value_test2 = Y_test.iloc[i]
  pred_value_test2 = predictions_test2[i]
  print(f'vorhergesagt:\t{np.around(pred_value_test2, 1)}\t tatsächlich:\t{true_value_test2}')
  pred_list_test2.append(pred_value_test2)


mean_error = np.around(np.mean([abs(predictions_test[i] - Y_test.iloc[i]) / Y_test.iloc[i] for i in range(len(Y_test))])*100, 2)         #Endgültigen Modellgenauigkeiten des komplexen Modells
print(f'Durchschnittliche/r Abweichung/Fehler der gesamten Vorhersagen:\t{mean_error}%\nBzw. durchschnittliche Modellgenauigkeit liegt bei:\t\t{100-mean_error}%')


mean_error2 = np.around(np.mean([abs(predictions_test2[i] - Y_test.iloc[i]) / Y_test.iloc[i] for i in range(len(Y_test))])*100, 2)          #Endgültigen Modellgenauigkeiten des einfachen Modells
print(f'Durchschnittliche/r Abweichunsg/Fehler der gesamten Vorhersagen:\t{mean_error2}%\nBzw. durchschnittliche Modellgenauigkeit liegt bei:\t\t{100-mean_error2}%')



#Bestimme Konfidenzintervall von 90% für die einzelnen Schätzungen
#schreibe zunächst Listen mit den jeweiligen Predictions enthalten
list_20 = []
list_50 = []
list_100 = []
for i in range (len(pred_list)):   #nehme hier pred_list, diese kann aber auch anders sein, je nach prediction
  if pred_list[i] <= 30:           #hier werden die einzelnen Konzentrationen in weitere Listen geschrieben
    list_20.append(pred_list[i])
  if 30 <= predictions_test2[i] <= 70:
    list_50.append(pred_list[i])
  if predictions[i] > 70:
    list_100.append(pred_list[i])
#bestimme den Durchschnittswert:
x20_mean = 0
for n in range (len(list_20)):
  x20_mean += list_20[n]/len(list_20)

x50_mean = 0
for n in range (len(list_50)):
  x50_mean += list_50[n]/len(list_50)

x100_mean = 0
for n in range (len(list_100)):
  x100_mean += list_100[n]/len(list_100)
#bestimme standardabweichung
sigma_20 = 0
for k in range (len(list_20)):
  sigma_20 += (((list_20[k] - 20)**2)/len(list_20))**0.5

sigma_50 = 0
for k in range (len(list_50)):
  sigma_50 += (((list_50[k] - 50)**2)/len(list_50))**0.5

sigma_100 = 0
for k in range (len(list_100)):
  sigma_100 += (((list_100[k] - 100)**2)/len(list_100))**0.5
#bestimme Konfidenz Intervalle:
untere_Grenze_20mg = x20_mean - 1.645*sigma_20
obere_Grenze_20mg = x20_mean + 1.645*sigma_20
untere_Grenze_50mg =x50_mean - 1.645*sigma_50
obere_Grenze_50mg =x50_mean + 1.645*sigma_50
untere_Grenze_100mg =x100_mean - 1.645*sigma_100
obere_Grenze_100mg =x100_mean + 1.645*sigma_100
print('UntereGrenze:', untere_Grenze_20mg, 'ObereGrenze:', obere_Grenze_20mg)        #Gibt die Grenzen der Konfidenzintervalle aus
print('UntereGrenze:', untere_Grenze_50mg, 'ObereGrenze:', obere_Grenze_50mg)
print('UntereGrenze:', untere_Grenze_100mg, 'ObereGrenze:', obere_Grenze_100mg)



#Bestimme Konfidenzintervall von 90% für die einzelnen Schätzungen
#schreibe zunächst Listen mit den jeweiligen Predictions enthalten
list_20 = []
list_50 = []
list_100 = []
for i in range (len(pred_list)):
  if pred_list[i] <= 30:
    list_20.append(pred_list[i])
  if 30 <= predictions_test[i] <= 70:
    list_50.append(pred_list[i])
  if predictions[i] > 70:
    list_100.append(pred_list[i])
#bestimme den Durchschnittswert:
x20_mean = 0
for n in range (len(list_20)):
  x20_mean += list_20[n]/len(list_20)

x50_mean = 0
for n in range (len(list_50)):
  x50_mean += list_50[n]/len(list_50)

x100_mean = 0
for n in range (len(list_100)):
  x100_mean += list_100[n]/len(list_100)
#bestimme standardabweichung
sigma_20 = 0
for k in range (len(list_20)):
  sigma_20 += (((list_20[k] - 20)**2)/len(list_20))**0.5

sigma_50 = 0
for k in range (len(list_50)):
  sigma_50 += (((list_50[k] - 50)**2)/len(list_50))**0.5

sigma_100 = 0
for k in range (len(list_100)):
  sigma_100 += (((list_100[k] - 100)**2)/len(list_100))**0.5
#bestimme Intervalle:
untere_Grenze_20mg = x20_mean - 1.645*sigma_20
obere_Grenze_20mg = x20_mean + 1.645*sigma_20
untere_Grenze_50mg =x50_mean - 1.645*sigma_50
obere_Grenze_50mg =x50_mean + 1.645*sigma_50
untere_Grenze_100mg =x100_mean - 1.645*sigma_100
obere_Grenze_100mg =x100_mean + 1.645*sigma_100
print('UntereGrenze:', untere_Grenze_20mg, 'ObereGrenze:', obere_Grenze_20mg)
print('UntereGrenze:', untere_Grenze_50mg, 'ObereGrenze:', obere_Grenze_50mg)
print('UntereGrenze:', untere_Grenze_100mg, 'ObereGrenze:', obere_Grenze_100mg)


# import os
# filename_path = 'C:/Users/Mo/Projekt/Neural Network/NNspektrometermodel.h5'
# if os.path.isfile(filename_path) is False:
#   model.save(filename_path)    # save architectures, weights, training config and state of optimizer


## for just saving only the architecture of the model
# save as json
# json_string = model.to_json()


from tensorflow.keras.models import load_model # type: ignore
loaded_model = load_model('NNspektrometermodel.h5')   #Model st als .h5 Datei gespeichert
# loaded_model.get_weights() # to see what the trained weights looks like


## for loading json
# from tensorflow.keras.models import model_from_json
# model_architecture = model_from_json(json_string)
