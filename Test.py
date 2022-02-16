import numpy as np
import matplotlib.pyplot as plt
from scyipy.signal import savgol_filter
from matplotlib.widgets import Slider

#Testdaten
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x) + np.cos(x) + np.random.random(100)

#Savitzky-Golay Filter anwenden
y_filtered = savgol_filter(y, 99, 3)

#Plotting
fig = plt.figure()
ax = fig.subplots()
p, = ax.plot(x,y_filtered, 'g')
plt.subplots_adjust(bottom=0.25)

#Slider einfügen
ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03]) #x position, y position, width, height
win_len = Slider(ax_slide, 'Window length', valmin=5, valmax=99, valinit=99, valstep=2)

#Update Funktion für Plot
def update(val):
    current_v = int(win_len.val)
    new_y = savgol_filter(y, current_v, 3)
    p.set_ydata(new_y)
    fig.canvas.draw()

# calling the function "update" when the value of the slider is changed
win_size.on_changed(update)
plt.show()