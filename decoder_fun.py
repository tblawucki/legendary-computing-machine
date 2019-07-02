#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from matplotlib.widgets import Slider, Button, RadioButtons


class IntProp:
    def __init__(self, init_val = 0):
        self.value = init_val
        
    def __call__(self):
        return self.value

    def set(self, val):
        self.value = val
        
decoder = load_model('./models/decoder.pkl')
decoder.summary()

fig, ax = plt.subplots(figsize=(10, 20))
plt.subplots_adjust(left=0.25, bottom=0.25)

sequence = np.random.uniform(0, 1, 10)
sequence = np.zeros((10,))
decoded = decoder.predict(sequence.reshape(-1, 10)).ravel()
l, = plt.plot(decoded, lw=2)
plt.axis([-5, 55, -2, 2])
feature_idx = IntProp(0)

axcolor = 'lightgoldenrodyellow'

features_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
featuresSlider = Slider(features_ax, f'feature val', -5, 5, valinit=0, valstep=0.05)
    
#sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0, valstep=delta_f)
#samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

def update(val):
#    amp = samp.val
    print('changing')
    sequence[feature_idx()] = featuresSlider.val
    print('SQ SHP: ',sequence.shape)
    reshaped = sequence.reshape(-1, 10)
    print('SQ SHP: ',sequence.shape)
    decoded = decoder.predict(reshaped)
    l.set_ydata(decoded.ravel())
    fig.canvas.draw_idle()

featuresSlider.on_changed(update)

#sfreq.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    for i in range(10):
        sequence[i] = 0
    featuresSlider.set_val(0)
button.on_clicked(reset)

# =============================================================================
rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, [f'feature {i}' for i in range(10)], active=0)


def colorfunc(label):
    feature_idx.set( int(radio.value_selected[-1]) )
    featuresSlider.set_val(sequence[feature_idx()])
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)
# =============================================================================

plt.show()
