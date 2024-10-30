

import numpy as np
from yaptxd.utils import sinc_pulse
from yaptxd.spokes import SpokesForm
from yaptxd.maps import FieldMap
from yaptxd.sta import StaOpt

sinc2 = sinc_pulse(256, 2)
spokes = SpokesForm(3)

spokes.set_subpulse(sinc2, 2.0)

spokes.set_ksamples(np.array([[0,0], [1,1], [0,0]])*15)

spokes.plot_pulse()

m = FieldMap('demo-data/phantom-mat/AdjDataUser.mat')

sta = StaOpt(spokes, m.flattened())
sta.solve_mls()
sta.plot_sta()

print("")