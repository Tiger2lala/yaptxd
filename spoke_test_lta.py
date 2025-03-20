

import numpy as np
from yaptxd.utils import sinc_pulse
from yaptxd.spokes import SpokesForm
from yaptxd.maps import FieldMap
from yaptxd.sta import StaOpt
from yaptxd.lta import LtaOpt

sinc2 = sinc_pulse(256, 2)
spokes = SpokesForm(2)

spokes.set_subpulse(sinc2, 2.0)

spokes.set_ksamples(np.array([[1,1], [0,0]])*15)

m = FieldMap('demo-data/phantom-mat/AdjDataUser.mat')

sta = StaOpt(spokes, m.flattened())
sta.solve_mls(tikhonov=1e-6)
sta.plot_sta()

print("")

lta = LtaOpt(spokes, m.flattened())
lta.tikhonov = 1e-6
lta.coeff = sta.coeff
lta.optimize()

sta.est_fa = lta.est_fa
sta.plot_sta()