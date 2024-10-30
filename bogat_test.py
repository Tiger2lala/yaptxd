import numpy as np
from yaptxd.utils import sinc_pulse
from yaptxd.spokes import SpokesForm
from yaptxd.maps import FieldMap
from yaptxd.sta import StaOpt
from yaptxd.gradient_opt import BOGAT

sinc2 = sinc_pulse(256, 2)
spokes = SpokesForm(3)

spokes.set_subpulse(sinc2, 2.0)

spokes.set_ksamples(np.array([[-1,0], [0,1], [0,0]])*10)

# spokes.plot_pulse()

m = FieldMap('demo-data/phantom-mat/AdjDataUser.mat')
m.group_slices()

bogat_solver = BOGAT(m, spokes)
bogat_solver.set_rf_kwargs(tikhonov=1e-2)
bogat_solver.optimize(k_range=15, phase_adoption=False)

sta = StaOpt(bogat_solver.pulse_form_best[0], m.flattened_maps[0])
sta.create_A_matrix()
sta.coeff = bogat_solver.coeff_best[0][0]
sta.plot_sta()

print(bogat_solver.pulse_form_best[0].spoke_location)
print(bogat_solver.coeff_best)
