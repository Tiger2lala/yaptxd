import numpy as np
from yaptxd.utils import sinc_pulse, gradient_delay_adjust
from yaptxd.spokes import SpokesForm
from yaptxd.maps import FieldMap
from yaptxd.sta import StaOpt
from yaptxd.gradient_opt import BOGAT
from yaptxd.io import write_spokes_xy_moment, write_spokes_coeff

sinc2 = sinc_pulse(147, 2)
spokes = SpokesForm(2)

spokes.set_subpulse(sinc2, 2.0)

spokes.set_ksamples(np.array([[0,1], [0,0]])*10)

# spokes.plot_pulse()

m = FieldMap('demo-data/vol-8-sli/AdjDataUser.mat')
m.group_slices(2,4)

bogat_solver = BOGAT(m, spokes)
bogat_solver.set_rf_kwargs(tikhonov=1e-2)
bogat_solver.optimize(k_range=15, phase_adoption=False)

coeff_sol = bogat_solver.coeff_best

sta = StaOpt(bogat_solver.pulse_form_best[0], m.flattened_maps[0])
sta.create_A_matrix()
sta.coeff = coeff_sol[0][0]
sta.plot_sta()

print(bogat_solver.pulse_form_best[0].spoke_location)
print(coeff_sol)

for ipulse in range(len(coeff_sol)):
    for iMB in range(len(coeff_sol[0])):
        pulse_id = ipulse + iMB * len(coeff_sol)
        coeff = gradient_delay_adjust(coeff_sol[ipulse][iMB], 
                                  m.z[pulse_id], 
                                  np.max(spokes.g[:,2]), 
                                  np.array([0,0,-10]), 
                                  np.array([0,0,10]))
        write_spokes_coeff(f'{pulse_id}.txt', coeff)
        write_spokes_xy_moment(f'{pulse_id}-xy.txt', bogat_solver.pulse_form_best[ipulse])