
from yaptxd.maps import FieldMap
from yaptxd.utils import sinc_pulse
from yaptxd import static_shim

m = FieldMap('demo-data/phantom-mat/AdjDataUser.mat')
m.quick_plot()
pulse = sinc_pulse(256, 4)
x = static_shim.static_shim(m, pulse)

