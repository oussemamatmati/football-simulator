import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from footballsim import config as C
from footballsim.plotting import plot_possession, zone_heatmap


def test_plot_possession_and_heatmap_run():
    fig, ax = plt.subplots(1, 2)
    plot_possession(ax[0], ["Pass_ok_z8", "Carry_na_z12", "Shot_Goal_z22"], "demo")
    plot_possession(ax[0], [])  # empty possession must not crash
    zone_heatmap(ax[1], np.ones(C.N_ZONES) / C.N_ZONES, "uniform")
    plt.close(fig)
