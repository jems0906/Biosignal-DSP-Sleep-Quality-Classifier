"""Load MNE across SciPy releases that renamed sph_harm."""
import scipy.special


if not hasattr(scipy.special, "sph_harm") and hasattr(scipy.special, "sph_harm_y"):
    def sph_harm(m, n, theta, phi):
        return scipy.special.sph_harm_y(n, m, phi, theta)

    scipy.special.sph_harm = sph_harm

import mne

__all__ = ["mne"]
