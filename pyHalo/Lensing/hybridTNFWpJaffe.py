import numpy as np
from pyHalo.Lensing.TNFW import TNFWLensing
from pyHalo.Lensing.PJaffe import PJaffeLensing

class TNFWpJaffeLensing(object):

    hybrid = True

    lenstronomy_ID = ['TNFW', 'PJAFFE']

    def __init__(self, lens_cosmo = None, zlens = None, z_source = None):

        if lens_cosmo is None:
            from pyHalo.Cosmology.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo(zlens, z_source)

        self.lens_cosmo = lens_cosmo

        self._tnfw = TNFWLensing(lens_cosmo)
        self._pjaffe = PJaffeLensing(lens_cosmo)

    def _interpolating_function(self, rs, r_core, b_min = 0.25,
                 b_max = 0.75, b_crit = 0.5, c = 1):

        beta = r_core * rs ** -1

        b_half = 0.5 * (b_max - b_min)

        arg = (beta - b_crit) * b_half ** -1

        return 0.5 * (1 + np.tanh(c * arg))

    def params(self, x, y, mass, redshift, concentration, r_trunc, b, r_1):

        kwargs_tnfw = self._tnfw.params(x, y, mass, redshift, concentration, r_trunc)

        r_truncpjaffe = kwargs_tnfw['Rs'] * 1000
        r_core = b * kwargs_tnfw['Rs']
        pjaffe_mass = self.pjaffe_mass_normalization(kwargs_tnfw, b)

        kwargs_pjaffe = self._pjaffe.params(x, y, pjaffe_mass, redshift,
                                          concentration, r_truncpjaffe, r_core)

        f = self._interpolating_function(r_core, r_1)

        kwargs_tnfw['theta_Rs'] = kwargs_tnfw['theta_Rs'] * f
        kwargs_pjaffe['theta_Rs'] = kwargs_pjaffe['theta_Rs'] * (1-f)

        return [kwargs_tnfw, kwargs_pjaffe], None

    def pjaffe_mass_normalization(self, m, c, z):
        """
        :param m200: m200
        :return: physical mass corresponding to m200
        """
        rho0, Rs, r200 = self.lens_cosmo.NFW_params_physical(m,c,z)
        return 4*np.pi*rho0*Rs**3*(np.log(1+c)-c*(1+c)**-1)


