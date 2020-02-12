import numpy as np
from pyHalo.defaults import *
from pyHalo.Spatial.uniform import LensConeUniform
from copy import copy, deepcopy
from pyHalo.Massfunc.parameterizations import DeltaFunction

class LOSDelta(object):

    def __init__(self, args, lensing_mass_function):

        self.mass_function_kwargs = self.check_kwargs(args)
        zmin, zmax = args['zmin'], args['zmax']

        self._lensing_mass_func = lensing_mass_function
        self._geometry = lensing_mass_function.geometry

        self.logM = self.mass_function_kwargs['logM_delta']
        self.LOS_norm = args['LOS_normalization']

        zstep = lenscone_default.default_z_step
        self._redshift_range = np.arange(zmin, zmax + zstep, zstep)
        self._spatial_parameterization = LensConeUniform(args['cone_opening_angle'],
                                                         lensing_mass_function.geometry)

        self._compute2halo = self._lensing_mass_func._two_halo_term

    def _render_positions_atz(self, z, nhalos):

        x_kpc, y_kpc, r2d_kpc, r3d_kpc = self._spatial_parameterization.draw(nhalos, z)

        kpc_per_asec = self._geometry.kpc_per_arcsec(z)
        x_arcsec = x_kpc * kpc_per_asec ** -1
        y_arcsec = y_kpc * kpc_per_asec ** -1

        return x_arcsec, y_arcsec, r2d_kpc, r3d_kpc

    def _draw_at_redshift(self, z, delta_z):

        mfunc_kwargs = deepcopy(self.mass_function_kwargs)
        mfunc_kwargs['normalization'] = self._lensing_mass_func.dN_comoving_deltaFunc(10**mfunc_kwargs['logM_delta'],
                                                                                      z, delta_z,
                                                                                      mfunc_kwargs['mass_fraction'])
        mfunc = DeltaFunction(**mfunc_kwargs)

        masses = mfunc.draw()
        x_arcsec, y_arcsec, r2d, r3d = self._render_positions_atz(z, len(masses))

        if len(masses) > 0:
            redshifts = [z] * len(masses)
        else:
            redshifts = []

        return masses, x_arcsec, y_arcsec, r2d, r3d, np.array(redshifts)

    def __call__(self):

        init = True
        delta_z = self._redshift_range[1] - self._redshift_range[0]

        for idx, zcurrent in enumerate(self._redshift_range):

            if zcurrent == self._lensing_mass_func.geometry._zlens:
                continue

            mi, xi, yi, r2di, r3di, zi = self._draw_at_redshift(zcurrent, delta_z)

            if init:
                masses, x, y, r2d, r3d, z = mi, xi, yi, r2di, r3di, zi
                init = False
            else:
                masses = np.append(masses, mi)
                x, y = np.append(x, xi), np.append(y, yi)
                r2d, r3d = np.append(r2d, r2di), np.append(r3d, r3di)
                z = np.append(z, zi)

        return masses, x, y, r2d, r3d, z, [False]*len(masses)

    def check_kwargs(self, args):

        mass_function_kwargs = {}

        required_keys = ['zmin', 'zmax', 'logM_delta', 'LOS_normalization', 'mass_fraction',
                         'draw_poisson']

        for key in required_keys:
            assert key in args.keys()
            mass_function_kwargs[key] = args[key]

        return mass_function_kwargs
