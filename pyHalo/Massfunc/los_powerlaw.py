import numpy as np
from pyHalo.defaults import *
from pyHalo.Spatial.uniform import LensConeUniform
from copy import copy, deepcopy
from pyHalo.Massfunc.parameterizations import BrokenPowerLaw

class LOSPowerLaw(object):

    def __init__(self, args, lensing_mass_function):

        self.mass_function_kwargs = self.check_kwargs(args)
        zmin, zmax = args['zmin'], args['zmax']

        self._lensing_mass_func = lensing_mass_function
        self._geometry = lensing_mass_function.geometry

        self.log_mlow, self.log_mhigh = args['log_mlow'], args['log_mhigh']
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

    def _draw_at_redshift(self, z, delta_z, boost):

        log_m_centers = self._lensing_mass_func.log_mass_centers
        mstep = log_m_centers[1] - log_m_centers[0]
        mfuncs = []

        for logmi in log_m_centers:

            mfunc_kwargs = deepcopy(self.mass_function_kwargs)
            norm = boost * self._lensing_mass_func.norm_at_z(logmi, z, delta_z)
            slope = self._lensing_mass_func.plaw_index_z(logmi, z)

            mfunc_kwargs['normalization'] = norm * self.LOS_norm
            mfunc_kwargs['power_law_index'] = slope
            mfunc_kwargs['log_mlow'] = np.round(logmi - mstep/2, 1)
            mfunc_kwargs['log_mhigh'] = np.round(logmi + mstep/2, 1)
            mfuncs.append(BrokenPowerLaw(**mfunc_kwargs))

        init = True
        masses = []
        x_arcsec, y_arcsec, r2d, r3d = [], [], [], []

        for mfunc in mfuncs:
            if init:
                masses = mfunc.draw()
                x_arcsec, y_arcsec, r2d, r3d = self._render_positions_atz(z, len(masses))
                init = False
            else:
                new_masses = mfunc.draw()
                masses = np.append(masses, new_masses)
                x_new, y_new, r2d_new, r3d_new = self._render_positions_atz(z, len(new_masses))
                x_arcsec = np.append(x_arcsec, x_new)
                y_arcsec = np.append(y_arcsec, y_new)
                r2d = np.append(r2d, r2d_new)
                r3d = np.append(r3d, r3d_new)

        return masses, x_arcsec, y_arcsec, r2d, r3d

    def __call__(self):

        zlens = self._lensing_mass_func.geometry._zlens

        if self._compute2halo:
            try:
                z_2halo_term = self._redshift_range[np.where(self._redshift_range < zlens)][-1]
            except:
                z_2halo_term = None
        else:
            z_2halo_term = None

        init = True
        delta_z = self._redshift_range[1] - self._redshift_range[0]

        for idx, zcurrent in enumerate(self._redshift_range):

            if zcurrent == self._lensing_mass_func.geometry._zlens:
                continue

            if zcurrent == z_2halo_term:
                place_at_zlens = True
                rmax = self._lensing_mass_func._cosmo.T_xy(zlens - delta_z, zlens)
                boost = self._lensing_mass_func.two_halo_boost(M_halo=self.mass_function_kwargs['parent_m200'],
                                                               z=zcurrent, rmax=rmax)

            else:
                place_at_zlens = False
                boost = 1

            mi, xi, yi, r2di, r3di = self._draw_at_redshift(zcurrent, delta_z, boost)

            if len(mi) > 0:
                if place_at_zlens:
                    redshifts = [self._geometry._zlens] * len(mi)
                else:
                    redshifts = [zcurrent] * len(mi)
            else:
                redshifts = []

            if init:
                masses, x, y, r2d, r3d, z = mi, xi, yi, r2di, r3di, np.array(redshifts)
                init = False
            else:
                masses = np.append(masses, mi)
                x, y = np.append(x, xi), np.append(y, yi)
                r2d, r3d = np.append(r2d, r2di), np.append(r3d, r3di)
                z = np.append(z, np.array(redshifts))

        return masses, x, y, r2d, r3d, z, [False]*len(masses)

    def check_kwargs(self, args):

        mass_function_kwargs = {}

        required_keys = ['zmin', 'zmax', 'log_m_break', 'log_mlow',
                         'log_mhigh', 'parent_m200', 'LOS_normalization',
                         'draw_poisson']

        for key in required_keys:
            assert key in args.keys()

        required_keys_mfunc = ['log_mlow', 'log_mhigh', 'draw_poisson', 'parent_m200']

        for key in required_keys_mfunc:
            assert key in args.keys()
            mass_function_kwargs[key] = args[key]

        if 'log_m_break' in args.keys():
            assert 'c_power' in args.keys()
            assert 'c_scale' in args.keys()
            assert 'break_index' in args.keys()
            mass_function_kwargs['log_m_break'] = args['log_m_break']
            mass_function_kwargs['c_power'] = args['c_power']
            mass_function_kwargs['c_scale'] = args['c_scale']
            mass_function_kwargs['break_index'] = args['break_index']
        else:
            mass_function_kwargs['log_m_break'] = 0.
            mass_function_kwargs['c_power'] = 0.
            mass_function_kwargs['c_scale'] = 1.
            mass_function_kwargs['break_index'] = 1.

        return mass_function_kwargs
