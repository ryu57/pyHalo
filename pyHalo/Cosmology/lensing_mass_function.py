from colossus.lss.mass_function import *
from pyHalo.Cosmology.geometry import *
from scipy.interpolate import RegularGridInterpolator
from pyHalo.defaults import *
from colossus.lss.bias import twoHaloTerm

class MassFunctionInterp(object):

    def __init__(self, log_mlow, log_mhigh, mfunc_function, zmax, delta_z):

        self.log_mstep = 0.1

        zvalues = np.arange(0.01, zmax+delta_z, delta_z)

        logm_centers = np.arange(log_mlow, log_mhigh, self.log_mstep) + self.log_mstep/2
        if logm_centers[-1] > log_mhigh:
            logm_centers = logm_centers[0:-1]

        zz, logmm = np.meshgrid(zvalues, logm_centers)

        shape0 = zz.shape

        slope_array, log_norm_array = [], []

        for i, (zi, logmm) in enumerate(zip(zz.ravel(), logmm.ravel())):

            log_m_values = np.linspace(logmm - self.log_mstep/2, logmm + self.log_mstep/2, 10)
            log_dN_dmdV = np.log10(mfunc_function(10 ** log_m_values, zi))
            [slope, log_norm] = np.polyfit(log_m_values, log_dN_dmdV, 1)

            slope_array.append(slope)
            log_norm_array.append(log_norm)

        log_norm_array = np.array(log_norm_array).reshape(shape0)
        slope_array = np.array(slope_array).reshape(shape0)
        points = (logm_centers, zvalues)

        self._interp_norm = RegularGridInterpolator(points, log_norm_array)
        self._interp_slope = RegularGridInterpolator(points, slope_array)
        self.log_mass_centers = logm_centers

    def norm(self, log_mass, redshift):
        return 10**self._interp_norm((log_mass, redshift))

    def slope(self, log_mass, redshift):
        return self._interp_slope((log_mass, redshift))

class LensingMassFunction(object):

    def __init__(self, cosmology, mlow, mhigh, zlens, zsource, cone_opening_angle,
                 mass_function_model=None, two_halo_term=True,
                 geometry_type=None):

        self._cosmo = cosmology

        if mass_function_model is None:
            mass_function_model = realization_default.default_mass_function

        if geometry_type is None:
            geometry_type = lenscone_default.default_geometry

        self.geometry = Geometry(cosmology, zlens, zsource, cone_opening_angle, geometry_type)
        self._mass_function_model = mass_function_model
        self._mlow, self._mhigh = mlow, mhigh
        self._two_halo_term = two_halo_term

        dz_interp = 0.2
        self.mass_function_interpolated = MassFunctionInterp(np.log10(mlow), np.log10(mhigh),
                                                             self.dN_dMdV_comoving, zsource, dz_interp)

        self.log_mass_centers = self.mass_function_interpolated.log_mass_centers
        self.log_mstep = self.mass_function_interpolated.log_mstep

    def norm_at_z_density(self, log_mass_scale, z):

        norm = self.mass_function_interpolated.norm(log_mass_scale, z)

        return norm

    def plaw_index_z(self, log_mass_scale, z):

        index = self.mass_function_interpolated.slope(log_mass_scale, z)

        return index

    def norm_at_z(self, log_mass_scale, z, delta_z):

        norm_dV = self.norm_at_z_density(log_mass_scale, z)

        dV = self.geometry.volume_element_comoving(z, delta_z)

        return norm_dV * dV

    def two_halo_boost(self, M_halo, z, rmin=0.5, rmax=10):

        boost = 1 + 2 * self.integrate_two_halo(M_halo, z, rmin=rmin, rmax=rmax) / (rmax - rmin)
        return boost

    def norm_at_z_biased(self, log_mass_scale, z, delta_z, M_halo, rmin = 0.5, rmax = 10):

        if self._two_halo_term:

            # factor of 2 for symmetry
            boost = self.two_halo_boost(M_halo, z, rmin, rmax)

            return boost * self.norm_at_z(log_mass_scale, z, delta_z)
        else:
            return self.norm_at_z(z, delta_z)

    def integrate_two_halo(self, m200, z, rmin = 0.5, rmax = 10):

        def _integrand(x):
            return self.twohaloterm(x, m200, z)

        boost = quad(_integrand, rmin, rmax)[0]

        return boost

    def twohaloterm(self, r, M, z, mdef='200c'):

        h = self._cosmo.h
        M_h = M * h
        r_h = r * h

        rho_2h = twoHaloTerm(r_h, M_h, z, mdef=mdef) * self._cosmo._colossus_cosmo.rho_m(z) ** -1

        return rho_2h * h ** -2

    def dN_dMdV_comoving(self, M, z):

        """
        :param M: m (in physical units, no little h)
        :param z: redshift
        :return: differential number per unit mass per cubic Mpc (physical)
        [N * M_sun ^ -1 * Mpc ^ -3]
        """

        h = self._cosmo.h

        M_h = M*h

        return h ** 4 * massFunction(M_h, z, q_out='dndlnM') * M_h ** -1

    def dN_comoving_deltaFunc(self, M, z, delta_z, component_fraction):

        """

        :param z: redshift
        :param component_fraction: density parameter; fraction of the matter density (not fraction of critical density!)
        :return: the number of objects of mass M * Mpc^-3
        """

        #a_z = 1/(1+z)
        dN_dV = component_fraction * self._cosmo.rho_dark_matter_crit(z)/M

        return dN_dV * self.geometry.volume_element_comoving(z, delta_z)

    def integrate_mass_function(self, z, delta_z, mlow, mhigh, log_m_break, break_index, break_scale, n=1,
                                norm_scale = 1):

        mbin = self.log_mstep
        logm_centers = np.arange(np.log10(mlow), np.log10(mhigh), mbin) + mbin/2
        if logm_centers[-1]>np.log10(mhigh):
            logm_centers = logm_centers[0:-1]
        moment = 0

        for mi in logm_centers:

            norm = self.norm_at_z(mi, z, delta_z)
            plaw_index = self.plaw_index_z(mi, z)
            moment += self.integrate_power_law(norm_scale * norm, 10**(mi-mbin/2), 10**(mi+mbin/2), log_m_break, n, plaw_index,
                                          break_index=break_index, break_scale=break_scale)

        return moment

    def integrate_power_law(self, norm, m_low, m_high, log_m_break, n, plaw_index, break_index=0, break_scale=1):

        def _integrand(m, m_break, plaw_index, n):

            return norm * m ** (n + plaw_index) * (1 + break_scale * m_break / m) ** break_index

        moment = quad(_integrand, m_low, m_high, args=(10**log_m_break, plaw_index, n))[0]

        return moment

    def cylinder_volume(self, cylinder_diameter_kpc, z_max,
                           z_min=0):

        dz = 0.01
        zsteps = np.arange(z_min+dz, z_max+dz, dz)
        volume = 0
        for zi in zsteps:

            dr = self.geometry._delta_R_comoving(zi, dz)
            radius = 0.5 * cylinder_diameter_kpc * 0.001
            dv_comoving = np.pi*radius**2 * dr
            volume += dv_comoving

        return volume

