import numpy as np
try:
    import lenstronomy.Util.multi_gauss_expansion as mge
except:
    raise Exception('using the cored TNFW class requires an installation of '
                    'lenstronomy: https://github.com/sibirrer/lenstronomy')

class coreTNFW_MGE(object):

    hybrid = False

    lenstronomy_ID = 'MULTI_GAUSSIAN_KAPPA'

    def __init__(self, lens_cosmo):

        self.lens_cosmo = lens_cosmo

        self._beta_domain = np.arange(0.01, 1.01, 0.01)
        self._tau_domain = np.linspace(1., 40, len(self._beta_domain))

        xvalues = np.logspace(-2., 3, 200)
        amplitudes_list, sigmas_list = [], []

        for bi in self._beta_domain:
            for ti in self._tau_domain:
                rho = self._profileSIDM(xvalues, bi, ti)
                amp_i, sig_i, _ = mge.mge_1d(xvalues, rho, N=50, linspace=False)
                amplitudes_list.append(amp_i)
                sigmas_list.append(sig_i)

        self._amplitudes_list = amplitudes_list
        self._sigmas_list = sigmas_list
        self._length = len(self._beta_domain)

    def params(self, x, y, mass, redshift, concentration, r_trunc_kpc, b, rho_central):


        """

        :param x: x coordinate of halo (arcsec)
        :param y: y coordinate of halo (arcsec)
        :param mass: halo mass
        :param redshift:
        :param concentration: halo concentration defined as r_200 / rs
        :param r_trunc_kpc: truncation radius in kpc
        :param b: core radius in units of Rs
        :param rho_central: central core density units solar mass / kpc ^ 3
        :return:
        """

        Rs_angle, theta_Rs, kappa_central = self.lens_cosmo.nfw_physical2angle_SIDM(
            mass, concentration, redshift, rho_central)

        x, y = np.round(x, 4), np.round(y, 4)

        Rs_angle = np.round(Rs_angle, 10)

        r_trunc = r_trunc_kpc * self.lens_cosmo.cosmo.kpc_per_asec(redshift) ** -1
        tau = r_trunc / Rs_angle

        kwargs = self._interpMGE(1., x, y, b, tau, Rs_angle)

        return kwargs, None

    def _interpMGE(self, rho_central, center_x, center_y, beta, tau, Rs_angle):

        idx_min_beta = np.argmin(abs(beta - self._beta_domain))
        idx_min_tau = np.argmin(abs(tau - self._tau_domain))

        idx = idx_min_beta * self._length + idx_min_tau
        amp, sigma_unitsx = self._amplitudes_list[idx], self._sigmas_list[idx]

        sigma = np.array(sigma_unitsx) * Rs_angle

        return {'amp': amp, 'sigma': sigma, 'scale_factor': rho_central,
                 'center_x': center_x, 'center_y': center_y}

    @staticmethod
    def _profileSIDM(x, beta, tau):

        """
        Cored TNFW profile normalized such that rho(x = 0) = 1

        rho(x) ~ 1 / [ (x ^ 10 + b ^ 10)^(1/10) * (x + 1)^2 * (x^2 + t^2) ]
        :return: rho(x)
        """
        a = 10
        term1 = (x ** a + beta ** a) ** (-1 / a)
        term2 = (x + 1) ** -2
        term3 = tau ** 2 / (x ** 2 + tau ** 2)
        return term1 * term2 * term3 * beta


