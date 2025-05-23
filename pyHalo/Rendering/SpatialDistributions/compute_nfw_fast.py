from scipy.integrate import quad
from scipy.interpolate import interp1d
from time import time
import numpy as np

"""
The functions and classes in this module are used only to compute a set of lookup tables for the NFW profile that I
interpolated in order to sample from a cored NFW profile cumulative distribution function. Sampling from this profile
is accurate, stable, and tested, so I am commenting out these lines so that they do not affect the test coverage of
pyHalo. I am leaving them here in case I ever want to come back and re-compute these CDF tables, or access these routines
for something else.
"""

# def nfw_kappa_integrand(x):
#     return 2 * np.pi * x * nfw_kappa(x)
#
# def nfw_kappa(x):
#     return 2 * (1 - F(x)) / (x ** 2 - 1)
#
# def nfw_rho(x):
#     return 1 / (x * (1 + x) ** 2)
#
# def _F(x):
#     if x < 1:
#         root = np.sqrt(1 - x ** 2)
#         return np.arctanh(root) / root
#     elif x > 1:
#         root = np.sqrt(x ** 2 - 1)
#         return np.arctan(root) / root
#     else:
#         return 1.
#
# def F(x):
#     if isinstance(x, list) or isinstance(x, np.ndarray):
#         out = [_F(xi) for xi in x]
#         return np.array(out)
#     else:
#         return _F(x)
#
# def cdf_projected(c, xmin=1e-8):
#     step = c / 250
#     x2d_range = np.arange(xmin, c + step, step)
#
#     cdf = []
#     for xmax in x2d_range:
#         integral = quad(nfw_kappa_integrand, xmin, xmax)[0]
#         cdf.append(integral)
#
#     cdf = np.array(cdf)
#     return x2d_range, cdf / np.max(cdf)
#
# def nfw_integrand(xz, x2d):
#     return nfw_rho(np.sqrt(x2d ** 2 + xz ** 2))
#
# def cdf_pz_given_x2d(c, x2d):
#     xz_range = np.linspace(0, np.sqrt(c ** 2 - x2d ** 2), 1000)
#     cdf = []
#     for xmax in xz_range:
#         integral = quad(nfw_integrand, 0., xmax, args=(x2d))[0]
#         cdf.append(integral)
#
#     cdf = np.array(cdf)
#     return xz_range, cdf / np.max(cdf)
#
# def invert_cdf(c, cdf_function):
#     domain, cdf = cdf_function(c)
#     interp = interp1d(cdf, domain)
#     return interp
#
# def azimuthal_avg(r2d, rmax, nbins=20):
#     r = []
#     avg = []
#     rbin = np.linspace(0., rmax, nbins)
#     for i in range(0, len(rbin) - 1):
#         dr = rbin[i + 1] - rbin[i]
#         bin_center = 0.5 * (rbin[i + 1] + rbin[i])
#         area = np.pi * (rbin[i + 1] ** 2 - rbin[i] ** 2)
#         inds = np.where(np.absolute(r2d - bin_center) < dr)[0]
#         avg.append(len(inds) / area)
#         r.append(bin_center)
#     r = np.array(r)
#     avg = np.array(avg)
#     return r, avg
#
#
# class ComputeProjected(object):
#
#     def __init__(self):
#
#         c_values = np.arange(c_min, c_max + c_step, c_step)
#
#         domains = None
#         cdfs = None
#
#         for ci in c_values:
#             domain, cdf_proj = cdf_projected(ci)
#             if domains is None:
#                 domains = domain
#                 cdfs = cdf_proj
#             else:
#                 domains = np.vstack((domains, domain))
#                 cdfs = np.vstack((cdfs, cdf_proj))
#
#         np.savetxt('domains_2D.txt', domains)
#         np.savetxt('cdfs_2D.txt', cdfs)
#         np.savetxt('c_values_2D.txt', c_values)
#
# class Compute3D(object):
#
#     def __init__(self):
#
#         x2d_step = 0.05
#
#         x2d_values = np.arange(x2d_min, x2d_max + x2d_step, x2d_step)
#         c_values = np.arange(c_min, c_max + c_step, c_step)
#
#         nx2d = len(x2d_values)
#         nc = len(c_values)
#
#         domains = None
#         cdfs = None
#
#         count = 0
#         N_comb = nx2d * nc
#         print('N combinations:', N_comb)
#         t0 = time()
#
#         c_values_3D, x2d_values_3D = [], []
#
#         for x2di in x2d_values:
#             for ci in c_values:
#                 if count % 50 == 0 and count > 0:
#                     print('progress: ', str(100 * np.round(count / N_comb, 2)) + '%')
#                     tel = time()
#                     rate = count / (tel - t0)
#                     print('time remaining (sec): ', np.round((N_comb - count) / rate))
#                 if x2di >= ci:
#                     continue
#
#                 domain, cdf = cdf_pz_given_x2d(ci, x2di)
#
#                 c_values_3D.append(ci)
#                 x2d_values_3D.append(x2di)
#
#                 if domains is None:
#                     domains = domain
#                     cdfs = cdf
#                 else:
#                     domains = np.vstack((domains, domain))
#                     cdfs = np.vstack((cdfs, cdf))
#
#         np.savetxt('domains_3D.txt', domains)
#         np.savetxt('cdfs_3D.txt', cdfs)
#         np.savetxt('x2d_values_3D.txt', x2d_values_3D)
#         np.savetxt('c_values_3D.txt', c_values_3D)
#
class LookupProjected(object):

    def __init__(self, fpath):

        lookup_table = []

        c_values = np.loadtxt(fpath + 'c_values_2D.txt')
        domains = np.loadtxt(fpath + 'domains_2D.txt')
        cdfs = np.loadtxt(fpath + 'cdfs_2D.txt')

        self.c_min = c_values[0]
        self.c_max = c_values[-1]
        self._c_step = c_values[1] - c_values[0]

        for i in range(0, len(c_values)):

            cdf_proj_inverted = interp1d(cdfs[i,:], domains[i,:])
            lookup_table.append(cdf_proj_inverted)

        self._c_lookup = c_values
        self._lookup_table = lookup_table

    def __call__(self, c):
        assert c >= self.c_min and c <= self.c_max

        dc = np.absolute(c - self._c_lookup) / self._c_step
        idx = np.argmin(dc)
        u = np.random.rand()
        return self._lookup_table[idx](u)

class Lookup3D(object):

    def __init__(self, fpath):

        x2d_values = np.loadtxt(fpath + 'x2d_values_3D.txt')
        c_values = np.loadtxt(fpath + 'c_values_3D.txt')

        domains = np.loadtxt(fpath + 'domains_3D.txt')
        cdfs = np.loadtxt(fpath + 'cdfs_3D.txt')

        self.c_min = c_values[0]
        self.c_max = c_values[-1]
        self.x2d_min = x2d_values[0]
        self.x2d_max = x2d_values[-1]

        lookup_table = []

        for i in range(0, len(c_values)):

            domain, cdf = domains[i,:], cdfs[i,:]
            cdf_inverted = interp1d(cdf, domain)
            lookup_table.append(cdf_inverted)

        for i in range(0, len(x2d_values)):
            if i==0:
                x2d0 = x2d_values[i]
            else:
                if x2d_values[i] != x2d0:
                    x2dstep = x2d_values[i] - x2d0
                    break
        else:
            raise Exception('error occured')

        for i in range(0, len(c_values)):
            if i==0:
                c0 = c_values[i]
            else:
                if c_values[i] != c0:
                    c_step = c_values[i] - c0
                    break
        else:
            raise Exception('error occured')

        self._c_step = c_step
        self._x2d_step = x2dstep

        self._c_lookup = c_values
        self._x2d_lookup = x2d_values
        self._lookup_table = lookup_table

    def __call__(self, x2d_sample_function, c):

        assert c > self.c_min and c < self.c_max

        while True:
            x2d_sample = x2d_sample_function(c)
            if x2d_sample >= self.x2d_min and x2d_sample <= self.x2d_max:
                break

        dc = np.absolute(c - self._c_lookup) / self._c_step
        dx2d = np.absolute(x2d_sample - self._x2d_lookup) / self._x2d_step
        dr = (dc ** 2 + dx2d ** 2) ** 0.5
        idx = np.argmin(dr)

        u = np.random.rand()
        return x2d_sample, self._lookup_table[idx](u)

class FastNFW(object):

    def __init__(self, file_path):

        self.lookup2 = LookupProjected(file_path)
        self.lookup = Lookup3D(file_path)

    def sample(self, c, N):

        x, y, z = [], [], []
        theta = np.random.uniform(0, 2 * np.pi, N)
        sign = np.ones(N)
        u_sign = np.random.rand(N)
        sign[np.where(u_sign > 0.5)] *= -1
        for signi, ti in zip(sign, theta):
            x2d, xz = self.lookup(self.lookup2, c)
            xz *= signi
            x.append(x2d * np.cos(ti))
            y.append(x2d * np.sin(ti))
            z.append(xz)

        return np.array(x), np.array(y), np.array(z)

# c_min, c_max, c_step = 1, 16, 1.
# x2d_min, x2d_max, x2d_step = 0.001, 3., 0.1
# import inspect
# local_path = inspect.getfile(inspect.currentframe())[0:-19]
# print(local_path)
# fastnfw = FastNFW(local_path)

# _ = ComputeProjected()
# _ = Compute3D()
# lookup2d = LookupProjected('./')
# lookup3d = Lookup3D('./')

# fname = 'NFWfast_lookup'
# file = open(fname, 'wb')
# dill.dump(fast_nfw, file)
# file.close()
