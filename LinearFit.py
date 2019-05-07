# LinearFit 2.0
# copyright: Jai Chowdhry Beeman, Frederic Parrenin, Lea Gest, 2018
# 
# License: MIT
#
# To run the code, you need python2 with the math, random, [sys/argparse], numpy, scipy, scikits.sparse, emceee, time, os and matplotlib modules
#
# Then, simply run:
# python LinearFit.py data [-opt] [-points] [-init] [-prop]
# where data is your datafile (x and y again in columns)
# Information on other arguments is given below, or run python LinearFit.py --help
# (Optimization method, initial fit and correlation length are optional arguments)

#Some hidden parameters below

mpi_activate=False
walkers = 100
residuals_correlated=True

import sys

sys.path.extend(['','/home/jcb232/nix-tests/python/venv/lib/python2.7/site-packages', '/nix/store/9w2zsb4skzs1j516wl12akywn5f086b6-python2.7-setuptools-19.4/lib/python2.7/site-packages/setuptools-19.4-py2.7.egg', '/nix/store/9w2zsb4skzs1j516wl12akywn5f086b6-python2.7-setuptools-19.4/lib/python2.7/site-packages/setuptools-19.4-py2.7.egg', '/nix/store/rv593nah61ym1yklsgm0hawxn07f6r85-python-2.7.11/lib/python2.7/site-packages', '/nix/store/40k7ssmlsih2bhzdja6xadgwv1jcyvv8-python2.7-numpy-1.10.4/lib/python2.7/site-packages', '/nix/store/d945ibfx9x185xf04b890y4f9g3cbb63-python-2.7.11/lib/python2.7/site-packages', '/nix/store/9w2zsb4skzs1j516wl12akywn5f086b6-python2.7-setuptools-19.4/lib/python2.7/site-packages', '/nix/store/5f2050cfj9r09sadkadq1rxpcqylrvmp-python2.7-scipy-0.17.0/lib/python2.7/site-packages', '/nix/store/s1zhj6w7lgk5y5i6131dcvmpdji0vd56-python2.7-pip-8.0.2/lib/python2.7/site-packages', '/nix/store/ra025z4fajhnnsz631d9k4kx64wn552n-python2.7-matplotlib-1.5.0/lib/python2.7/site-packages', '/nix/store/w7v6i3ckznl4kffdiyf0a3fza3pjik4s-python2.7-cycler-0.9.0/lib/python2.7/site-packages', '/nix/store/n086vp0yfjb9lnlzcsf479s2l934zvyf-python2.7-six-1.10.0/lib/python2.7/site-packages', '/nix/store/imrn17l9hmz29z9c9z66p09swgighb6q-python2.7-dateutil-2.4.2/lib/python2.7/site-packages', '/nix/store/bmkfsfwch6jj7il5nz53wxy7wdvbsj33-python2.7-nose-1.3.7/lib/python2.7/site-packages', '/nix/store/69z9cq3wp14cm1fig61x1k3vxs2nfhky-python2.7-coverage-4.0.1/lib/python2.7/site-packages', '/nix/store/s2wrbk2vndy44wdkn35ffc8zg3q63ykm-python2.7-pyparsing-2.0.1/lib/python2.7/site-packages', '/nix/store/flw17nylvz02iif8i5m5a1p5194qhc3c-python2.7-tornado-4.2.1/lib/python2.7/site-packages', '/nix/store/b0pf625rinj4b9z34rlmzcb6b1s15j8i-python2.7-backports.ssl_match_hostname-3.4.0.2/lib/python2.7/site-packages', '/nix/store/qmjg50bz6i7wh0nfxkj1p78164253ii6-python2.7-certifi-2015.9.6.2/lib/python2.7/site-packages', '/nix/store/4zdaqg59n7c9yg4ibnw3xq3gxv7mfahl-python2.7-pkgconfig-1.1.0/lib/python2.7/site-packages', '/nix/store/mw82kzsx06ry9a45vnn92xs86d8kq4r9-python2.7-mock-1.3.0/lib/python2.7/site-packages', '/nix/store/g9z0rnr68nb0asznw1x0y2g4jplnn5bi-python2.7-funcsigs-0.4/lib/python2.7/site-packages', '/nix/store/2whif46cpv8jv178bl23vyq8d0vm4rr0-python2.7-pbr-1.8.1/lib/python2.7/site-packages', '/nix/store/szwk9q45am71mkwlllc7r447rpdg5hs1-python2.7-pytz-2015.7/lib/python2.7/site-packages', '/nix/store/vzsfdbc3dnzrdqywz0mvvi06f7f1i92q-python2.7-virtualenv-13.1.2/lib/python2.7/site-packages', '/nix/store/9d8lbpa8lvla8s4q40pijg1ny7gya3qm-python-readline-2.7.11/lib/python2.7/site-packages', '/nix/store/a01d3bvj4nj8pkjdr6jq7wvpqzyi4qz6-python-sqlite3-2.7.11/lib/python2.7/site-packages', '/nix/store/pbsbjh1ma1gp6icbzv9y4vhls1kkgbif-python-curses-2.7.11/lib/python2.7/site-packages', '/nix/store/rv593nah61ym1yklsgm0hawxn07f6r85-python-2.7.11/lib/python27.zip', '/nix/store/rv593nah61ym1yklsgm0hawxn07f6r85-python-2.7.11/lib/python2.7', '/nix/store/rv593nah61ym1yklsgm0hawxn07f6r85-python-2.7.11/lib/python2.7/plat-linux2', '/nix/store/rv593nah61ym1yklsgm0hawxn07f6r85-python-2.7.11/lib/python2.7/lib-tk', '/nix/store/rv593nah61ym1yklsgm0hawxn07f6r85-python-2.7.11/lib/python2.7/lib-old', '/nix/store/rv593nah61ym1yklsgm0hawxn07f6r85-python-2.7.11/lib/python2.7/lib-dynload', '/home/jcb232/.local/lib/python2.7/site-packages'])


import math as m
import argparse
import numpy as np
from scipy.optimize import *
from schwimmbad import MPIPool
from multiprocessing import Pool
import matplotlib.pyplot as mpl
import scipy.linalg
import scipy.sparse.linalg
from sksparse.cholmod import cholesky as cholesky_sparse
import time
import os
import emcee
import scipy.stats

parser = argparse.ArgumentParser(description='LinearFit')
parser.add_argument('file_name', metavar='file_name', type=str, help='Time series file name')
parser.add_argument('-opt', help='Optimization Method', choices=['leastsq', 'emcee'], default='emcee')
parser.add_argument('-points', help='Number of points, required for MC, emcee methods.', type=int, default=6)
parser.add_argument('-init', help='Number of proposals for initialization monte carlo simulation, required for MC, emcee methods.', type=int,
                    default=10000)
parser.add_argument('-prop', help='Number of proposals for true monte carlo simulation, required for MC, emcee methods.', type=int, default=0)
parser.add_argument('-fit0', metavar='file_name_fit0', type=str,
                    help='Fit initial guess file name. Required for leastsq method')
parser.add_argument('-cor_length',
                    help='User specified correlation length, for linearly interpolated correlation matrix',
                    type=float)  # , default = 0.00001)
arg = parser.parse_args()
file_name, file_name_fit0, opt, cor_length, N, init, prop = arg.file_name, arg.fit0, arg.opt, arg.cor_length, arg.points, arg.init, arg.prop

if opt == 'leastsq':
    if file_name_fit0 is None:
        parser.error('Initial fit file required for leastsq method. Run python LinearFit.py -h for more information.')
if opt == 'MC':
    if N is None:
        parser.error('Number of points required for MC method. Run python LinearFit.py -h for more information.')

readarray = np.loadtxt(file_name)
x = readarray[:, 0]
y = readarray[:, 1]
sigma_model = np.ones(np.size(y)) #Before we impose the covariance matrix, we assume it to be identity
if np.shape(readarray)[1]>2:
    sigma_data=readarray[:,2]
else:
    sigma_data = 0

sigma = np.ones(np.shape(sigma_model)) #Before we impose the covariance matrix, we assume it to be identity. (So measurement sigma doesn't affect model estimate). Is this where instability comes from?

if opt == 'leastsq':
    readarray = np.loadtxt(file_name_fit0)
    xfit0 = readarray[:, 0]
    yfit0 = readarray[:, 1]
    N = np.size(yfit0)

    mpl.plot(xfit0, yfit0, color='b', label='initial fit')

    xfit0 = xfit0[1:-1]
    fit0 = np.concatenate((xfit0, yfit0))

def resix(fitvar):  # Split residuals into two functions so that this one can be used in correlation matrix calculation
    xfitvar = np.concatenate((np.array([min(x)]), fitvar[:N - 2], np.array([max(x)])))
    yfitvar = fitvar[N - 2:]
    for i in range(np.size(xfitvar)-1,1,-1):
        if xfitvar[i] < xfitvar[i-1]:
            xfitvar[i - 1] , xfitvar[i] = xfitvar[i] , xfitvar[i - 1]
            yfitvar[i - 1], yfitvar[i] = yfitvar[i], yfitvar[i - 1]
    resi = np.empty_like(x)
    f = np.empty_like(x)
    for i in range(np.size(x)):
        for j in range(N - 1):
            if (xfitvar[j] <= x[i]) and (x[i] <= xfitvar[j + 1]):
                f[i] = yfitvar[j] + (yfitvar[j + 1] - yfitvar[j]) * (x[i] - xfitvar[j]) / (xfitvar[j + 1] - xfitvar[j])
                resi[i] = (f[i] - y[i])
    return resi

cor_matrix = np.abs(np.ones((np.size(x), np.size(x))) * x - np.transpose(np.ones((np.size(x), np.size(x))) * x))

if cor_length is None:
    x_offset = np.min(np.abs(np.diff(x)))
    x_resample = np.arange(np.min(x), np.max(x), x_offset)
    dummy_matrix = np.abs(np.ones((np.size(x_resample), np.size(x_resample))) * x_resample - np.transpose(np.ones((np.size(x_resample), np.size(x_resample))) * x_resample))  # FIXME: This line is sub-optimal. We could calculate a single vector rather than a matrix


def fct_autocorr_4(bestfit, sigmas): #Fixme build function based on least squares estimate of tau/a for R1 process, i.e. Mudelsee 2002, and robust estimator of Chang & Politis 2016

    def func_exp(a, ti, ti0):
        return a**(ti-ti0)

    def cost_exp(a, yi, yi0, ti, ti0):
        resid = (yi - (yi0*(func_exp(a, ti, ti0))))**2
        return np.sum(resid)

    resids = resix(bestfit)

    residsi = resids[1:]
    residsi0 = resids[:-1]

    xi = x[1:]
    xi0 = x[:-1]


    print np.shape(xi), np.shape(xi0), np.shape(residsi), np.shape(residsi0)

    result = least_squares(cost_exp, 0.8, args=(residsi, residsi0, xi, xi0), loss='soft_l1')

    rho = result['x'][0]
    print "rho: ", rho


    cor_mat = ((scipy.stats.iqr(resids)/1.349)**2)*(rho**cor_matrix) #Lots of initialization iterations necessary for variance value to be correct. Use IQR as a rough, much more robust estimate.
    cor_mat += np.diag(sigmas)**2

    print "a = ", rho, "for correlation matrix"

    cor_csc = scipy.sparse.csc_matrix(cor_mat)
    cor_chol = cholesky_sparse(cor_csc)
    cor_lu_var = scipy.sparse.linalg.splu(cor_chol.L())

    return cor_lu_var


def residuals(fitvar, cor_lu_var):
    resi = np.asarray(resix(fitvar)) / np.asarray(sigma)
    resi_decor = resi
    return resi_decor

def residuals_leastsq(fitvar):
    resi = resix(fitvar) / np.asarray(sigma)
    resi_decor= cor_lu_piv.solve(resi)
    return resi_decor

def costfct(fitvar):
    cost = -np.sum(np.abs(residuals(fitvar, cor_lu_piv))) / 2
    if opt == 'emcee':
        xfitvar = np.concatenate((np.array([min(x)]), fitvar[:N - 2], np.array([max(x)])))
        for j in range(N - 1):
            if xfitvar[j] > xfitvar[-1]:
                cost = - np.inf
            if xfitvar[j] < xfitvar[0]:
                cost = -np.inf
    return cost

def costfct_l1(fitvar):
    cost = -np.sum(residuals(fitvar, cor_lu_piv) ** 2) / 2
    if opt == 'emcee':
        xfitvar = np.concatenate((np.array([min(x)]), fitvar[:N - 2], np.array([max(x)])))
        for j in range(N - 1):
            if xfitvar[j] > xfitvar[-1]:
                cost = - np.inf
            if xfitvar[j] < xfitvar[0]:
                cost = -np.inf
    return cost

def costfct_null(fitvar):
    return 1.0

def likelihood(fitvar):
    lik = m.exp(costfct(fitvar))
    xfitvar = np.concatenate((np.array([min(x)]), fitvar[:N - 2], np.array([max(x)])))
    for j in range(N - 1):
        if xfitvar[j] > xfitvar[-1]:
            lik = 0
        if xfitvar[j] < xfitvar[0]:
            lik = 0
    return lik


def fct_leastsq(fit):
    cost_zero = costfct(fit)
    fitopt, covar, infodict, mess, ier = leastsq(residuals_leastsq, fit, full_output=1, maxfev=10000)
    mean = np.mean(residuals_leastsq(fitopt))
    covar = covar * np.std(residuals_leastsq(fitopt)) ** 2
    costopt = costfct(fitopt)
    return cost_zero, fitopt, covar, mess, costopt, mean

if opt == 'emcee':

    time0 = time.time()

    pos0 = []

    for i in range(walkers):
        xfit = np.sort(np.random.uniform(min(x), max(x), N - 2))
        yfit = np.append(np.append(y[0], np.interp(xfit, x, y)), y[-1])
        fitref = fit_opt = fit = np.concatenate((xfit, yfit))

        deltaxfit = np.random.uniform(0 - (max(x) - min(x)) / (2 * N), 0 + (max(x) - min(x)) / (2 * N), N - 2)
        deltayfit = np.random.uniform(0 - (max(y) - min(y)) / (2 * N), 0 + (max(y) - min(y)) / (2 * N), N)
        deltafit = np.concatenate((deltaxfit, deltayfit)) * 0.2

        fit = fitref+deltafit
        pos0.append(fit)

    cor_lu_piv = np.identity(np.shape(dummy_matrix)[0]) #Begin with identity matrix
    like0 = likelihood(fit)


    if mpi_activate: #TODO schwimmbad module appears not to work with enumerate.
        if __name__ == "__main__":
            pool = MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            steps = emcee.EnsembleSampler(walkers, np.size(fitref), costfct, pool=pool, a=2) #args=[cor_lu_piv],) #Try with costfct_l1 in case of bad convergence

            print "MPI running. This could take forever, don't wait around..."
            steps.run_mcmc(pos0, init)
    else:
        pool = Pool(processes=8)
        steps = emcee.EnsembleSampler(walkers, np.size(fitref), costfct_l1, a=2, pool=pool)
        steps.run_mcmc(pos0, nsteps=init, progress=True)
        pool.close()

    maxindex = np.argmax(steps.lnprobability)
    fitref = fit_opt = steps.flatchain[maxindex]
    like_opt = likelihood(fit_opt)

    if prop > 0:
        sigma_model = np.sqrt(np.mean(np.asarray(resix(fitref) ** 2)))  # L2 norm #FIXME should this be standard deviation of residuals?
        print "Modeling variance (biased estimator) : ", np.var(resix(fitref))
        print "Modeling variance (unbiased estimator) : ", scipy.stats.iqr(resix(fitref)/1.349)**2
        sigma = np.mean(sigma_data)*np.ones(len(y))

        pos0 = []

        for i in range(walkers):
            pos0.append(steps.chain[i,-1,:])

        if residuals_correlated:
            cor_lu_piv = fct_autocorr_4(fitref,sigma)

            def residuals(fitvar):
                resi = np.asarray(resix(fitvar))
                resi_decor = cor_lu_piv.solve(resi)
                return resi_decor


            def costfct(fitvar):
                cost = -np.sum(residuals(fitvar) ** 2) / 2
                if opt == 'emcee':
                    xfitvar = np.concatenate((np.array([min(x)]), fitvar[:N - 2], np.array([max(x)])))
                    for j in range(N - 1):
                        if xfitvar[j] > xfitvar[-1]:
                            cost = - np.inf
                        if xfitvar[j] < xfitvar[0]:
                            cost = -np.inf
                return cost

        like0 = likelihood(fit)

        if mpi_activate: #TODO schwimmbad module appears not to work with enumerate
            if __name__ == "__main__":
                steps2 = emcee.EnsembleSampler(walkers, np.size(fitref), costfct, pool=pool, a=2)
                print "MPI running. This could take forever, don't wait around..."
                steps2.run_mcmc(pos0, prop)
                print "MPI finished"
                pool.close()
        else:
            pool = Pool(processes=8)
            steps = emcee.EnsembleSampler(walkers, np.size(fitref), costfct, a=2, pool=pool)
            steps.run_mcmc(pos0, nsteps=prop, progress=True)
            pool.close()

        maxindex = np.argmax(steps.lnprobability)
        fitref = steps.flatchain[maxindex]
        xfitref = np.concatenate((np.array([min(x)]), fitref[:N - 2], np.array([max(x)])))
        yfitref = fitref[N - 2:]



    xfit = np.concatenate((np.array([min(x)]), fit_opt[:N - 2], np.array([max(x)])))
    yfit = fit_opt[N - 2:]

    fitstore = steps.flatchain

    sigma_MC = np.std(fitstore, axis=0)
    x_std = np.concatenate((np.array([0]), sigma_MC[:N - 2], np.array([0])))
    y_std = sigma_MC[N - 2:]

    time1 = time.time()
    print('Metropolis-Hastings exited in ' + str(time1 - time0) + ' seconds')

print 'X_i', xfit
print 'Y_i', yfit

file_name = os.path.splitext(os.path.basename(file_name))[0]

if opt == 'emcee':
    output = np.vstack((xfit, x_std, yfit, y_std))
    headers = 'x x_stddev y y_stddev\n'
    output_name = file_name + '_output'
    with open(file_name + '_accepted.txt', 'w') as f:
        np.savetxt(f, fitstore)


with open(output_name + '.txt', 'w') as f:
    f.write(headers)
    np.savetxt(f, np.transpose(output))
print 'Results saved to output.txt'
print 'To end the program, close the figure(s).'

f.close()