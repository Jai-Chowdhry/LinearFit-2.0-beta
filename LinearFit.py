# LinearFit 2.0
# copyright: Jai Chowdhry Beeman, Frédéric Parrenin, Léa Gest, 2018
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

import math as m
import random
import sys
import argparse
import numpy as np
from scipy.optimize import *
import matplotlib.pyplot as mpl
import scipy.linalg
import scipy.sparse.linalg
from scikits.sparse.cholmod import cholesky as cholesky_sparse
import time
import os
import emcee

parser = argparse.ArgumentParser(description='LinearFit')
parser.add_argument('file_name', metavar='file_name', type=str, help='Time series file name')
parser.add_argument('-opt', help='Optimization Method', choices=['leastsq', 'MC', 'emcee'], default='MC')
parser.add_argument('-points', help='Number of points, required for MC, emcee methods.', type=int, default=6)
parser.add_argument('-init', help='Number of proposals for initialization monte carlo simulation, required for MC, emcee methods.', type=int,
                    default=10000)
parser.add_argument('-prop', help='Number of proposals for true monte carlo simulation, required for MC, emcee methods.', type=int, default=10000)
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
# Account for modeling uncertainty
sigma_model = np.std(y)
#sigma_model=0
if np.shape(readarray)[1]>2:
    sigma_data=readarray[:,2]
else:
    sigma_data = 0

sigma = np.sqrt(sigma_data ** 2 + sigma_model ** 2)


print 'To continue close the figure...'
mpl.figure('Data')
mpl.plot(x, y, marker='o', linestyle=' ', label='data')

if opt == 'leastsq':
    readarray = np.loadtxt(file_name_fit0)
    xfit0 = readarray[:, 0]
    yfit0 = readarray[:, 1]
    N = np.size(yfit0)

    mpl.plot(xfit0, yfit0, color='b', label='initial fit')

    xfit0 = xfit0[1:-1]
    fit0 = np.concatenate((xfit0, yfit0))

axes = mpl.gca()
axes.set_xlabel('Time WD2014 (ka BP)')
if min(y) < 100:
    axes.set_ylabel('Temperature variations (Celsius degree)')
else:
    axes.set_ylabel('CO2 variations (ppmv)')
mpl.legend()
mpl.show()


def resix(fitvar):  # Split residuals into two functions so that this one can be used in correlation matrix calculation
    xfitvar = np.concatenate((np.array([min(x)]), fitvar[:N - 2], np.array([max(x)])))
    yfitvar = fitvar[N - 2:]
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
    x_offset = np.diff(x)
    x_resample = np.arange(np.min(x), np.max(x), x_offset.min())
    dummy_matrix = np.abs(np.ones((np.size(x_resample), np.size(x_resample))) * x_resample - np.transpose(np.ones((np.size(x_resample), np.size(x_resample))) * x_resample))  # FIXME: This line is sub-optimal. We could calculate a single vector rather than a matrix

def fct_autocorr(fitvar):
    if cor_length is not None:
        cor_mat = np.interp(cor_matrix, np.array([0., cor_length, max(x) - min(x)]), np.array([1., 0., 0.]))
    else:
        resid_resample = np.interp(x_resample, x, resix(fitvar))
        residcorr = np.correlate(resid_resample, resid_resample, mode='full')  # Calculate residual autocorrelation
        residcorr = residcorr[len(residcorr) // 2:]
        residcorr /= residcorr[0]
        cor_mat = np.interp(cor_matrix, dummy_matrix[0, :], residcorr)

    # Copied from IceChrono
    #self.matrix_csc = scipy.sparse.csc_matrix(self.tuning_correlation[proxy])
    #self.tuning_chol.update({proxy: cholesky_sparse(self.matrix_csc)})
    #self.tuning_lu_piv.update({proxy: scipy.sparse.linalg.splu(self.tuning_chol[proxy].L())})

    cor_csc = scipy.sparse.csc_matrix(cor_mat)
    cor_chol = cholesky_sparse(cor_csc)
    cor_lu_var = scipy.sparse.linalg.splu(cor_chol.L())

    #cor_chol = scipy.linalg.cholesky(cor_mat)
    #cor_lu_var = scipy.linalg.lu_factor(np.transpose(
    #    cor_chol))  # FIXME: we LU factor a triangular matrix. This is suboptimal. We should set lu_piv directly instead.

    return cor_lu_var

def residuals(fitvar, cor_lu_var):
    resi = np.asarray(resix(fitvar)) / np.asarray(sigma)
    #resi_decor = scipy.linalg.lu_solve(cor_lu_var, resi)
    resi_decor = resi
    return resi_decor

def residuals_leastsq(fitvar):
    resi = resix(fitvar) / np.asarray(sigma)
    #resi_decor=scipy.linalg.lu_solve(cor_lu_piv,resi)
    resi_decor= cor_lu_piv.solve(resi)
    return resi_decor

def costfct(fitvar):
    cost = -np.sum(residuals(fitvar, cor_lu_piv) ** 2) / 2
    if opt == 'emcee':
        xfitvar = np.concatenate((np.array([min(x)]), fitvar[:N - 2], np.array([max(x)])))
        for j in range(N - 1):
            if xfitvar[j] > xfitvar[j + 1]:
                cost = -np.inf
            if xfitvar[j] > xfitvar[-1]:
                cost = - np.inf
            if xfitvar[j] < xfitvar[0]:
                cost = -np.inf
    return cost


def likelihood(fitvar):
    lik = m.exp(costfct(fitvar))
    xfitvar = np.concatenate((np.array([min(x)]), fitvar[:N - 2], np.array([max(x)])))
    for j in range(N - 1):
        if xfitvar[j] > xfitvar[j + 1]:
            lik = 0.
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

    for i in range(50):
        xfit = np.sort(np.random.uniform(min(x), max(x), N - 2))
        yfit = np.append(np.append(y[0], np.interp(xfit, x, y)), y[-1])
        fitref = fit_opt = fit = np.concatenate((xfit, yfit))

        deltaxfit = np.random.uniform(0 - (max(x) - min(x)) / (2 * N), 0 + (max(x) - min(x)) / (2 * N), N - 2)
        deltayfit = np.random.uniform(0 - (max(y) - min(y)) / (2 * N), 0 + (max(y) - min(y)) / (2 * N), N)
        deltafit = np.concatenate((deltaxfit, deltayfit)) * 0.2

        fit = fitref+deltafit
        pos0.append(fit)

    cor_lu_piv = fct_autocorr(fit)
    like0 = likelihood(fit)

    mpl.ion()
    mpl.plot(x, y, marker='o', color='blue', linestyle=' ', label='data')

    steps = emcee.EnsembleSampler(50, np.size(fitref), costfct, threads = 8, a = 2) #args = [cor_lu_piv])

    for i, result in enumerate(steps.sample(pos0, iterations=init)):
        if (i) % 100 == 0:
            fitref = result[0][0]
            xfitref = np.concatenate((np.array([min(x)]), fitref[:N - 2], np.array([max(x)])))
            yfitref = fitref[N - 2:]
            mpl.plot(xfitref, yfitref, color='k', alpha=0.1)
            if i == 0:
                text = mpl.text(x.max() - 1000, y.max(), i)
            else:
                text.set_text(i)
            mpl.pause(0.000000001)

    maxindex = np.argmax(steps.lnprobability)
    fitref = fit_opt = steps.flatchain[maxindex]
    like_opt = likelihood(fit_opt)

    sigma_model = np.sqrt(np.mean(np.asarray(resix(fitref) ** 2)))  # L2 norm
    sigma_data = np.sqrt(sigma_data ** 2 + sigma_model ** 2)
    sigma = sigma_model

    pos0 = []

    mpl.close()

    for i in range(50):
        #FIXME!
        # deltaxfit = np.random.uniform(0 - (max(x) - min(x)) / (2 * N), 0 + (max(x) - min(x)) / (2 * N), N - 2)
        # deltayfit = np.random.uniform(0 - (max(y) - min(y)) / (2 * N), 0 + (max(y) - min(y)) / (2 * N), N)
        # deltafit = np.concatenate((deltaxfit, deltayfit)) * 0.2
        # pos0.append(fitref+deltafit)
        # print i
        pos0.append(result[0][i])

    cor_lu_piv = fct_autocorr(fitref)

    def residuals(fitvar):
        resi = np.asarray(resix(fitvar)) / np.asarray(sigma)
        #resi_decor = scipy.linalg.lu_solve(cor_lu_piv, resi)
        resi_decor = cor_lu_piv.solve(resi)
        return resi_decor


    def costfct(fitvar):
        cost = -np.sum(residuals(fitvar) ** 2) / 2
        if opt == 'emcee':
            xfitvar = np.concatenate((np.array([min(x)]), fitvar[:N - 2], np.array([max(x)])))
            for j in range(N - 1):
                if xfitvar[j] > xfitvar[j + 1]:
                    cost = -np.inf
                if xfitvar[j] > xfitvar[-1]:
                    cost = - np.inf
                if xfitvar[j] < xfitvar[0]:
                    cost = -np.inf
        return cost

    like0 = likelihood(fit)

    mpl.ion()
    mpl.plot(x, y, marker='o', color='blue', linestyle=' ', label='data')

    steps = emcee.EnsembleSampler(50, np.size(fitref), costfct, threads=8, a=2) #args=[cor_lu_piv],)

    for i, result in enumerate(steps.sample(pos0, iterations=prop)):
        if (i) % 100 == 0:
            fitref = result[0][0]
            xfitref = np.concatenate((np.array([min(x)]), fitref[:N - 2], np.array([max(x)])))
            yfitref = fitref[N - 2:]
            mpl.plot(xfitref, yfitref, color='k', alpha=0.1)
            if i == 0:
                text = mpl.text(x.max() - 1000, y.max(), i)
            else:
                text.set_text(i)
            mpl.pause(0.000000001)

    xfit = np.concatenate((np.array([min(x)]), fit_opt[:N - 2], np.array([max(x)])))
    yfit = fit_opt[N - 2:]

    fitstore = steps.flatchain

    sigma_MC = np.std(fitstore, axis=0)
    x_std = np.concatenate((np.array([0]), sigma_MC[:N - 2], np.array([0])))
    y_std = sigma_MC[N - 2:]

    time1 = time.time()
    print('Metropolis-Hastings exited in ' + str(time1 - time0) + ' seconds')



if opt == 'MC':  # Maybe need to use variables 'like' instead of 'cost' here, since we use likelihood instead of cost
    print 'Initializing Metropolis Algorithm...coffee break?'
    time0 = time.time()
    likeref = 0.
    xfit = np.sort(np.random.uniform(min(x), max(x), N - 2))
    yfit = np.append(np.append(y[0], np.interp(xfit, x, y)), y[-1])
    fitref = fit_opt = fit = np.concatenate((xfit, yfit))
    cor_length = 1
    cor_lu_piv = fct_autocorr(fitref)
    likeref = like_opt = like0 = likelihood(fit, cor_lu_piv)
    n = 0
    while likeref == 0:   # "Greedy" start
        n += 1
        if n % 10 == 0:
            sys.stdout.flush()
            print "\r",
            print n,
        xfit = np.sort(np.random.uniform(min(x), max(x), N - 2))
        yfit = np.random.uniform(min(y), max(y), N)
        fitref = fit_opt = fit = np.concatenate((xfit, yfit))
        likeref = like_opt = like0 = likelihood(fit, cor_lu_piv)
    accepted = zeros = better = 0
    fitstore = [fit]
    fitbetter = [fit]
    cov = np.identity(2 * N - 2) * 10000
    mpl.ion()
    mpl.plot(x, y, marker='o', color='blue', linestyle=' ', label='data')
    axes = mpl.gca()
    axes.set_xlabel('Time WD2014 (ka BP)')
    if min(y) < 100:
        axes.set_ylabel('Temperature variations (Celsius degree)')
    else:
        axes.set_ylabel('CO2 variations (ppmv)')
    for i in range(init):

        deltaxfit = np.random.uniform(0 - (max(x) - min(x)) / (2 * N), 0 + (max(x) - min(x)) / (2 * N), N - 2)
        deltayfit = np.random.uniform(0 - (max(y) - min(y)) / (2 * N), 0 + (max(y) - min(y)) / (2 * N), N)
        deltafit = np.concatenate((deltaxfit, deltayfit)) * 0.2
        fit = fitref + deltafit
        like = likelihood(fit, cor_lu_piv)

        if like > 0.:
            if like / likeref > random.random():
                fits_old = len(fitstore)
                likeref = like
                fitref = fit
                accepted += 1
                fitstore.append(fit)
            if likeref > like_opt:
                like_opt = likeref
                fit_opt = fitref
                fitbetter.append(fitref)
                better += 1
        else:
            zeros += 1

        if i % 500 == 0:
            xfitref = np.concatenate((np.array([min(x)]), fitref[:N - 2], np.array([max(x)])))
            yfitref = fitref[N - 2:]
            mpl.plot(xfitref, yfitref, color='k', alpha=0.1)
            if i == 0:
                text = mpl.text(x.max() - 1000, y.max(), i)
            else:
                text.set_text(i)
            mpl.pause(0.000000001)

    print('Metropolis-Hastings initialized with ' + str(
        accepted) + ' scenarios accepted, ' + str(better) + ' improvements and ' + str(zeros) + ' zeros')
    mpl.close()

    sigma_model = np.sqrt(np.mean(np.asarray(resix(fitref)**2)))  #L2 norm
    sigma_data = np.sqrt(sigma_data ** 2 + sigma_model**2)
    sigma = sigma_model

    def residuals(fitvar, cor_lu_var):
        resi = np.asarray(resix(fitvar)) / np.asarray(sigma)
        #resi_decor = scipy.linalg.lu_solve(cor_lu_var, resi)
        resi_decor = cor_lu_piv.solve(resi)
        return resi_decor


    def costfct(fitvar, cor_lu_var):
        cost = -np.sum(residuals(fitvar, cor_lu_var) ** 2) / 2
        return cost


    def likelihood(fitvar, cor_lu_var):
        lik = m.exp(costfct(fitvar, cor_lu_var))
        xfitvar = np.concatenate((np.array([min(x)]), fitvar[:N - 2], np.array([max(x)])))
        for j in range(N - 1):
            if xfitvar[j] > xfitvar[j + 1]:
                lik = 0.
        return lik


    fit_good = fit_opt
    cov = np.cov(fitstore, rowvar=0)
    likeref = 0.
    cor_lu_piv = fct_autocorr(fit_opt)
    fitref = fit = fit_good
    likeref = like_opt = like0 = likelihood(fitref, cor_lu_piv)
    while likeref == 0:
        n += 1
        if n % 10 == 0:
            sys.stdout.flush()
            print "\r",
            print n,
        fitref = fit = np.random.multivariate_normal(fit_good, cov)
        likeref = like_opt = like0 = likelihood(fit, cor_lu_piv)
    accepted = zeros = better = 0
    fitstore = [fit]
    fitbetter = [fit]
    fits_old = len(fitstore)

    mpl.ion()
    mpl.plot(x, y, marker='o', color='blue', linestyle=' ', label='data')
    axes = mpl.gca()
    axes.set_xlabel('Time WD2014 (ka BP)')
    if min(y) < 100:
        axes.set_ylabel('Temperature variations (Celsius degree)')
    else:
        axes.set_ylabel('CO2 variations (ppmv)')
    for i in range(prop):

        deltaxfit = np.random.uniform(0 - (max(x) - min(x)) / (2 * N), 0 + (max(x) - min(x)) / (2 * N), N - 2)
        deltayfit = np.random.uniform(0 - (max(y) - min(y)) / (2 * N), 0 + (max(y) - min(y)) / (2 * N), N)
        deltafit = np.concatenate((deltaxfit, deltayfit)) * 0.05
        fit = fitref + deltafit
        like = likelihood(fit, cor_lu_piv)
        if like > 0.:
            if like / likeref > random.random():
                fits_old = len(fitstore)
                likeref = like
                fitref = fit
                accepted += 1
            if likeref > like_opt:
                like_opt = likeref
                fit_opt = fitref
                fitbetter.append(fitref)
                better += 1
        else:
            zeros += 1
        fitstore.append(fitref)

        if i % 500 == 0:
            xfitref = np.concatenate((np.array([min(x)]), fitref[:N - 2], np.array([max(x)])))
            yfitref = fitref[N - 2:]
            mpl.plot(xfitref, yfitref, color='k', alpha=0.1)
            if i == 0:
                text = mpl.text(x.max() - 1000, y.max(), i)
            else:
                text.set_text(i)
            mpl.pause(0.000000001)


    time1 = time.time()
    print('Metropolis-Hastings exited in ' + str(time1 - time0) + ' seconds with ' + str(
        accepted) + ' scenarios accepted, ' + str(better) + ' improvements and ' + str(zeros) + ' zeros')
    xfit = np.concatenate((np.array([min(x)]), fit_opt[:N - 2], np.array([max(x)])))
    yfit = fit_opt[N - 2:]

    sigma_MC = np.std(fitstore, axis=0)
    x_std = np.concatenate((np.array([0]), sigma_MC[:N - 2], np.array([0])))
    y_std = sigma_MC[N - 2:]

# result=basinhopping(costfct, fit0)
# fit_opt=result.x
# cost_opt=result.fun
# message=result.message

# fit_opt,cost_opt=anneal(costfct, fit0)




if opt == 'leastsq':

    cor_lu_piv = fct_autocorr(fit0)
    mean = np.mean(resix(fit0))
    if cor_length is None:
        for step in np.arange(10):
            #fit0 += np.random.rand(len(fit0))*(fit0)*0.001
            cost0, fit_opt, cov, message, cost_opt, mean = fct_leastsq(fit0)
            print mean
            cor_lu_piv = fct_autocorr(fit_opt)
    cost0, fit_opt, cov, message, cost_opt, mean = fct_leastsq(fit0)
    print "cost function " + str(cost0)
    cor_lu_piv = fct_autocorr(fit_opt)



    like0 = likelihood(fit0, cor_lu_piv)
    xfit0 = np.concatenate((np.array([min(x)]), xfit0, np.array([max(x)])))
    xfit = np.concatenate((np.array([min(x)]), fit_opt[:N - 2], np.array([max(x)])))
    yfit = fit_opt[N - 2:]
    xfit_cov = np.empty_like(xfit)
    xfit_cov[0] = 0.
    for i in range(np.size(xfit) - 2):
        xfit_cov[i + 1] = m.sqrt(cov[i, i])
    xfit_cov[-1] = 0.
    yfit_cov = np.empty_like(yfit)
    for i in range(np.size(yfit)):
        yfit_cov[i] = m.sqrt(cov[i + N - 2, i + N - 2])

#print 'Optimal/initial density of probability:  ', like_opt / like0
print 'X_i', xfit
print 'Y_i', yfit

file_name = os.path.splitext(os.path.basename(file_name))[0]
if opt == 'leastsq':
    output = np.vstack((xfit, xfit_cov, yfit, yfit_cov))
    headers = 'x x_stddev y y_stddev\n'
    output_name = "{0}{1}{2}{3}{4}".format(file_name, opt, str('_'), str(N), str('pts_output'))
elif opt == 'MC':
    output = np.vstack((xfit, x_std, yfit, y_std))
    headers = 'x x_stddev y y_stddev\n'
    output_name = file_name + '_output'
    with open(file_name + '_accepted.txt', 'w') as f:
        np.savetxt(f, fitstore)
elif opt == 'emcee':
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

if opt == 'leastsq':
    fig1, ax = mpl.subplots(1, 1)
    mpl.plot(x, y, marker='o', linestyle=' ', label='data')
    axes = mpl.gca()
    axes.set_xlabel('Time WD2014 (yr BP)')
    if min(y) < 100:
        axes.set_ylabel('Temperature variations (Celsius degree)')
    else:
        axes.set_ylabel('CO2 variations (ppmv)')
    mpl.legend()
    mpl.errorbar(xfit, yfit, color='r', label='Least squares fit', xerr=xfit_cov, yerr=yfit_cov)
    roundnb = -int(m.log10(max(x) - min(x))) + 3
    for i in range(1, np.size(xfit) - 1):
        mpl.annotate(str(round(xfit[i], roundnb)) + '+/-' + str(round(xfit_cov[i], roundnb)), xy=(xfit[i], yfit[i]))
    fig1.savefig(output_name + '.png')
    mpl.show(fig1)
def costfct(fitvar, cor_lu_var):
        cost = -np.sum(residuals(fitvar, cor_lu_var) ** 2) / 2
        if opt == 'emcee':
            xfitvar = np.concatenate((np.array([min(x)]), fitvar[:N - 2], np.array([max(x)])))
            for j in range(N - 1):
                if xfitvar[j] > xfitvar[j + 1]:
                    cost = -np.inf
                if xfitvar[j] > xfitvar[-1]:
                    cost = - np.inf
                if xfitvar[j] < xfitvar[0]:
                    cost = -np.inf
        return cost
if opt == 'emcee':
    fig1, ax = mpl.subplots(1, 1)
    mpl.plot(x, y, marker='o', linestyle=' ', label='data')
    axes = mpl.gca()
    axes.set_xlabel('Age (ka)')
    if min(y) < 100:
        axes.set_ylabel('Temperature variations (Celsius degree)')
    else:
        axes.set_ylabel('CO2 variations (ppmv)')
    mpl.legend()
    mpl.plot(xfit, yfit, color='black', marker = ' ', linestyle='-')
    mpl.show(fig1)
    fig1.savefig(output_name + '.png', bbox_inches='tight')

if opt == 'MC':
    fig2 = mpl.figure(figsize=(10,3.75))
    ax = mpl.subplot(111)
    # ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    mpl.xlim(xfit.min(),xfit.max())
    mpl.ylim(y.min()-y_std.mean(),y.max()+y_std.mean())
    for number, element in enumerate(fitstore):
        xfitn = np.concatenate((np.array([min(x)]), element[:N - 2], np.array([max(x)])))
        if all(np.diff(xfit)) != 0:
            yfitn = element[N - 2:]
            ax.fill_between(xfitn, yfitn-0.0005, yfitn+0.0005, color=(20/255.,20/255.,20/255.), alpha=0.002)
    mpl.plot(x, y, marker='.', linestyle=' ', label='data', color=(0/255., 107/255., 165/255.))
    ax.fill_between(xfit, yfit-0.02, yfit+0.02, color=(200/255., 82/255., 0/255.))
    ax.errorbar(xfit,yfit,xerr=x_std,yerr=y_std, color=(200/255.,82/255.,0/255.))  #Fixme: error bars look nice but do not accurately represent the distribution
    axes = mpl.gca()
    axes.set_xlabel('Time WD2014 (yr BP)')
    if min(y) < 100:
        axes.set_ylabel('Temperature variations (Celsius degree)')
    else:
        axes.set_ylabel('CO2 variations (ppmv)')

    fig2.savefig(output_name + '.png',bbox_inches='tight')
    mpl.show(fig2)
