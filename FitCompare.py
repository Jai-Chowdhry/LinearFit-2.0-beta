import argparse
import numpy as np
import numpy.ma as ma
#import matplotlib
#matplotlib.use('module://mplh5canvas.backend_h5canvas')
import matplotlib.pyplot as mpl
from matplotlib.widgets import MultiCursor
#import mpld3
#import plotly.tools as tls
#import plotly.offline
#from scipy.interpolate import interp1d
#from matplotlib.offsetbox import AnchoredText
#from scipy.optimize import curve_fit
#from scipy.stats import norm

ylabel1 = (r'ATS2 (K)') #Fixme: this is certainly not ideal
ylabel2 = (r'CO$_2$ (ppmv)')
ylabel3 = (r'CH$_4$ (ppmv)')
#nbins1 = 250  #histogram bins (obsolete)
#nbins2 = 50

color0 = 'k'
color1 = (203/255., 65/255., 84/255.)

#TODO: make compatible with both MC and leastsq methods

parser = argparse.ArgumentParser(description='FitCompare')
parser.add_argument('data_name',metavar='data_name', type=str, help='Data series file name')
parser.add_argument('data_name2',metavar='data_name2', type=str, help='Data series file name 2')
parser.add_argument('--data_name3', metavar='data_name3', type=str, help='Data series file name 3')

arg = parser.parse_args()

data_name = arg.data_name
data_name2 = arg.data_name2
if arg.data_name3:
    data_name3 = arg.data_name3

file_name = data_name.replace(".txt","_output.txt")
file_name2 = data_name2.replace(".txt","_output.txt")
if arg.data_name3:
    file_name3 = data_name3.replace(".txt","_output.txt")

accepted_name = data_name.replace(".txt","_accepted.txt")
accepted_name2 = data_name2.replace(".txt","_accepted.txt")
if arg.data_name3:
    accepted_name3 = data_name3.replace(".txt","_accepted.txt")

readarray = np.loadtxt(data_name)
x_data = readarray[:, 0]
y_data = readarray[:, 1]

readarray2 = np.loadtxt(data_name2)
x_data2 = readarray2[:, 0]
y_data2 = readarray2[:, 1]

if arg.data_name3:
    readarray3 = np.loadtxt(data_name3)
    x_data3 = readarray3[:, 0]
    y_data3 = readarray3[:, 1]

readarray = np.loadtxt(file_name,skiprows=1)
x = readarray[:,0]
x_std = readarray[:,1]
y = readarray[:,2]
y_std = readarray[:,3]

N = len(x)

readarray2 = np.loadtxt(file_name2, skiprows=1)
x2 = readarray2[:,0]
x_std2 = readarray2[:,1]
y2 = readarray2[:,2]
y_std2 = readarray2[:,3]

if arg.data_name3:
    readarray3 = np.loadtxt(file_name3, skiprows=1)
    x3 = readarray3[:,0]
    x_std3 = readarray3[:,1]
    y3 = readarray3[:,2]
    y_std3 = readarray3[:,3]

fitstore = np.fromfile(accepted_name, dtype=float, count=-1, sep=' ')
fitstore = np.reshape(fitstore, (np.size(fitstore) / (2*N), 2*N)) #[::10,:]
fitstore2 = np.fromfile(accepted_name2, dtype=float, count=-1, sep=' ')
fitstore2 = np.reshape(fitstore2, (np.size(fitstore2) / (2*N), 2*N)) #[::10,:]
if arg.data_name3:
    fitstore3 = np.fromfile(accepted_name3, dtype=float, count=-1, sep=' ')
    fitstore3 = np.reshape(fitstore3, (np.size(fitstore3) / (2*N), 2*N)) #[::10,:]

points = np.size(fitstore2[1,:])
print points

# fig0 = mpl.figure()
# for n in range(points/2-1):
#     ax = mpl.subplot(2, points / 4, n + 1)
#     nbins = int(2*len(fitstore[:,n])**(1./3.))
#
#     histo, xhisto, yhisto = np.histogram2d(fitstore[:,n],fitstore[:,n+points/2],bins=nbins)
#     x_bin_sizes = (xhisto[1:] - xhisto[:-1]).reshape((nbins)) #From GitHub user ardn, see adrn/density_contour.py
#     y_bin_sizes = (xhisto[1:] - yhisto[:-1]).reshape((nbins))
#     pdf = (histo / (x_bin_sizes * y_bin_sizes))
#     X, Y = 0.5 * (xhisto[1:] + xhisto[:-1]), 0.5 * (yhisto[1:] + yhisto[:-1])
#     Z=pdf.T
#     cont = ax.contourf(X,Y,Z,cmap='bone_r')
#     #mpl.xlim(x[n+1]-(x_std[n+1]), x[n+1]+(x_std[n+1]))
#     #mpl.ylim(y[n+1]-(y_std[n+1]),y[n+1]+(y_std[n+1]))
#     mpl.xticks(rotation='vertical')
#     ax.set_xlabel('$X_'+str(n+2)+'$')
#     ax.set_ylabel('$Y_' + str(n+2) + '$')
#     mpl.colorbar(cont)
# mpl.tight_layout()
# mpl.show()
# fig0.savefig('hist2d1.png',bbox_inches='tight')
#
# fig1 = mpl.figure()
# for n in range(points/2-1):
#     ax = mpl.subplot(2, points/4,n+1)
#
#     nbins2 = int(2*len(fitstore2[:,n])**(1./3.))
#     histo, xhisto, yhisto = np.histogram2d(fitstore2[:,n],fitstore2[:,n+points/2],bins=nbins2)
#     x_bin_sizes = (xhisto[1:] - xhisto[:-1]).reshape((nbins2)) #From GitHub user ardn, see adrn/density_contour.py
#     y_bin_sizes = (xhisto[1:] - yhisto[:-1]).reshape((nbins2))
#     pdf = (histo / (x_bin_sizes * y_bin_sizes))
#     X, Y = 0.5 * (xhisto[1:] + xhisto[:-1]), 0.5 * (yhisto[1:] + yhisto[:-1])
#     Z=pdf.T
#     cont = ax.contourf(X,Y,Z,cmap='bone_r')
#     #mpl.xlim(x2[n+1]-(3*x_std2[n+1]), x2[n+1]+(3*x_std2[n+1]))
#     #mpl.ylim(y2[n+1]-(3*y_std2[n+1]),y2[n+1]+(3*y_std2[n+1]))
#     mpl.xticks(rotation='vertical')
#     ax.set_xlabel('$X_'+str(n+2)+'$')
#     ax.set_ylabel('$Y_' + str(n+2) + '$')
#     mpl.colorbar(cont)
# mpl.tight_layout()
# mpl.show()
# fig1.savefig('hist2d2.png',bbox_inches='tight')
#
# if arg.data_name3:
#     fig1a = mpl.figure()
#     for n in range(points/2-1):
#         ax = mpl.subplot(2, points/4,n+1)
#
#         nbins3 = int(2*len(fitstore3[:,n])**(1./3.))
#         histo, xhisto, yhisto = np.histogram2d(fitstore3[:,n],fitstore3[:,n+points/2],bins=nbins3)
#         x_bin_sizes = (xhisto[1:] - xhisto[:-1]).reshape((nbins3)) #From GitHub user ardn, see adrn/density_contour.py
#         y_bin_sizes = (xhisto[1:] - yhisto[:-1]).reshape((nbins3))
#         pdf = (histo / (x_bin_sizes * y_bin_sizes))
#         X, Y = 0.5 * (xhisto[1:] + xhisto[:-1]), 0.5 * (yhisto[1:] + yhisto[:-1])
#         Z=pdf.T
#         cont = ax.contourf(X,Y,Z,cmap='bone_r')
#         #mpl.xlim(x2[n+1]-(3*x_std2[n+1]), x2[n+1]+(3*x_std2[n+1]))
#         #mpl.ylim(y2[n+1]-(3*y_std2[n+1]),y2[n+1]+(3*y_std2[n+1]))
#         mpl.xticks(rotation='vertical')
#         ax.set_xlabel('$X_'+str(n+2)+'$')
#         ax.set_ylabel('$Y_' + str(n+2) + '$')
#         mpl.colorbar(cont)
#     mpl.tight_layout()
#     mpl.show()
#     fig1a.savefig('hist2d3.png',bbox_inches='tight')
#
# fig2 = mpl.figure()
# for n in range(points/2-1):
#     ax = mpl.subplot(2, points / 4, n + 1)
#
#     nbins = int(2 * len(fitstore[:, n]) ** (1. / 3.))
#     nbins2 = int(2 * len(fitstore2[:, n]) ** (1. / 3.))
#     if arg.data_name3:
#         nbins3 = int(2 * len(fitstore3[:, n]) ** (1. / 3.))
#
#     histn, bins, patches = ax.hist(fitstore2[:,n],bins=nbins2,normed=True,histtype='stepfilled',color=(65 / 255., 68 / 255., 81 / 255.),alpha=0.5,label=ylabel2)
#     histn1, bins, patches = ax.hist(fitstore[:,n],bins=nbins,normed=True,histtype='stepfilled',color=(0/255., 107/255., 165/255.),alpha=0.5,label=ylabel1)
#     if arg.data_name3:
#         histn3, bins, patches = ax.hist(fitstore3[:, n], bins=nbins, normed=True, histtype='stepfilled',
#                                     color=(0 / 255., 0/ 255.,  0/ 255.), alpha=0.5, label=ylabel3)
#
#     ax.set_xlabel('x$_{}$, yr BP'.format(n+2))
#     mpl.xticks(rotation='vertical')
#     #mpl.xlim(min(x[n + 1] - (5 * x_std[n + 1]),x2[n + 1] - (5 * x_std2[n + 1])), max(x[n + 1] + (5 * x_std[n + 1]),x2[n + 1] + (5 * x_std2[n + 1])))
#     if not arg.data_name3:
#         histn3 = [0, 0]
#     mpl.ylim(0,max(max(histn),max(histn1),max(histn3)))
# mpl.legend(frameon=False,fontsize='x-small')
# mpl.tight_layout()
# mpl.show()
# fig2.savefig('comparison',bbox_inches='tight')
#
# fig2a = mpl.figure()
# ax = mpl.subplot(1, 1, 1)
# n = points/2-1
# histn, bins, patches = ax.hist(fitstore2[:, 0:n].flatten(), bins=nbins2, normed=True, histtype='stepfilled',
#                                color=(65 / 255., 68 / 255., 81 / 255.), alpha=0.5, label=ylabel2)
# histn1, bins, patches = ax.hist(fitstore[:, 0:n].flatten(), bins=nbins, normed=True, histtype='stepfilled',
#                                 color=(0 / 255., 107 / 255., 165 / 255.), alpha=0.5, label=ylabel1)
# mpl.legend(frameon=False,fontsize='x-small')
# mpl.show()
# fig2a.savefig('comparison-flattened',bbox_inches='tight')
#
# fig3 = mpl.figure(figsize=(3*points/2+1,7.5))
# print points
# print y[0]
# print y[-1]
# for n in range(points):
#     if n < points/2-1:
#         ax = mpl.subplot(2,points/2+1,n+2)
#         labelstr = '$X_' + str(n+2) + '$'
#         best = x[n+1]
#     else:
#         ax = mpl.subplot(2,points/2+1,n+3)
#         labelstr = '$Y_{'+str(n+2-points/2)+'}$'
#         best = y[n-points/2+1]
#     nbins = int(2 * len(fitstore[:, n]) ** (1. / 3.))
#     ax.hist(fitstore[:,n], normed=True, bins=nbins, color=(0/255., 107/255., 165/255.), histtype='stepfilled')
#     ax.axvline(best,color='red')
#     mpl.xticks(rotation='vertical')
#     text_label = AnchoredText(labelstr, loc=1, frameon=False)
#     ax.add_artist(text_label)
#
# mpl.tight_layout()
# mpl.figtext(0.05,0.75,ylabel1,fontsize='x-large')
# mpl.show()
# fig3.savefig('hist1d1.png',bbox_inches='tight')
#
# fig4 = mpl.figure(figsize=(3*points/2+1,7.5))
# for n in range(points):
#     if n < points/2-1:
#         ax = mpl.subplot(2,points/2+1,n+2)
#         labelstr = '$X_' + str(n+2) + '$'
#         best = x2[n+1]
#     else:
#         ax = mpl.subplot(2,points/2+1,n+3)
#         labelstr = '$Y_{'+str(n+2-points/2)+'}$'
#         best = y2[n-points/2+1]
#
#     nbins2 = 2 * len(fitstore2[:, n]) ** (1. / 3.)
#     ax.hist(fitstore2[:,n], normed=True, bins=nbins, color=(0/255., 107/255., 165/255.), histtype='stepfilled')
#     ax.axvline(best,color='red')
#     mpl.xticks(rotation='vertical')
#     text_label = AnchoredText(labelstr, loc=1, frameon=False)
#     ax.add_artist(text_label)
#
# mpl.tight_layout()
# mpl.figtext(0.05,0.75,ylabel2,fontsize='x-large')
# mpl.show()
# fig4.savefig('hist1d2.png',bbox_inches='tight')

fig5, (ax, ax6) = mpl.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw = {'height_ratios':[4, 1]}, figsize=(10,5))

if arg.data_name3:

    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax4 = ax.twinx()
    ax5 = ax.twinx()
else:
    ax2 = ax.twinx()
    ax4 = ax.twinx()
    ax5 = ax.twinx()


# ax2.spines["top"].set_visible(False)
#ax2.spines["bottom"].set_visible(False)
#ax4.spines["bottom"].set_visible(False)
#ax.spines["bottom"].set_visible(True)
#ax6.spines["top"].set_visible(False)
# ax4.spines["top"].set_visible(False)
# if arg.data_name3:
#     ax3.spines["top"].set_visible(False)
#     ax3.spines["bottom"].set_visible(False)


#ax.get_yaxis().tick_left()
ax4.tick_params(bottom='on')

if arg.data_name3:
    ax.set_xlim(max(x2.min(),x.min(),x3.min()), min(x2.max(),x.max(),x3.max()))
    ax.set_ylim(min(y_data.min(),y.min())-y_std.max(),max(y_data.max(),y.max())+y_std.max())
else:
    ax.set_xlim(max(x2.min(),x.min()), min(x2.max(),x.max()))
    #ax.set_ylim(min(y_data.min(),y.min())-y_std.max(),max(y_data.max(),y.max())+y_std.max())
    ax.set_ylim(-14,2)
    ax.set_yticks([-12,-10,-8,-6,-4,-2,0,2])

# for number, element in enumerate(fitstore):
#     if number % 100 == 0:
#         xfitn = np.concatenate((np.array([min(x)]), element[:N - 2], np.array([max(x)])))
#         if all(np.diff(x)) != 0:
#             yfitn = element[N - 2:]
#             ax.fill_between(xfitn, yfitn-0.0005, yfitn+0.0005, color=(20/255.,20/255.,20/255.), alpha=0.002)
ax.plot(x_data, y_data, marker='.', linestyle=' ', label='data', color= color1)#(174/255., 199/255., 232/255.))
print np.shape(x)
print np.shape(y)
#ax.fill_between(x, y-0.02, y+0.02, color=(0 / 255., 107 / 255., 165 / 255.))
#ax.errorbar(x,y,xerr=x_std,yerr=y_std, color=(200/255.,82/255.,0/255.))  #Fixme: error bars look nice but do not accurately represent the distribution
#for element in x:
#    ax.axvline(element, color=(0 / 255., 107 / 255., 165 / 255.))
#    ax2.axvline(element, color=(0 / 255., 107 / 255., 165 / 255.))
#    if arg.data_name3:
#        ax3.axvline(element, color=(0 / 255., 107 / 255., 165 / 255.))

if arg.data_name3:
    ax2.set_xlim(max(x2.min(),x.min(),x3.min()), min(x2.max(),x.max(),x3.max()))
    ax2.set_ylim(min(y_data2.min(),y2.min())- y_std2.max(), max(y_data2.max(),y2.max())+ y_std2.max())
else:
    ax2.set_xlim(max(x2.min(), x.min()), min(x2.max(), x.max()))
    ax2.set_ylim(min(y_data2.min(), y2.min()) - y_std2.max(), max(y_data2.max(), y2.max()) + y_std2.max())

# for number, element in enumerate(fitstore2):
#     if number % 100 == 0:
#         xfitn = np.concatenate((np.array([min(x)]), element[:N - 2], np.array([max(x)])))
#         if all(np.diff(x)) != 0:
#             yfitn = element[N - 2:]
#             ax2.fill_between(xfitn, yfitn - 0.0005, yfitn + 0.0005, color=(20 / 255., 20 / 255., 20 / 255.), alpha=0.002)
ax2.plot(x_data2, y_data2, marker='.', linestyle=' ', label='data', color=color0)
#ax2.fill_between(x2, y2 - 0.02, y2 + 0.02, color=(65 / 255., 68 / 255., 81 / 255.))
#ax2.errorbar(x2, y2, xerr=x_std2, yerr=y_std2, color=(0/255., 107/255., 165/255.))  # Fixme: error bars look nice but do not accurately represent the distribution
#for element in x2:
#    ax.axvline(element, color =(65 / 255., 68 / 255., 81 / 255.))
#    ax2.axvline(element, color =(65 / 255., 68 / 255., 81 / 255.))
#    if arg.data_name3:
#        ax3.axvline(element, color=(65 / 255., 68 / 255., 81 / 255.))

if arg.data_name3:
    ax3.set_xlim(max(x2.min(),x.min(),x3.min()), min(x2.max(),x.max(),x3.max()))
    ax3.set_ylim(min(y_data3.min(), y3.min()) - y_std3.max(), max(y_data3.max(), y3.max()) + y_std3.max())
    # for number, element in enumerate(fitstore3):
    #     if number % 100 == 0:
    #         xfitn = np.concatenate((np.array([min(x)]), element[:N - 2], np.array([max(x)])))
    #         if all(np.diff(x)) != 0:
    #             yfitn = element[N - 2:]
    #             ax3.fill_between(xfitn, yfitn - 0.0005, yfitn + 0.0005, color=(20 / 255., 20 / 255., 20 / 255.), alpha=0.002)
    ax3.plot(x_data3, y_data3, marker='.', linestyle=' ', label='data', color=(65 / 255., 68 / 255., 81 / 255.))
    ax3.fill_between(x3, y3 - 0.02, y3 + 0.02, color=(177 / 255., 177 / 255., 177 / 255.))
    # ax2.errorbar(x2, y2, xerr=x_std2, yerr=y_std2, color=(0/255., 107/255., 165/255.))  # Fixme: error bars look nice but do not accurately represent the distribution
    for element in x3:
        ax.axvline(element, color=(177 / 255., 177 / 255., 177 / 255.))
        ax2.axvline(element, color=(177 / 255., 177 / 255., 177 / 255.))
        ax3.axvline(element, color=(177 / 255., 177 / 255., 177 / 255.))

#    ax3.get_yaxis().tick_left()
#    ax3.get_xaxis().tick_bottom()

n = points/2-1

if arg.data_name3:
    nbins3 = int(2 * len(fitstore3[:, n]) ** (1. / 3.))
    ax4.set_xlim(max(x2.min(), x.min(), x3.min()), min(x2.max(), x.max(), x3.max()))
else:
    ax4.set_xlim(max(x2.min(), x.min()), min(x2.max(), x.max()))


xarray = np.concatenate((np.concatenate((np.ones((len(fitstore[:,0:n]),1))*min(x_data), fitstore[:,0:n]),axis=1), np.ones((len(fitstore[:,0:n]),1))*max(x_data)), axis=1)
yarray = fitstore[:,n:]

xarrayleft = xarray[:,:-2]
print np.shape(xarrayleft)
xarraycenter = xarray[:,1:-1]
print np.shape(xarraycenter)
xarrayright = xarray[:,2:]
print np.shape(xarrayright)
yarrayleft = yarray[:,:-2]
print np.shape(yarrayleft)
yarrayright = yarray[:,2:]
print np.shape(yarrayright)
yarraycenter = yarray[:,1:-1]
print np.shape(yarraycenter)

slopea = np.array(yarraycenter-yarrayright)*1000./np.array(xarrayright-xarraycenter)
slopeb = np.array(yarrayleft-yarraycenter)*1000./np.array(xarraycenter-xarrayleft)

xarrayupmask = ma.masked_where(slopea >= slopeb,xarraycenter)
xarraydownmask = ma.masked_where(slopea <= slopeb,xarraycenter)

xarrayupmask = ma.compressed(xarrayupmask)
xarraydownmask = ma.compressed(xarraydownmask)


xarray2 = np.concatenate((np.concatenate((np.ones((len(fitstore2[:,0:n]),1))*min(x_data2), fitstore2[:,0:n]), axis=1), np.ones((len(fitstore2[:,0:n]),1))*max(x_data2)), axis=1)
yarray2 = fitstore2[:,n:]

xarrayleft2 = xarray2[:,:-2]
xarraycenter2 = xarray2[:,1:-1]
xarrayright2 = xarray2[:,2:]
yarrayleft2 = yarray2[:,:-2]
yarrayright2 = yarray2[:,2:]
yarraycenter2 = yarray2[:,1:-1]

slopea2 = np.array(yarraycenter2-yarrayright2)*1000./np.array(xarrayright2-xarraycenter2)
slopeb2 = np.array(yarrayleft2-yarraycenter2)*1000./np.array(xarraycenter2-xarrayleft2)

xarray2upmask = ma.masked_where(slopea2 >= slopeb2, np.array(xarraycenter2))
xarray2downmask = ma.masked_where(slopea2 <= slopeb2, np.array(xarraycenter2))

xarray2upmask = ma.compressed(xarray2upmask)
xarray2downmask = ma.compressed(xarray2downmask)

nbins = int(2 * len(xarrayupmask) ** (1. / 3.))
nbinsd = int(2 * len(xarraydownmask) ** (1. / 3.))
nbins2 = int(2 * len(xarray2upmask) ** (1. / 3.))
nbins2d = int(2 * len(xarray2downmask) ** (1. / 3.))

ax4.set_ylim(-0.0024,0.0024) #(-0.005,0.01)
ax5.set_ylim(-0.0024,0.0024) #(-0.01,0.005)
ax5.invert_yaxis()

histn, bins, patches = ax5.hist(xarray2downmask.flatten(), bins=nbins2d, normed=True, histtype='stepfilled',
                               color=color0, alpha=0.75, label=ylabel2) #alpha=0.5, label=ylabel2)
histnd, binsd, patchesd = ax4.hist(xarray2upmask.flatten(), bins=nbins2, normed=True, histtype='stepfilled',
                               color=color0, alpha=0.75, label=ylabel2) #alpha=0.5, label=ylabel2)
histn1, bins1, patches1 = ax5.hist(xarraydownmask.flatten(), bins=nbinsd, normed=True, histtype='stepfilled',
                                color=color1, alpha = 0.5, label=ylabel1) #(174/255., 199/255., 232/255.), alpha=0.5,
histn1d, bins1d, patches1d = ax4.hist(xarrayupmask.flatten(), bins=nbins, normed=True, histtype='stepfilled',
                                color=color1, alpha = 0.5, label=ylabel1) #(174/255., 199/255., 232/255.), alpha=0.5,
mpl.legend(frameon=False,fontsize='x-small')

#ax.set_yticks([])
#ax2.set_yticks([])
ax4.set_yticks([])
ax4.set_yticks([])
ax5.set_yticks([])
ax5.set_yticks([])

ax.set_ylabel(ylabel1)
ax2.set_ylabel(ylabel2)
#ax2.set_yticks([180,200,220,240,260,280])


if arg.data_name3:
    ax3.set_ylabel(ylabel3)


uncertainty = np.loadtxt('dating_uncertainty.txt')
ax6.plot(uncertainty[:,0], uncertainty[:,1], ':', color = 'k')
ax6.set_xlim(max(x2.min(), x.min()), min(x2.max(), x.max()))
ax6.set_xlabel('Age (yr BP)')
ax6.set_ylim(0,175)
ax6.set_yticks([0,50,100])
ax6.set_ylabel(r'$\sigma_{chron}$ (yr)')

mpl.tight_layout()

multi = MultiCursor(fig5.canvas, (ax, ax2, ax4, ax5, ax6), color='r', lw=1,
                    horizOn=False, vertOn=True)

#Commented to try mpld3
fig5.savefig('compare.png',dpi=300) #bbox_inches='tight',
mpl.show(fig5)


#ax.patch.set_alpha(0.0)
#ax2.patch.set_alpha(0.0)
#ax4.patch.set_alpha(0.0)
#ax5.patch.set_alpha(0.0)

#mpld3.show()
#mpld3.save_html(fig5,'fig5.html')

#plotly_fig = tls.mpl_to_plotly(fig5)
#plotly.offline.plot(plotly_fig, filename='fig5.html')