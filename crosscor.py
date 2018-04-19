import argparse
import numpy as np
import pandas
import matplotlib.pyplot as mpl
from scipy.interpolate import interp1d
from matplotlib.offsetbox import AnchoredText
from scipy.optimize import curve_fit
import numpy.ma as ma
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm

ylabel1 = ('$\Delta$ T (K)') #Fixme: this is certainly not ideal
ylabel2 = ('CO$_2$ (ppmv)')
#nbins1 = 250  #histogram bins (obsolete)
#nbins2 = 50
m_ddepth=0 #Should be 0
sigma_ddepth=30 #Max ddepth uncertainty? Half of max ddepth uncertainty??

#TODO: make compatible with both MC and leastsq methods

parser = argparse.ArgumentParser(description='FitCompare')
parser.add_argument('data_name',metavar='data_name', type=str, help='Data series file name')
parser.add_argument('data_name2',metavar='data_name2', type=str, help='Data series file name 2')
parser.add_argument('-concave', type=str, choices=['up','down'], default='up', help='Concavity')
parser.add_argument('--range', type=float, nargs=2, help='X range for histograms in form xmin xmax')


arg = parser.parse_args()

data_name = arg.data_name
data_name2 = arg.data_name2

if hasattr(arg,'range'):
    x_limits = tuple(arg.range)

file_name = data_name.replace(".txt","_output.txt")
file_name2 = data_name2.replace(".txt","_output.txt")

accepted_name = data_name.replace(".txt","_accepted.txt")
accepted_name2 = data_name2.replace(".txt","_accepted.txt")

readarray = np.loadtxt(data_name)
x_data = readarray[:, 0]
y_data = readarray[:, 1]

readarray2 = np.loadtxt(data_name2)
x_data2 = readarray2[:, 0]
y_data2 = readarray2[:, 1]

readarray = np.loadtxt(file_name,skiprows=1)
x = readarray[:,0]
x_std = readarray[:,1]
y = readarray[:,2]
y_std = readarray[:,3]

N = len(x)
n = N-1

readarray2 = np.loadtxt(file_name2, skiprows=1)
x2 = readarray2[:,0]
x_std2 = readarray2[:,1]
y2 = readarray2[:,2]
y_std2 = readarray2[:,3]

fitstore = pandas.read_csv(accepted_name,' ',header=None).values
fitstore2 = pandas.read_csv(accepted_name2,' ',header=None).values

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

if arg.concave == 'up':
    fitstore = xarrayupmask
    fitstore2 = xarray2upmask
else:
    fitstore = xarraydownmask
    fitstore2 = xarray2downmask

#points = np.size(fitstore2[1,:])
#print points


fig2a, ax = mpl.subplots()

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

gaussianlist = []


histo1, xhisto1 = np.histogram(fitstore, range=x_limits, bins='fd', normed=True)
x_bin_sizes1 = (xhisto1[1:] - xhisto1[:-1])#.reshape((len(xhisto1)))
pdf1 = (histo1)  #/x_bin_sizes1)

histo2, xhisto2 = np.histogram(fitstore2, range=x_limits, bins='fd', normed=True)
x_bin_sizes2 = (xhisto2[1:] - xhisto2[:-1])#.reshape((len(xhisto2)))
pdf2 = (histo2)  #/ x_bin_sizes2)

nbins = min(len(xhisto1),len(xhisto2))/4 #Fixme: why do we need to reduce????
print('nbins ',nbins)

reinterp_x = np.arange(min(xhisto1.min(),xhisto2.min()), max(xhisto1.max(),xhisto2.max()), int((max(xhisto1.max(),xhisto2.max())-min(xhisto1.min(),xhisto2.min()))/((nbins))))
finterp1 = interp1d(xhisto1[1:]+np.random.random(len(xhisto1[1:])),pdf1, bounds_error=False, fill_value=0)
finterp2 = interp1d(xhisto2[1:]+np.random.random(len(xhisto2[1:])), pdf2, bounds_error=False, fill_value=0)
pdflag1 = finterp1(reinterp_x)
pdflag2 = finterp2(reinterp_x)
ddepth = np.random.normal(loc=m_ddepth,scale=sigma_ddepth,size=np.size(pdflag1))

lag = np.correlate(pdflag1,pdflag2,mode='full') #Todo: estimate a normal distribution here.
#lag = np.convolve(lag,ddepth) #convolve with ddepth uncertainty
#(mean, sigma) = norm(lag) #Todo: how does scipy.stats.norm function work??
a = int((max(xhisto1.max(),xhisto2.max())-min(xhisto1.min(),xhisto2.min()))/nbins)
t = np.concatenate((np.arange(-len(lag)/2,0),np.arange(len(lag)/2)))*a
lagarray = np.array(lag)

ax.bar(t,lagarray,width=a,color=(47./255., 47./255., 47./255.),alpha=1,linewidth=0)
#ax.set_xlabel('Lag x$_{}$'.format(n+2))
mpl.xticks(rotation='vertical',fontsize=20)
mpl.yticks(fontsize=20)

coeff, var_matrix = curve_fit(gauss, t, lag, p0=[1.,0.,100.])
gaussianlist.append([coeff[1],coeff[2]])
print gaussianlist
#mpl.xlim(coeff[1]-5*np.abs(coeff[2]),coeff[1]+5*np.abs(coeff[2]))

ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(5))
ax.locator_params(axis=y,nticks=4)
mpl.grid(color='gray',linewidth=4)
mpl.tight_layout()
mpl.show()
fig2a.savefig('correlation',bbox_inches='tight')
#np.savetxt('lags',np.array(gaussianlist))
