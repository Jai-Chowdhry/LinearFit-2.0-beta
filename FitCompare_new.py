import numpy as np
import matplotlib.pyplot as mpl
import numpy.ma as ma

ylabel1 = ('ATS3') #Fixme: this is certainly not ideal
ylabel2 = ('CO$_2$')

t = np.loadtxt('ATS3.txt')
c = np.loadtxt('CO2_3.txt')
b = np.loadtxt('CH4_WAIS.txt')
acid = np.loadtxt('wais_acid.txt')

x_data = t[:, 0]
y_data = t[:, 1]

x_data2 = c[:, 0]
y_data2 = c[:, 1]


N=9

fitstore = np.fromfile('ATS3-9pts_accepted.txt', dtype=float, count=-1, sep=' ')
fitstore = np.reshape(fitstore, (len(fitstore) / (2*N-2), 2*N-2)) #[::10,:]
fitstore2 = np.fromfile('CO2_3-9pts_accepted.txt', dtype=float, count=-1, sep=' ')
fitstore2 = np.reshape(fitstore2, (len(fitstore2) / (2*N-2), 2*N-2)) #[::10,:]

points = np.size(fitstore2[1,:])

fig, (a1,ax6) = mpl.subplots(nrows=2,ncols=1,gridspec_kw = {'height_ratios':[4, 1]},figsize=(10,7))

#a3 = a.twinx()
#a1 = a.twinx()
a2 = a1.twinx()
ax4 = a2.twinx()
ax5 = a2.twinx()

n = points/2-1

xarray = np.empty((np.shape(fitstore)[0], np.shape(fitstore)[1]/2 + 1))
yarray = np.empty((np.shape(fitstore)[0], np.shape(fitstore)[1]/2 + 1))

xindex = np.arange(np.shape(fitstore[:,0:n])[0])[:,np.newaxis], np.argsort(fitstore[:,0:n])
xarray = fitstore[:,0:n][xindex]
xarray = np.concatenate((np.concatenate((np.ones((len(fitstore),1))*min(x_data), xarray),axis=1), np.ones((len(fitstore),1))*max(x_data)), axis=1)
yindex = xindex
yarray = fitstore[:,n]
yarray=np.expand_dims(yarray,1)
print np.shape(yarray)
print np.shape(fitstore[:,n+1:][yindex])
yarray=np.column_stack((yarray,fitstore[:,n+1:][yindex]))
yarray=np.column_stack((yarray,fitstore[:,n+n]))
print np.shape(yarray)

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


xindex2 = np.arange(np.shape(fitstore2[:,0:n])[0])[:,np.newaxis], np.argsort(fitstore2[:,0:n])
xarray2 = fitstore2[:,0:n][xindex2]
xarray2 = np.concatenate((np.concatenate((np.ones((len(fitstore2),1))*min(x_data2), xarray2),axis=1), np.ones((len(fitstore2),1))*max(x_data2)), axis=1)
yindex2 = xindex2
yarray2 = fitstore2[:,n]
yarray2=np.expand_dims(yarray2,1)
print np.shape(yarray2)
print np.shape(fitstore2[:,n+1:][yindex2])
yarray2=np.column_stack((yarray2,fitstore2[:,n+1:][yindex2]))
yarray2=np.column_stack((yarray2,fitstore2[:,n+n]))
print np.shape(yarray2)


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

ax4.set_yticks([-0.002,0,0.002])
ax4.set_ylim(-0.02,0.003) #(-0.005,0.01)
ax5.set_ylim(-0.003,0.02) #(-0.01,0.005)
ax5.invert_yaxis()

histn, bins, patches = ax5.hist(xarray2downmask.flatten(), bins=nbins2d, normed=True, histtype='stepfilled',
                               color='k', alpha=0.75, label=ylabel2) #alpha=0.5, label=ylabel2)
histnd, binsd, patchesd = ax4.hist(xarray2upmask.flatten(), bins=nbins2, normed=True, histtype='stepfilled',
                               color='k', alpha=0.75, label=ylabel2) #alpha=0.5, label=ylabel2)
histn1, bins1, patches1 = ax5.hist(xarraydownmask.flatten(), bins=nbinsd, normed=True, histtype='stepfilled',
                                color=(178/255., 34/255., 34/255.), alpha = 0.75, label=ylabel1) #(174/255., 199/255., 232/255.), alpha=0.5,
histn1d, bins1d, patches1d = ax4.hist(xarrayupmask.flatten(), bins=nbins, normed=True, histtype='stepfilled',
                                color=(178/255., 34/255., 34/255.), alpha = 0.75, label=ylabel1) #(174/255., 199/255., 232/255.), alpha=0.5,
mpl.legend(frameon=False,fontsize='x-small')

#a.plot(b[:,0],b[:,1], color=(75/255.,0/255.,130/255.), marker='.',label=r'CH$_4$')
a1.plot(t[:,0],t[:,1], color=(178/255., 34/255., 34/255.), marker='.',label=r'ATS3')
a2.plot(c[:,0],c[:,1], color='k',marker='.',label=r'CO$_2$')
#a3.plot(acid[:,0], acid[:,1], color=(51/255.,51/255.,0/255.),marker='', label=r'Acidity') #(220./255., 78./255., 22./255.)
#a3.axhline(2,15000/23000.,23000/23000., linestyle=':', color='gray')

#lns = lns1+lns2
#labs = [l.get_label() for l in lns]


#a.set_xlim(min(c[:,0]),max(c[:,0]))
#a.set_ylim(340,1500)
a1.set_ylim(-12,6)
a2.set_ylim(170,370)
#a3.set_ylim(-40,25)

#a2.axvline(18100,0,1000, color=(178/255., 34/255., 34/255.), linestyle='dotted')
#a2.axvline(17710,0,1000, color=(178/255., 34/255., 34/255.), linestyle='dotted')
#a2.axvline(16150,0,1000, color='k', linestyle='dotted')
#a2.axvline(16070,0,1000, color='k', linestyle='dotted')
#a2.axvline(15900,0,1000, color='k', linestyle='dotted')
#a2.axvline(14640,0,1000, color='k', linestyle='dotted')
#a2.axvline(14420,0,1000, color='k', linestyle='dotted')
#a2.axvline(12900,0,1000, color='k', linestyle='dotted')
#a2.axvline(12660,0,1000, color=(178/255., 34/255., 34/255.), linestyle='dotted')
#a2.axvline(11570,0,1000, color='k', linestyle='dotted')
#a2.axvline(11530,0,1000, color='k', linestyle='dotted')

#a2.axvspan(14650,14310,alpha=0.5,color='b')

#a.set_xlabel('Age (yr BP)')
#a.set_yticks([400,500,600,700])
a1.set_yticks([-6,-4,-2,0,2])
a2.set_yticks([180, 200,220,240,260])
a1.tick_params(top=True,bottom=True,left=True,right=False, labelleft=True, labelright=False)
a2.tick_params(left=False,right=True, labelleft=False, labelright=True)
#a3.set_yticks([2,4,6,8,10])
#ax4.set_yticks([])
#ax4.set_yticks([])
ax5.set_yticks([])
ax4.set_ylabel(r'$\rho$')
#ax5.set_yticks([])
a2.set_ylabel(r'CO$_2$ (ppmv)')
a2.yaxis.label.set_color('k')
a2.yaxis.set_label_coords(1.07,0.35)
#mpl.legend(frameon=False,fontsize='x-small')
#a.set_ylabel(r'CH$_4$ (ppbv)')
#a.yaxis.set_label_coords(-0.07,0.175)
#a.yaxis.label.set_color((75/255.,0/255.,130/255.))
#a.tick_params(axis='y',colors=(75/255.,0/255.,130/255.))
#a3.set_ylabel(r'Acidity $\mu$eq L$^{-1}$')
#a3.yaxis.set_label_coords(1.07,0.72)
#a3.yaxis.label.set_color((51/255.,51/255.,0/255.))
#a3.tick_params(axis='y',colors=(51/255.,51/255.,0/255.))
a1.yaxis.label.set_color((178/255., 34/255., 34/255.))
a1.set_ylabel(r'ATS3 (per mil)')
a1.yaxis.set_label_coords(-0.1,0.65)
a1.tick_params(axis='y',colors=(178/255., 34/255., 34/255.))
a1.set_xlim(8800, 22000)

uncertainty = np.loadtxt('christo_totalerror.txt')
ax6.plot(uncertainty[:,0], uncertainty[:,1], ':', color = 'k')
ax6.set_xlim(8800, 22000)
ax6.set_xlabel('Age (yr BP)')
ax6.set_ylim(0,75)
ax6.set_yticks([0,25,50,75])
ax6.set_ylabel(r'$\sigma_{chron}$ (yr)')

mpl.tight_layout()

mpl.show()
fig.savefig('compare-9pts.svg')#,dpi=300)
