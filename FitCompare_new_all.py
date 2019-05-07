import numpy as np
import matplotlib.pyplot as mpl
import numpy.ma as ma

ylabel1 = ('ATS2') #Fixme: this is certainly not ideal
ylabel = (r'WD CO$_2$',r'ATS3',r'$\delta^{18}$O EDML',r'$\delta^{18}$O TD',r'$\delta^{18}$O EDC',r'$\delta^{18}$O DF',r'$\delta^{18}$O WD')
colors=mpl.cm.get_cmap(name='inferno')(np.arange(0,500,40))
t = np.loadtxt('ATS3.txt')
tDF = np.loadtxt('christo_DF_isotopes.txt')
tEDC = np.loadtxt('christo_EDC_isotopes.txt')
tTD = np.loadtxt('christo_TD_isotopes.txt')
tEDML = np.loadtxt('christo_EDML_isotopes.txt')
tWD = np.loadtxt('christo_WD_isotopes.txt')
c = np.loadtxt('CO2_dummy.txt')
b = np.loadtxt('CH4_WAIS.txt')
acid = np.loadtxt('wais_acid.txt')

tDF[:,1]+=54.889846128504765
tEDC[:,1]+=50.538899573875504
tEDML[:,1]+=44.8931783148609
tTD[:,1]+=36.08909036833579
tWD[:,1]+=33.98926771276664

x_data = t[:, 0]
y_data = t[:, 1]

x_data2 = c[:, 0]
y_data2 = c[:, 1]


N=8


fitstore_CO2 = np.fromfile('CO2_3-8pts_accepted.txt', dtype=float, count=-1, sep=' ')
fitstore_CO2 = np.reshape(fitstore_CO2, (len(fitstore_CO2) / (2 * N - 2), 2 * N - 2)) #[::10,:]
fitstore_ATS3 = np.fromfile('ATS3-8pts_accepted.txt', dtype=float, count=-1, sep=' ')
fitstore_ATS3 = np.reshape(fitstore_ATS3, (len(fitstore_ATS3) / (2 * N - 2), 2 * N - 2)) #[::10,:]
fitstore_WD = np.fromfile('christo_WD_isotopes_accepted.txt', dtype=float, count=-1, sep=' ')
fitstore_WD = np.reshape(fitstore_WD, (len(fitstore_WD) / (2 * N - 2), 2 * N - 2)) #[::10,:]
fitstore_EDML = np.fromfile('christo_EDML_isotopes_accepted.txt', dtype=float, count=-1, sep=' ')
fitstore_EDML = np.reshape(fitstore_EDML, (len(fitstore_EDML) / (2 * N - 2), 2 * N - 2)) #[::10,:]
fitstore_TD = np.fromfile('christo_TD_isotopes_accepted.txt', dtype=float, count=-1, sep=' ')
fitstore_TD = np.reshape(fitstore_TD, (len(fitstore_TD) / (2 * N - 2), 2 * N - 2)) #[::10,:]
fitstore_EDC = np.fromfile('christo_EDC_isotopes_accepted.txt', dtype=float, count=-1, sep=' ')
fitstore_EDC = np.reshape(fitstore_EDC, (len(fitstore_EDC) / (2 * N - 2), 2 * N - 2)) #[::10,:]
fitstore_DF = np.fromfile('christo_DF_isotopes_accepted.txt', dtype=float, count=-1, sep=' ')
fitstore_DF = np.reshape(fitstore_DF, (len(fitstore_DF) / (2 * N - 2), 2 * N - 2)) #[::10,:]

#DF_standard = np.mean(DF[:25,1])
#EDC_standard = np.mean(EDC[:18,1])
#EDML_standard = np.mean(EDML[:31,1])
#TD_standard = np.mean(TD[:34,1])
#WD_standard = np.mean(WD[:76,1])

points = np.size(fitstore_CO2[1, :])

fig, a1 = mpl.subplots(figsize=(10,10))
fig.subplots_adjust(right=0.7)
#a3 = a.twinx()
#a1 = a.twinx()
a2 = a1.twinx()
ax1 = a2.twinx()
ax2 = a2.twinx()
ax4 = a2.twinx()
ax5 = a2.twinx()

axEDML = a2.twinx()
axEDML2 = a2.twinx()
axTD = a2.twinx()
axTD2 = a2.twinx()
axEDC = a2.twinx()
axEDC2 = a2.twinx()
axDF = a2.twinx()
axDF2 = a2.twinx()
axWD = a2.twinx()
axWD2 = a2.twinx()

ax4.set_ylim(-0.038, 0.0015)  # (-0.005,0.01)
ax5.set_ylim(-0.0015, 0.038)  # (-0.01,0.005)
ax5.invert_yaxis()

ax1.set_ylim(-0.038, 0.0015)  # (-0.005,0.01)
ax2.set_ylim(-0.0015, 0.038)  # (-0.01,0.005)
ax2.invert_yaxis()


axEDML.set_ylim(-0.037, 0.002)  # (-0.005,0.01)
axEDML2.set_ylim(-0.002, 0.037)  # (-0.01,0.005)
axEDML2.invert_yaxis()

axTD.set_ylim(-0.036, 0.003)  # (-0.005,0.01)
axTD2.set_ylim(-0.003, 0.036)  # (-0.01,0.005)
axTD2.invert_yaxis()

axEDC.set_ylim(-0.035, 0.004)  # (-0.005,0.01)
axEDC2.set_ylim(-0.004, 0.035)  # (-0.01,0.005)
axEDC2.invert_yaxis()

axDF.set_ylim(-0.034, 0.005)  # (-0.005,0.01)
axDF2.set_ylim(-0.005, 0.034)  # (-0.01,0.005)
axDF2.invert_yaxis()

axWD.set_ylim(-0.033, 0.006)  # (-0.005,0.01)
axWD2.set_ylim(-0.006, 0.033)  # (-0.01,0.005)
axWD2.invert_yaxis()


axeslist = [(ax4,ax5), (ax1,ax2), (axEDML, axEDML2), (axTD, axTD2), (axEDC, axEDC2), (axDF, axDF2), (axWD, axWD2)]

for index, fitstore in enumerate((fitstore_CO2, fitstore_ATS3, fitstore_EDML, fitstore_TD, fitstore_EDC, fitstore_DF, fitstore_WD)):
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

    nbins = int(2 * len(xarrayupmask) ** (1. / 3.))
    nbinsd = int(2 * len(xarraydownmask) ** (1. / 3.))

    if index < 1:
        color=colors[index]
    elif index == 1:
        color = (178/255., 34/255., 34/255.)
    else:
        color = colors[index-1]

    histn, bins, patches = axeslist[index][1].hist(xarraydownmask.flatten(), bins=nbinsd, normed=True, histtype='stepfilled',
                               color=color, alpha=0.75, label=ylabel[index]) #alpha=0.5, label=ylabel2)
    histnd, binsd, patchesd = axeslist[index][0].hist(xarrayupmask.flatten(), bins=nbins, normed=True, histtype='stepfilled',
                               color=color, alpha=0.75, label=ylabel[index]) #alpha=0.5, label=ylabel2)
n=0.025
for ax in axeslist[-1:0:-1]:
    ax[0].legend(frameon=False,fontsize='medium',loc=(0.6,0.6+n))
    n+=0.025

axeslist[0][1].legend(frameon=False,fontsize='medium',loc=(0.6,0.6+n))

#a.plot(b[:,0],b[:,1], color=(75/255.,0/255.,130/255.), marker='.',label=r'CH$_4$')
a1.plot(t[:,0],t[:,1], color=(178/255., 34/255., 34/255.), linestyle='-',label=r'ATS2')
a2.plot(c[:,0],c[:,1], color='k',linestyle='-',label=r'CO$_2$')
#a3.plot(acid[:,0], acid[:,1], color=(51/255.,51/255.,0/255.),marker='', label=r'Acidity') #(220./255., 78./255., 22./255.)
#a3.axhline(2,15000/23000.,23000/23000., linestyle=':', color='gray')

#lns = lns1+lns2
#labs = [l.get_label() for l in lns]



#a.set_xlim(min(c[:,0]),max(c[:,0]))
#a.set_ylim(340,1500)
a1.set_ylim(-15,7)
a2.set_ylim(70,320)
#a3.set_ylim(-40,25)

axtempWD = a2.twinx()
axtempDF = a2.twinx()
axtempEDC = a2.twinx()
axtempTD = a2.twinx()
axtempEDML = a2.twinx()

axes = (axtempEDML, axtempTD, axtempEDC, axtempDF, axtempWD)

for index, tx in enumerate((tEDML,tTD,tEDC,tDF,tWD)):
    axes[index].plot(tx[:, 0], tx[:, 1], color=colors[index+1], alpha=1, linestyle='-', linewidth=0.5, label=ylabel[index+1])
    axes[index].tick_params(axis='y', color=colors[index+1])
    axes[index].spines['right'].set_bounds(-8, 0)
    axes[index].spines['right'].set_edgecolor(colors[index+1])
    axes[index].set_yticks([-8,-4,0])

axtempWD.set_ylim(-8.5, 26)
axtempDF.set_ylim(-8.5, 21.5)
axtempEDC.set_ylim(-12, 20)
axtempTD.set_ylim(-13, 14)
axtempEDML.set_ylim(-15, 13)

axtempWD.spines["right"].set_position(("axes", 1))
axtempDF.spines["right"].set_position(("axes", 1.1))
axtempEDC.spines["right"].set_position(("axes", 1.2))
axtempTD.spines["right"].set_position(("axes", 1.3))
axtempEDML.spines["right"].set_position(("axes", 1.4))

for ax in axeslist:
    ax[0].set_yticks([])
    ax[1].set_yticks([])


a1.set_yticks([-8,-6,-4,-2,0,2])
a2.set_yticks([200,220,240,260,280,300])
a1.tick_params(top=True,bottom=True,left=False,right=True, labelleft=False, labelright=True, labeltop=True)
a2.tick_params(left=True,right=False, labelleft=True, labelright=False)
#a3.set_yticks([2,4,6,8,10])

ax4.set_yticks([])
ax5.set_yticks([])

a2.set_ylabel(r'CO$_2$ (ppmv)')
a2.yaxis.label.set_color('k')
a2.yaxis.set_label_coords(-0.15,0.7)
a1.yaxis.label.set_color((178/255., 34/255., 34/255.))
a1.set_ylabel(r'ATS3 (per mil)')
a1.set_xlabel(r'Age (yr BP)')
a1.yaxis.set_label_coords(1.15,0.5)
a1.tick_params(axis='y',colors=(178/255., 34/255., 34/255.))


mpl.show()
fig.savefig('fitcompare.svg')#,dpi=300)
