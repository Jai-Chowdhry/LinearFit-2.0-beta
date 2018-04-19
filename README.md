# LinearFit 2.0

LinearFit 2.0 is in beta form. Significant testing, code streamlining, and usability development
(including writing a better Readme file) are still in progress.

Please contact Jai Chowdhry Beeman at jai.beeman@univ-grenoble-alpes.fr for questions about running or development.

LinearFit 2.0 is coded in Python 2.7

The following Python packages are needed to run LinearFit 2.0: 

NumPy
http://www.numpy.org/

SciPy
https://www.scipy.org/

Matplotlib
https://matplotlib.org/

scikits.sparse
https://pypi.org/project/scikit-sparse/

emcee
http://dfm.io/emcee/current/

Three main scripts make up the LinearFit tool: LinearFit.py, FitCompare.py and crosscor.py

LinearFit.py assigns an ensemble of linear fits to a time series.

FitCompare.py plots histograms of the timings of the change points of two series.

Crosscor.py uses the cross-correlation operator to calculate the lead/lag distribution over a given time interval between two series.

For more information on running the scripts, run

python LinearFit.py --help
python FitCompare.py --help
python Crosscor.py --help
