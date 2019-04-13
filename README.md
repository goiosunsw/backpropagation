# Backpropagation

Folder includes:

* Example jupyter notebooks:
  
  * [Synchronised Backpropagation.ipynb](./Synchronised%20Backpropagation.ipynb) exemplifying how to adjust a fit of the vocal tract using data from the recording
  * [Async Backpropagation.ipynb](./Async%20Backpropagation.ipynb) exemplifying how to do backpropagation using a vocal tract measurement fitted prior to the analysis. Fitted data can be generated with [Synchronised Backpropagation.ipynb](Synchronised%20Backpropagation.ipynb). This is not working as it should...
  * [Async Backpropagation-oldRec.ipynb](./Async%20Backpropagation-oldRec.ipynb) using previous data where the pressure and flow calibration work
* optimiser helper functions in [multi-optimiser.py](./multi-optimiser.py)
* Annotation data for the audacity project of recent recordings (12/04/2019)
* Example fitted data for vocal tracts

## Also requires:

* [PyPeVoc](https://github.com/goiosunsw/PyPeVoc) for estimation of fundamental frequency and harmonic amplitudes and phases
* [ImpedancePython](https://github.com/goiosunsw/ImpedancePython) for reading impedance files and parameters and calculating impedances from geometry
* [audacity.py](https://github.com/goiosunsw/audacity.py) to read audacity projects
* [multi_optimiser.py](./multi-optimiser.py), auxiliary functions for optimisation. Should be in this repository
