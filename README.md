# MSMT-CSD_INR
Code base used in *"Implicit Neural Representation of Multi-shell Constrained
Spherical Deconvolution for Continuous Modeling of Diffusion MRI"*

Please cite:
Hendriks, Tom, Anna Vilanova, and Maxime Chamberland. 
"Implicit Neural Representation of Multi-shell Constrained Spherical Deconvolution for Continuous Modeling of Diffusion MRI." 
Imaging Neuroscience (2025).

### Note  
This repository contains a runnable example of the code used for our work mentioned above.
It has been cleaned, and reduced in size to provide a more usable end product.
We have not rigorously tested this repository, so if any issues occur, please let me know.  

### Known issues
- Hardcoded weighting for multi-shell fitting (only correct for b0 + 2 shells of equal size)
- Multi shell only returns a single b0 volume
- Issue where b-vectors need to be flipped before fitting to be properly visualized in MRTrix3.
