# Archetypes for Redrock

Archetypes for [redrock](https://github.com/desihub/redrock): the main redshift fitter for the Dark Energy Spectroscopic Instrument (DESI) survey. These are based on physically motivated theoretical spectra of galaxie, quasars and stars generated with [desihub/desisim](https://github.com/desihub/desisim).

These archetypes are designed for use with redrock’s `--archetypes` option.  
- To generate the galaxy archetypes, use `create_galaxy_archetypes.py`.  
- For QSOs and stars, use the scripts stored in the `bin` directory.  

## Installation

This repository contains large files tracked with **Git LFS**.  
Depending on whether you have LFS installed, please follow the appropriate instructions:

### If you have Git LFS installed
```bash

# Just clone the repo as usual
git clone https://github.com/abhi0395/new-archetypes.git
cd new-archetypes
```

### If you do not have Git LFS
Cloning without LFS will only fetch placeholder files for large objects (not the actual data).  
```bash
git clone https://github.com/abhi0395/new-archetypes.git
cd new-archetypes
```

If later you install `git-lfs`, you can fetch the large files by running:
```bash
git lfs install
git lfs pull
```

## Usage with Redrock

Once cloned, you can run redrock with the archetypes by pointing to the desired file(s):

```bash
rrdesi_mpi --archetypes path/to/archetype-file.fits ...
```

For more details, see the [redrock documentation](https://github.com/desihub/redrock).

## Citation

If you use these archetypes in your work, please cite:

```bibtex
@ARTICLE{2024AJ....168..124A,
       author = {{Anand}, Abhijeet and {Guy}, Julien and {Bailey}, Stephen and {Moustakas}, John and {Aguilar}, J. and {Ahlen}, S. and {Bolton}, A.~S. and {Brodzeller}, A. and {Brooks}, D. and {Claybaugh}, T. and {Cole}, S. and {de la Macorra}, A. and {Dey}, Biprateep and {Fanning}, K. and {Forero-Romero}, J.~E. and {Gazta{\~n}aga}, E. and {Gontcho A Gontcho}, S. and {Gutierrez}, G. and {Honscheid}, K. and {Howlett}, C. and {Juneau}, S. and {Kirkby}, D. and {Kisner}, T. and {Kremin}, A. and {Lambert}, A. and {Landriau}, M. and {Le Guillou}, L. and {Manera}, M. and {Meisner}, A. and {Miquel}, R. and {Mueller}, E. and {Niz}, G. and {Palanque-Delabrouille}, N. and {Percival}, W.~J. and {Poppett}, C. and {Prada}, F. and {Raichoor}, A. and {Rezaie}, M. and {Rossi}, G. and {Sanchez}, E. and {Schlafly}, E.~F. and {Schlegel}, D. and {Schubnell}, M. and {Sprayberry}, D. and {Tarl{\'e}}, G. and {Warner}, C. and {Weaver}, B.~A. and {Zhou}, R. and {Zou}, H.},
        title = "{Archetype-based Redshift Estimation for the Dark Energy Spectroscopic Instrument Survey}",
      journal = {\aj},
     keywords = {Galaxy spectroscopy, Astronomical methods, Redshift surveys, Astronomy software, Astronomy data analysis, 2171, 1043, 1378, 1855, 1858, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
        month = sep,
       volume = {168},
       number = {3},
          eid = {124},
        pages = {124},
          doi = {10.3847/1538-3881/ad60c2},
archivePrefix = {arXiv},
       eprint = {2405.19288},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024AJ....168..124A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Contact

For issues please contact **abhijeetanand2011@gmail.com**

Thanks,  
Abhijeet Anand  
Lawrence Berkeley National Lab  
