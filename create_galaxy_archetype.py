from desispec.interpolation import resample_flux
import numpy as np
from astropy.io import fits
from desisim.archetypes import ArcheTypes, compute_chi2
import redrock
import fitsio
import os
import numba as nb
import time
from astropy.table import Table

template_dir = '/global/cfs/cdirs/desicollab/users/ioannis/fastspecfit/redrock-templates/data/'

def read_galaxy_template_properties(subtype):
    
    filename = template_dir+f'ftemplates-iron-{subtype}.fits' 
    hdu = fits.open(filename)
    metadata = hdu["METADATA"].data
    flux = hdu["FLUX"].data
    wave = hdu["WAVE"].data
    hdu.close()
    return metadata, flux, wave


@nb.njit(fastmath=True, nopython=True)
def new_compute_chi2_numba(flux):

    """chi2 distance calulator for Setcover problem"""
    
    nspec, npix = flux.shape
    chi2 = np.zeros((nspec, nspec), dtype=np.float64)
    amp = np.zeros((nspec, nspec), dtype=np.float64)

    flux = flux.copy()
    rescale = np.sqrt(npix/np.sum(flux**2,axis=1))
    for i in range(nspec):
        for j in range(npix):
            flux[i, j] *= rescale[i]

    for ii in range(nspec):
        for jj in range(nspec):
            amp1 = 0.0
            for kk in range(npix):
                amp1 += flux[ii, kk]*flux[jj, kk]
            amp1 /= npix
            chi2[ii, jj] = npix*(1.-amp1**2)
            amp[ii, jj] = amp1

    for i in range(nspec):
        for j in range(nspec):
            amp[i, j] *= rescale[i]/rescale[j]
            if i == j:
                chi2[i, j] = 0.0
                amp[i, j] = 1.0
    
    return chi2, amp
        
def generate_synthetic_galaxies(nb=1000, dw=0.1, file_out=None):  

    """ Generate synthetic spectra and run SetCoverPy to construct a set of archetypes 
    
    Parameters:
    ------------------------
        nb (int): number of spectra to be generated (default 1000)
        dw (float); wavelength pixel size (in Ang) (default 0.1)
        file_out (string); filename to save the data (.fits format)

    Output:
    ------------------------
        Returns a fits file containing the archetype wavelengt, fluxes and galaxy properties
    """
    start = time.time()
    if file_out is None:
        print('Provide filename to save the archetypes\n')
        return None
    else:
        print(f'synthetic galaxy spectra will be saved at {file_out}')

    ### Rest - frame wavelength definition
    wave = np.arange(3500./(1.+1.85), 11000.+dw/2., dw)
    
    seed = 1234   # Random seed number 
    data = {}

    ### Galaxy properties
    prop_keys = ['LOGMSTAR', 'LOGSSFR', 'AGE', 'ZZSUN', 'AV', 'VDISP', 'TARGETID']
    units = ['Msun/yr', 'yr^-1', 'Gyr', '', 'mag', 'km/s', '']
    ### Galaxy types
    all_subtypes = ['ELG', 'LRG', 'BGS_ANY']

    for subtype in all_subtypes:
        data[subtype] = {}
        data[subtype]['NB'] = int(1./3.*nb)
    data['BGS_ANY']['NB'] = nb-data['ELG']['NB']-data['LRG']['NB']
    data['ELG']['SEED'] = seed
    data['LRG']['SEED'] = seed+data['ELG']['NB']
    data['BGS_ANY']['SEED'] = seed+data['ELG']['NB']+data['LRG']['NB']

   
    ### Selecting a small sample of synthetic spectra

    for subtype in all_subtypes:
        props, tflux, twave = read_galaxy_template_properties(subtype)
        rand = np.random.RandomState(seed=data[subtype]['SEED'])
        inds = rand.choice(tflux.shape[0], size=data[subtype]['NB'],replace=False)
        data[subtype]['FLUX']=tflux[inds]
        data[subtype]['WAVE']=twave
        logssfr = np.log10(props['SFR'][inds])-props['LOGMSTAR'][inds]
        data[subtype]['LOGSSFR']=logssfr
        data[subtype]['TEMPLATEID'] = inds
        for ppkey in prop_keys:
            if ppkey not in ['LOGSSFR']:
                data[subtype][ppkey] = props[ppkey][inds]
    
    ### Combining all galaxies
    nTot = np.sum([ data[k]['NB'] for k in list(data.keys()) ])
    
    ### On same grid
    i = 0
    flux = np.zeros((nTot, wave.size))
    subtype = np.array(['']*nTot, dtype='U32')
    templateid = np.array(['']*nTot, dtype='U32')
    for k in list(data.keys()):
        for j in range(data[k]['NB']):
            subtype[i] = k
            templateid[i] = data[k]['TEMPLATEID'][j]
            flux[i] = resample_flux(wave, data[k]['WAVE'], data[k]['FLUX'][j])
            i += 1
    flux /= np.median(flux,axis=1)[:,None]
    
    properties = {'FLUX':flux, 'SUBTYPE':subtype, 'TEMPLATEID':templateid}
    final_keys = []
    for kk in range(len(prop_keys)):
        key = prop_keys[kk]
        if key not in ['TARGETID', 'ZZSUN']:
            key1 = key+'(%s)'%(units[kk])
        else:
            key1 = key
        final_keys.append(key1)
        properties[key1] = np.concatenate([data[subtype][key] for subtype in all_subtypes])
    
    ## save generated synthetic spectra
    
    header = [
    {'name':'CRVAL1','value':wave[0],'comment':'Rest-Frame starting wavelength [Angstroms]'},
    {'name':'CDELT1','value':dw,'comment':'Rest-Frame wavelength step'},
    {'name':'LOGLAM','value':False,'comment':'Wavelength in log lambda'},
    {'name':'RRTYPE','value':'GALAXY','comment':'Spectral type'},
    {'name':'INSPEC','value':os.environ['DESI_BASIS_TEMPLATES'],'comment':'Vector basis'},
    {'name':'RRVER','value':redrock.__version__,'comment':'Redrock version'},
    {'name':'SEED','value':seed,'comment':'Input seed'},
    {'name':'NBINPUT','value':nb,'comment':'Number of generated spectra'}
    ]
    out = fitsio.FITS(file_out,'rw',clobber=True)
    props = [properties[key] for key in ['FLUX', 'SUBTYPE', 'TEMPLATEID']]
    out.write(props,names=['FLUX','SUBTYPE','TEMPLATEID'],extname='SPECTRA', header=header)
    final_props = [properties[key] for key in final_keys]
    out.write(final_props,names=final_keys,extname='PROPERTIES')
    out.close()

    print(f'time took to generate {nb} galaxies: {time.time()-start} [secs]\n')


def generate_galaxy_archetypes(parent_galaxy_filename, chi2_thresh=10**2.5, fileout=None):
    
    """
    parent_galaxy_filename (string); filename of parent synthetic galaxy spectra
    chi2_thresh (float); chi2 square threshold that will be used for setcoverpy run (default 10**2.5)
    fileout (string); filename to save the data (.fits format)

    Output:
    ------------------------
        Returns a fits file containing the archetype wavelength, fluxes and galaxy properties
    
    """
    
    start =time.time()

    if not os.path.exists(parent_galaxy_filename):
        generate_synthetic_galaxies(nb=1000, dw=0.1, file_out=parent_galaxy_filename)
    else:
        print(f'\nsynthetic galaxy spectra exists at {parent_galaxy_filename}, so using to generate archetype')

    print(f'\n ====   Generating archetypes using {chi2_thresh} ====== \n')
    seed = 1234

    h = fits.open(parent_galaxy_filename, memmap=False)
    hdr = h['SPECTRA'].header
    dw = hdr['CDELT1']

    ### Rest - frame wavelength definition
    wave = np.arange(3500./(1.+1.85), 11000.+dw/2., dw)
    
    flux = h['SPECTRA'].data['FLUX'].astype('float64')
    nb = flux.shape[0]

    subtypes = h['SPECTRA'].data['SUBTYPE']
    tempid  = h['SPECTRA'].data['TEMPLATEID']

    properties  = Table(h['PROPERTIES'].data)
    final_keys = list(properties.dtype.names)
    
    #chi2, amp = compute_chi2(flux)
    
    #numba is atleast twice faster than numpy
    chi2, amp = new_compute_chi2_numba(flux)
    chi2 = chi2.astype('float32')
    Arch = ArcheTypes(chi2)
    iarch = Arch.get_archetypes(chi2_thresh=chi2_thresh)
    print('Generated {} archetypes.'.format(iarch.size))

    ### FLux Normalization
    flux /= np.median(flux,axis=1)[:,None]

    ### Saving archetypes
    header = [
    {'name':'CRVAL1','value':wave[0],'comment':'Rest-Frame starting wavelength [Angstroms]'},
    {'name':'CDELT1','value':dw,'comment':'Rest-Frame wavelength step'},
    {'name':'LOGLAM','value':False,'comment':'Wavelength in log lambda'},
    {'name':'RRTYPE','value':'GALAXY','comment':'Spectral type'},
    {'name':'INSPEC','value':os.environ['DESI_BASIS_TEMPLATES'],'comment':'Vector basis'},
    {'name':'RRVER','value':redrock.__version__,'comment':'Redrock version'},
    {'name':'VERSION','value':'new_v1.1','comment':'Archetype version'},
    {'name':'SEED','value':seed,'comment':'Input seed'},
    {'name':'NBINPUT','value':nb,'comment':'Number of generated spectra'},
    {'name':'CHI2THRE','value':chi2_thresh,'comment':'Chi2 Threshold'},
    ]
    out = fitsio.FITS(fileout,'rw',clobber=True)
    
    props = [properties[key][iarch] for key in ['FLUX', 'SUBTYPE', 'TEMPLATEID']]
    final_props = [properties[key][iarch] for key in final_keys]
    out.write(props,names=['ARCHETYPE','SUBTYPE','TEMPLATEID'],extname='ARCHETYPES', header=header)
    out.write(final_props,names=final_keys,extname='PROPERTIES')
    out.close()
    print(f'\n === For chi2 = {chi2_thresh}, N (archetypes) = {iarch.size} ===== \n')
    print(f'time took for chi2 = {chi2_thresh} to generate archetypes: {time.time()-start} [secs]\n')

if __name__=='__main__':
    
    main_dir = f'/global/cfs/cdirs/desi/users/abhijeet/synthetic_spectra'
    parent_galaxy_filename = f'{main_dir}/archetypes/synthetic_galaxies.fits'
    for pow in [2, 2.25, 2.5, 3, 3.25]:
        chi2 = 10**pow
        generate_galaxy_archetypes(parent_galaxy_filename, chi2_thresh=chi2, fileout=main_dir+'/rrarchetype_chi2_%d_galaxy.fits'%(int(chi2)))


