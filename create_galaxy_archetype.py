from desisim.templates import ELG, LRG, BGS
from desispec.interpolation import resample_flux
import numpy as np
from astropy.io import fits
from desisim.archetypes import ArcheTypes, compute_chi2
import redrock
import fitsio
import os
import numba as nb
import time

template_dir = '/global/cfs/cdirs/desi/spectro/templates/basis_templates/v3.0/'

def read_a_few_properties(subtype, keys, inds):
    
    if subtype in ['elg', 'lrg']:
        filename = template_dir+'%s_templates_v2.1.fits'%(subtype)
    else:
        filename = template_dir+'%s_templates_v2.2.fits'%(subtype)
        
    hdu = fits.open(filename)
    data = hdu[1].data
    hdu.close()
    props = {}
    for key in keys:
        props[key] = data[key][inds]
    
    return props


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
        
def generate_archetype_galaxies(nb=1000, dw=0.1, chi2_thresh=10**2.5, file_out=None):  

    """ Generate synthetic spectra and run SetCoverPy to construct a set of archetypes 
    Input:
        nb (int): number of spectra to be generated (default 1000)
        dw (float); wavelength pixel size (in Ang) (default 0.1)
        chi2_thresh (float); chi2 square threshold that will be used for setcoverpy run (default 10**2.5)
        file_out (string); filename to save the data (.fits format)

    Output:
        Returns a fits file containing the archetype wavelengt, fluxes and galaxy properties
    """
    
    if file_out is None:
        print('Provide filename to save the archetypes\n')
        return None

    ### Rest - frame wavelength definition
    wave = np.arange(3500./(1.+1.85), 11000.+dw/2., dw)
    
    seed = 1234   # Random seed number 
    data = {}

    ### Galaxy properties
    prop_keys = ['LOGMSTAR  ', 'LOGSFR  ', 'AV_ISM  ']

    ### Galaxy types
    subtype1, subtype2, subtype3 = 'ELG', 'LRG', 'BGS'

    data[subtype1] = {}
    data[subtype2] = {}
    data[subtype3] = {}

    data[subtype1]['NB'] = int(1./3.*nb)
    data[subtype2]['NB'] = int(1./3.*nb)
    data[subtype3]['NB'] = nb-data[subtype1]['NB']-data[subtype2]['NB']

    ### Generating synthetic spectra

    ### For ELGs
    tseed = seed
    data[subtype1]['FLUX'], data[subtype1]['WAVE'], data[subtype1]['META'], data[subtype1]['OBJMETA'] = ELG().make_templates(data[subtype1]['NB'],restframe=True,nocolorcuts=True,seed=tseed)
    props = read_a_few_properties(subtype1.lower(), keys=prop_keys, inds =  data[subtype1]['META']['TEMPLATEID'])
    data[subtype1]['LOGSSFR'] = props['LOGSFR  ']-props['LOGMSTAR  ']
    data[subtype1]['AV_ISM'] = props['AV_ISM  ']
    data[subtype1]['LOGMSTAR'] = props['LOGMSTAR  ']
    data[subtype1]['LOGSFR'] = props['LOGSFR  ']
    data[subtype1]['TEMPLATEID'] = np.array(['%s_%d'%(subtype1, tt) for tt in data[subtype1]['META']['TEMPLATEID']], dtype='U32')
    
    ### For LRGs
    tseed = seed+data[subtype1]['NB']
    data[subtype2]['FLUX'], data[subtype2]['WAVE'], data[subtype2]['META'], data[subtype2]['OBJMETA'] = LRG().make_templates(data[subtype2]['NB'],restframe=True,nocolorcuts=True,seed=tseed)
    props = read_a_few_properties(subtype2.lower(), keys=prop_keys, inds =  data[subtype2]['META']['TEMPLATEID'])
    data[subtype2]['LOGSSFR'] = props['LOGSFR  ']-props['LOGMSTAR  ']
    data[subtype2]['AV_ISM'] = props['AV_ISM  ']
    data[subtype2]['LOGMSTAR'] = props['LOGMSTAR  ']
    data[subtype2]['LOGSFR'] = props['LOGSFR  ']
    data[subtype2]['TEMPLATEID'] = np.array(['%s_%d'%(subtype2, tt) for tt in data[subtype2]['META']['TEMPLATEID']], dtype='U32')

    ### For BGS
    tseed = seed+data[subtype1]['NB']+data[subtype2]['NB']
    data[subtype3]['FLUX'], data[subtype3]['WAVE'], data[subtype3]['META'], data[subtype3]['OBJMETA'] = BGS().make_templates(data[subtype3]['NB'],restframe=True,nocolorcuts=True,seed=tseed)
    props = read_a_few_properties(subtype3.lower(), keys=prop_keys, inds =  data[subtype3]['META']['TEMPLATEID'])
    data[subtype3]['LOGSSFR'] = props['LOGSFR  ']-props['LOGMSTAR  ']
    data[subtype3]['AV_ISM'] = props['AV_ISM  ']
    data[subtype3]['LOGMSTAR'] = props['LOGMSTAR  ']
    data[subtype3]['LOGSFR'] = props['LOGSFR  ']
    data[subtype3]['TEMPLATEID'] = np.array(['%s_%d'%(subtype3, tt) for tt in data[subtype3]['META']['TEMPLATEID']], dtype='U32')
    
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
    save_keys1, save_keys2, save_keys3 = ['LOGSFR', 'LOGSSFR', 'LOGMSTAR', 'AV_ISM'], ['FLUX_R', 'FLUX_G', 'FLUX_Z'], ['VDISP']
    
    units = ['Msun/yr', 'yr^-1', 'Msun', 'mag', 'nanomaggies', 'nanomaggies', 'nanomaggies', 'km/s']
    
    ii = 0
    for key in save_keys1:
        key1 = key+'(%s)'%(units[ii])
        properties[key1] = np.concatenate([data[subtype][key] for subtype in ['ELG', 'LRG', 'BGS']])
        ii=ii+1
    for key in save_keys2:
        key1 = key+'(%s)'%(units[ii])
        properties[key1] = np.concatenate([data[subtype]['META'][key] for subtype in ['ELG', 'LRG', 'BGS']])
        ii=ii+1
    for key in save_keys3:
        key1 = key+'(%s)'%(units[ii])
        properties[key1] = np.concatenate([data[subtype]['OBJMETA'][key] for subtype in ['ELG', 'LRG', 'BGS']])
        ii=ii+1
    
    start =time.time()
    
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
    {'name':'VERSION','value':'3.3','comment':'Archetype version'},
    {'name':'SEED','value':seed,'comment':'Input seed'},
    {'name':'NBINPUT','value':nb,'comment':'Number of generated spectra'},
    {'name':'CHI2THRE','value':chi2_thresh,'comment':'Chi2 Threshold'},
    ]
    out = fitsio.FITS(file_out,'rw',clobber=True)
    
    final_keys = list(properties.keys())
    props = [properties[key][iarch] for key in final_keys[3:]]
    
    out.write([properties['FLUX'][iarch], properties['SUBTYPE'][iarch], properties['TEMPLATEID'][iarch]],names=['ARCHETYPE','SUBTYPE', 'TEMPLATEID'],extname='ARCHETYPES', header=header)
    out.write(props,names=final_keys[3:],extname='PROPERTIES')
    out.close()

if __name__=='__main__':
    
    file_out = './rrarchetype-galaxy.fits'
    generate_archetype_galaxies(file_out=file_out)

