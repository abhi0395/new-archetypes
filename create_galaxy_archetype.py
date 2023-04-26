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
    

def generate_archetype_galaxies(nb, dw, file_out):  
    
    wave = np.arange(3500./(1.+1.85), 11000.+dw/2., dw)
    
    seed = 1234
    data = {}
    prop_keys = ['LOGMSTAR  ', 'LOGSFR  ', 'AV_ISM  ']

    ###
    data['ELG'] = {}
    data['LRG'] = {}
    data['BGS'] = {}
    data['ELG']['NB'] = int(1./3.*nb)
    data['LRG']['NB'] = int(1./3.*nb)
    data['BGS']['NB'] = nb-data['ELG']['NB']-data['LRG']['NB']

    tseed = seed
    data['ELG']['FLUX'], data['ELG']['WAVE'], data['ELG']['META'], data['ELG']['OBJMETA'] = ELG().make_templates(data['ELG']['NB'],restframe=True,nocolorcuts=True,seed=tseed)
    props = read_a_few_properties('elg', keys=prop_keys, inds =  data['ELG']['META']['TEMPLATEID'])
    data['ELG']['LOGSSFR'] = props['LOGSFR  ']-props['LOGMSTAR  ']
    data['ELG']['AV_ISM'] = props['AV_ISM  ']
    data['ELG']['LOGMSTAR'] = props['LOGMSTAR  ']
    data['ELG']['LOGSFR'] = props['LOGSFR  ']
    
    tseed = seed+data['ELG']['NB']
    data['LRG']['FLUX'], data['LRG']['WAVE'], data['LRG']['META'], data['LRG']['OBJMETA'] = LRG().make_templates(data['LRG']['NB'],restframe=True,nocolorcuts=True,seed=tseed)
    props = read_a_few_properties('lrg', keys=prop_keys, inds =  data['LRG']['META']['TEMPLATEID'])
    data['LRG']['LOGSSFR'] = props['LOGSFR  ']-props['LOGMSTAR  ']
    data['LRG']['AV_ISM'] = props['AV_ISM  ']
    data['LRG']['LOGMSTAR'] = props['LOGMSTAR  ']
    data['LRG']['LOGSFR'] = props['LOGSFR  ']

    tseed = seed+data['ELG']['NB']+data['LRG']['NB']
    data['BGS']['FLUX'], data['BGS']['WAVE'], data['BGS']['META'], data['BGS']['OBJMETA'] = BGS().make_templates(data['BGS']['NB'],restframe=True,nocolorcuts=True,seed=tseed)
    props = read_a_few_properties('bgs', keys=prop_keys, inds =  data['BGS']['META']['TEMPLATEID'])
    data['BGS']['LOGSSFR'] = props['LOGSFR  ']-props['LOGMSTAR  ']
    data['BGS']['AV_ISM'] = props['AV_ISM  ']
    data['BGS']['LOGMSTAR'] = props['LOGMSTAR  ']
    data['BGS']['LOGSFR'] = props['LOGSFR  ']
    
    ###
    nTot = np.sum([ data[k]['NB'] for k in list(data.keys()) ])
    
    ### On same grid
    i = 0
    flux = np.zeros((nTot, wave.size))
    subtype = np.array(['']*nTot, dtype='U32')
    for k in list(data.keys()):
        for j in range(data[k]['NB']):
            subtype[i] = k
            flux[i] = resample_flux(wave, data[k]['WAVE'], data[k]['FLUX'][j])
            i += 1
    flux /= np.median(flux,axis=1)[:,None]
    
    properties = {'FLUX':flux, 'SUBTYPE':subtype}
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
    chi2_thresh = 10**2.5
    iarch = Arch.get_archetypes(chi2_thresh=chi2_thresh)
    print('Generated {} archetypes.'.format(iarch.size))

    ###
    sort = resp.argsort()[::-1]
    subtype = (subtype[iarch])[sort]
    flux = (flux[iarch,:])[sort]
    flux /= np.median(flux,axis=1)[:,None]

    ###
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
    props = [properties[key][iarch] for key in final_keys[2:]]
    out.write([properties['FLUX'][iarch], properties['SUBTYPE'][iarch]],names=['ARCHETYPE','SUBTYPE'],extname='ARCHETYPES', header=header)
    out.write(props,names=final_keys[2:],extname='PROPERTIES')
    out.close()
    
    

if __name__=='__main__':
    generate_archetype_galaxies(nb=1000, dw=0.1, file_out=None)

