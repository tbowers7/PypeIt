""" Module for fluxing routines

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst

"""
import os
import glob
from pkg_resources import resource_filename

from IPython import embed

import numpy as np

from scipy import interpolate

from matplotlib import pyplot as plt

from astropy import units
from astropy import constants
from astropy import coordinates
from astropy import table
from astropy.io import ascii
from astropy import stats
from astropy import units as u
from astropy import constants as const

from linetools.spectra.xspectrum1d import XSpectrum1D

from pypeit import msgs
from pypeit import utils
from pypeit import bspline
from pypeit import io
from pypeit.wavemodel import conv2res
from pypeit.core.wavecal import wvutils
from pypeit.core import fitting
from pypeit.core import telluric


# TODO: Put these in the relevant functions
TINY = 1e-15
SN2_MAX = (20.0) ** 2
PYPEIT_FLUX_SCALE = 1e-17


def zp_unit_const():
    """
    This constant defines the units for the spectroscopic zeropoint. See the doc/dev/fluxing.rst doc for more information.
    Returns
    -------

    """
    return -2.5*np.log10(((u.angstrom**2/const.c)*(PYPEIT_FLUX_SCALE*u.erg/u.s/u.cm**2/u.angstrom)).to('Jy')/(3631 * u.Jy)).value


# This function is defined to convert AB magnitudes to cgs unit erg cm^-2 s^-1 A^-1
def mAB_to_cgs(mAB,wvl):
    return 10**((-48.6-mAB)/2.5)*3*10**18/wvl**2


def blackbody_func(a, teff):
    """
    Generate a blackbody spectrum based on the normalisation and effective temperature.
    See Suzuki & Fukugita, 2018, AJ, 156, 219:
    https://ui.adsabs.harvard.edu/abs/2018AJ....156..219S/abstract

    Args:
        a (float):
            flux normalisation factor
        teff (float):
            Effective temperature of the blackbody

    Returns:
        waves : `numpy.ndarray`_ of the wavelengths
        flam : `numpy.ndarray`_ flux in units of erg/s/cm^2/A
    """
    waves = np.arange(3000.0, 25000.0, 0.1) * u.AA
    # Setup the units
    teff *= u.K
    a *= 1.0E-23
    # Calculate the function
    flam = ((a*2*const.h*const.c**2)/waves**5)/(np.exp((const.h*const.c/(waves*const.k_B*teff)).to(u.m/u.m).value)-1.0)
    flam = flam.to(u.erg / u.s / u.cm ** 2 / u.AA).value / PYPEIT_FLUX_SCALE
    return waves.value, flam


# Define this global variable to avoid constantly recomputing, which could be costly in the telluric optimization routines.
# It has a value of ZP_UNIT_CONST = 40.092117379602044
ZP_UNIT_CONST = zp_unit_const()


def find_standard_file(ra, dec, toler=20.*units.arcmin, check=False):
    """
    Find a match for the input file to one of the archived
    standard star files (hopefully).  Priority is by order of search.

    Args:
        ra (float):
            Object right-ascension in decimal deg
        dec (float):
            Object declination in decimal deg
        toler (:class:`astropy.units.quantity.Quantity`, optional):
            Tolerance on matching archived standards to input.  Expected
            to be in arcmin.
        check (:obj:`bool`, optional):
            If True, the routine will only check to see if a standard
            star exists within the input ra, dec, and toler range.

    Returns:
        dict or bool: If check is True, return True or False depending on
        if the object is matched to a library standard star.
        If check is False and no match is found, return None.  Otherwise, return
        a dictionary with the matching standard star with the following
        meta data:

            - 'cal_file': str -- Filename table
            - 'name': str -- Star name
            - 'std_ra': str -- RA(J2000)
            - 'std_dec': str -- DEC(J2000)

    """
    # Priority
    std_sets = ['blackbody', 'xshooter', 'calspec', 'esofil', 'noao']

    # SkyCoord
    obj_coord = coordinates.SkyCoord(ra, dec, unit='deg')

    # Loop on standard sets
    closest = dict(sep=999 * units.deg)

    for sset in std_sets:
        path = resource_filename('pypeit', os.path.join('data', 'standards', sset))
        star_file =  os.path.join(path, '{0}_info.txt'.format(sset))
        if not os.path.isfile(star_file):
            msgs.warn('File does not exist!: {0}'.format(star_file))
            continue

        star_tbl = table.Table.read(star_file, comment='#', format='ascii')
        star_coords = coordinates.SkyCoord(star_tbl['RA_2000'], star_tbl['DEC_2000'],
                                           unit=(units.hourangle, units.deg))
        idx, d2d, d3d = coordinates.match_coordinates_sky(obj_coord, star_coords, nthneighbor=1)

        if d2d < toler:
            if check:
                # Found one so return
                return True

            # Generate a dict
            _idx = int(idx)
            std_dict = dict(cal_file=os.path.join(path, star_tbl[_idx]['File']),
                            name=star_tbl[_idx]['Name'],
                            std_ra=star_tbl[_idx]['RA_2000'],
                            std_dec=star_tbl[_idx]['DEC_2000'])

            if not os.path.isfile(star_file):
                # TODO: Error or warn?
                msgs.error("No standard star file found: {:s}".format(star_file))

            # TODO: Does this need to be globbed? Why isn't the file
            # name exact?
            fil = glob.glob(std_dict['cal_file'] + '*')
            if len(fil) == 0:
                # TODO: Error or warn?
                msgs.error("No standard star file: {:s}".format(std_dict['cal_file']))

            fil = fil[0]
            msgs.info("Loading standard star file: {:s}".format(fil))
            # TODO: Put this stuf in a method, like `read_standard`
            if sset == 'xshooter':
                # TODO let's add the star_mag here and get a uniform set of tags in the std_dict
                std_spec = table.Table.read(fil, format='ascii')
                std_dict['std_source'] = sset
                std_dict['wave'] = std_spec['col1'] * units.AA
                std_dict['flux'] = std_spec['col2'] / PYPEIT_FLUX_SCALE * \
                                   units.erg / units.s / units.cm ** 2 / units.AA
            elif sset == 'calspec':
                std_dict['std_source'] = sset
                std_spec = io.fits_open(fil)[1].data
                std_dict['wave'] = std_spec['WAVELENGTH'] * units.AA
                std_dict['flux'] = std_spec['FLUX'] / PYPEIT_FLUX_SCALE \
                                   * units.erg / units.s / units.cm ** 2 / units.AA
            elif sset == 'esofil':
                # TODO let's add the star_mag here and get a uniform set of tags in the std_dict
                std_spec = table.Table.read(fil, format='ascii')
                std_dict['std_source'] = sset
                std_dict['wave'] = std_spec['col1'] * units.AA
                std_dict['flux'] = std_spec['col2']*1e-16/PYPEIT_FLUX_SCALE * \
                                   units.erg / units.s / units.cm ** 2 / units.AA
                # At this low resolution, best to throw out entries affected by A and B-band absorption
                mask = (std_dict['wave'].value > 7551.) & (std_dict['wave'].value < 7749.)
                std_dict['wave'] = std_dict['wave'][np.logical_not(mask)]
                std_dict['flux'] = std_dict['flux'][np.logical_not(mask)]
            elif sset == 'noao': #mostly copied from 'esofil', need to convert the flux units
                # TODO let's add the star_mag here and get a uniform set of tags in the std_dict
                std_spec = table.Table.read(fil, format='ascii')
                std_dict['std_source'] = sset
                std_dict['wave'] = std_spec['col1'] * units.AA
                std_dict['flux'] = mAB_to_cgs(std_spec['col2'],std_spec['col1']) / PYPEIT_FLUX_SCALE * \
                                   units.erg / units.s / units.cm ** 2 / units.AA
                # At this low resolution, best to throw out entries affected by A and B-band absorption
                mask = (std_dict['wave'].value > 7551.) & (std_dict['wave'].value < 7749.)
                std_dict['wave'] = std_dict['wave'][np.logical_not(mask)]
                std_dict['flux'] = std_dict['flux'][np.logical_not(mask)]
            elif sset == 'blackbody':
                # TODO let's add the star_mag here and get a uniform set of tags in the std_dict
                waves, flam = blackbody_func(star_tbl[_idx]['a_x10m23'], star_tbl[_idx]['T_K'])
                std_dict['std_source'] = sset
                std_dict['wave'] = waves * units.AA
                std_dict['flux'] = flam * units.erg / units.s / units.cm ** 2 / units.AA
            else:
                msgs.error('Do not know how to parse {0} file.'.format(sset))
            msgs.info("Fluxes are flambda, normalized to 1e-17")
            return std_dict

        # Save closest found so far
        imind2d = np.argmin(d2d)
        mind2d = d2d[imind2d]
        if mind2d < closest['sep']:
            closest['sep'] = mind2d
            # TODO: Is this right? Do we need to use the imind2d from
            # above?
            closest.update(dict(name=star_tbl[int(idx)]['Name'],
                                ra=star_tbl[int(idx)]['RA_2000'],
                                dec=star_tbl[int(idx)]['DEC_2000']))

    # Standard star not found
    if check:
        return False

    msgs.error("No standard star was found within a tolerance of {:g}".format(toler) + msgs.newline()
               + "Closest standard was {:s} at separation {:g}".format(closest['name'], closest['sep'].to('arcmin')))

    return None

def stellar_model(V, sptype):
    """
    Parse Kurucz SED given T and g.  Also convert absolute/apparent
    magnitudes

    Parameters
    ----------
    V: float
        Apparent magnitude of the telluric star
    sptype: str
        Spectral type of the telluric star

    Returns
    -------
    loglam: `numpy.ndarray`_
        log wavelengths
    flux: `numpy.ndarray`_
        SED f_lambda (cgs units, I think, probably per Ang)
    """

    # Grab telluric star parameters
    # log(g) of the Sun
    logg_sol = np.log10(6.67259e-8) + np.log10(1.989e33) - 2.0 * np.log10(6.96e10)

    # Load Schmidt-Kaler (1982) table
    sk82_file = resource_filename('pypeit', 'data/standards/kurucz93/schmidt-kaler_table.txt')
    sk82_tab = ascii.read(sk82_file, names=('Sp', 'logTeff', 'Teff', '(B-V)_0', 'M_V', 'B.C.', 'M_bol', 'L/L_sol'))

    # TODO, currently this only works on select stellar types. Add ability to interpolate across types.
    # Match input type.
    mti = np.where(sptype == sk82_tab['Sp'])[0]
    if len(mti) != 1:
        raise ValueError('Not ready to interpolate yet.')

    # Calculate final quantities
    # Relation between radius, temp, and bolometric luminosity
    logR = 0.2 * (42.26 - sk82_tab['M_bol'][mti[0]] - 10.0 * sk82_tab['logTeff'][mti[0]])

    # Mass-bolometric luminosity relation from schimdt-kaler p28 valid for M_bol < 7.5
    logM = 0.46 - 0.10 * sk82_tab['M_bol'][mti[0]]
    logg = logM - 2.0 * logR + logg_sol
    M_V = sk82_tab['M_V'][mti[0]]
    Teff = sk82_tab['Teff'][mti[0]]

    # Flux factor (absolute/apparent V mag)
    # Define constants
    parsec = constants.pc.cgs  # 3.086e18
    R_sol = constants.R_sun.cgs  # 6.96e10

    # Distance modulus
    logd = 0.2 * (V - M_V) + 1.0
    D = parsec * 10. ** logd
    R = R_sol * 10. ** logR

    # Factor converts the kurucz surface flux densities to flux observed on Earth
    flux_factor = (R / D.value) ** 2

    # Grab closest T in Kurucz SEDs
    T1 = 3000. + np.arange(28) * 250
    T2 = 10000. + np.arange(6) * 500
    T3 = 13000. + np.arange(22) * 1000
    T4 = 35000. + np.arange(7) * 2500
    Tk = np.concatenate([T1, T2, T3, T4])
    indT = np.argmin(np.abs(Tk - Teff))

    # Grab closest g in Kurucz SEDs
    loggk = np.arange(11) * 0.5
    indg = np.argmin(np.abs(loggk - logg))

    # Grab Kurucz filename
    std_file = resource_filename('pypeit', '/data/standards/kurucz93/kp00/kp00_{:d}.fits.gz'.format(int(Tk[indT])))
    std = table.Table.read(std_file)

    # Grab specific spectrum
    loglam = np.array(np.log10(std['WAVELENGTH']))
    gdict = {0: 'g00', 1: 'g05', 2: 'g10', 3: 'g15', 4: 'g20',
             5: 'g25', 6: 'g30', 7: 'g35', 8: 'g40', 9: 'g45',
             10: 'g50'}
    flux = std[gdict[indg]]

    # scale the model to the V-band magnitude
    star_lam = 10 ** loglam
    star_flux = flux.data * flux_factor
    # Generate a dict matching the output of find_standard_file
    std_dict = dict(cal_file='KuruczTelluricModel', name=sptype, Vmag=V, std_ra=None, std_dec=None)
    std_dict['std_source'] = 'KuruczTelluricModel'
    std_dict['wave'] = star_lam * units.AA
    std_dict['flux'] = star_flux / PYPEIT_FLUX_SCALE * units.erg / units.s / units.cm ** 2 / units.AA

    return std_dict


def get_standard_spectrum(star_type=None, star_mag=None, ra=None, dec=None):
    """
    Get the standard spetrum using given information of your standard/telluric star.

    Args:
        star_type (str):
            Spectral type of your standard/telluric star
        star_mag (float):
            Apparent magnitude of the telluric star
        ra (float):
            Standard right-ascension in hh:mm:ss string format (e.g.,'05:06:36.6').
        dec (float):
            Object declination in dd:mm:ss string format (e.g., 52:52:01.0')

    Returns:
        dict: Dictionary containing the information you provided and the
        standard/telluric spectrum.
    """
    # Create star model
    if (ra is not None) and (dec is not None) and (star_mag is None) and (star_type is None):
        # Pull star spectral model from archive
        msgs.info("Getting archival standard spectrum")
        # Grab closest standard within a tolerance
        std_dict = find_standard_file(ra, dec)

    elif (star_mag is not None) and (star_type is not None):
        ## using vega spectrum
        if 'A0' in star_type:
            msgs.info('Getting vega spectrum')
            ## Vega model from TSPECTOOL
            vega_file = resource_filename('pypeit', '/data/standards/vega_tspectool_vacuum.dat')
            vega_data = table.Table.read(vega_file, comment='#', format='ascii')
            std_dict = dict(cal_file='vega_tspectool_vacuum', name=star_type, Vmag=star_mag, std_ra=ra, std_dec=dec)
            std_dict['std_source'] = 'VEGA'
            std_dict['wave'] = vega_data['col1'] * units.AA

            # vega is V=0.03
            std_dict['flux'] = vega_data['col2'] * 10**(0.4*(0.03-star_mag)) / PYPEIT_FLUX_SCALE * \
                               units.erg / units.s / units.cm ** 2 / units.AA
        ## using Kurucz stellar model
        else:
            # Create star spectral model
            msgs.info("Getting kurucz+93 stellar model")
            std_dict = stellar_model(star_mag, star_type)
            std_dict['std_ra'] = ra
            std_dict['std_dec'] = dec
    else:
        msgs.error('Insufficient information provided for fluxing. '
                   'Either the coordinates of the standard or a stellar type and magnitude are needed.')

    return std_dict


def load_extinction_data(longitude, latitude, toler=5. * units.deg):
    """
    Find the best extinction file to use, based on longitude and latitude
    Loads it and returns a Table

    Parameters
    ----------
    toler : Angle, optional
        Tolerance for matching detector to site (5 deg)

    Returns
    -------
    ext_file : Table
        astropy Table containing the 'wavelength', 'extinct' data for AM=1.
    """
    # Mosaic coord
    mosaic_coord = coordinates.SkyCoord(longitude, latitude, frame='gcrs', unit=units.deg)
    # Read list
    extinct_path = resource_filename('pypeit', '/data/extinction/')
    extinct_summ = extinct_path + 'README'
    extinct_files = table.Table.read(extinct_summ, comment='#', format='ascii')
    # Coords
    ext_coord = coordinates.SkyCoord(extinct_files['Lon'], extinct_files['Lat'], frame='gcrs',
                                     unit=units.deg)
    # Match
    idx, d2d, d3d = coordinates.match_coordinates_sky(mosaic_coord, ext_coord, nthneighbor=1)
    if d2d < toler:
        extinct_file = extinct_files[int(idx)]['File']
        msgs.info("Using {:s} for extinction corrections.".format(extinct_file))
    else:
        msgs.warn("No file found for extinction corrections.  Applying none")
        msgs.warn("You should generate a site-specific file")
        return None
    # Read
    extinct = table.Table.read(extinct_path + extinct_file, comment='#', format='ascii',
                               names=('iwave', 'mag_ext'))
    wave = table.Column(np.array(extinct['iwave']) * units.AA, name='wave')
    extinct.add_column(wave)
    # Return
    return extinct[['wave', 'mag_ext']]

def extinction_correction(wave, airmass, extinct):
    """
    Derive extinction correction
    Based on algorithm in LowRedux (long_extinct)

    Parameters
    ----------
    wave (`numpy.ndarray`_):
        Wavelengths for interpolation. Should be sorted Assumes
        Angstroms
    airmass : float
        Airmass
    extinct : Table
        Table of extinction values

    Returns:
    -------
    `numpy.ndarray`_:
        Flux corrections at the input wavelengths
    """
    # Checks
    if airmass < 1.:
        msgs.error("Bad airmass value in extinction_correction")
    # Interpolate
    f_mag_ext = interpolate.interp1d(extinct['wave'], extinct['mag_ext'], bounds_error=False,
                                     fill_value=0.)
    mag_ext = f_mag_ext(wave)#.to('AA').value)

    # Deal with outside wavelengths
    gdv = np.where(mag_ext > 0.)[0]

    if len(gdv) == 0:
        msgs.warn("No valid extinction data available at this wavelength range. Extinction correction not applied")
    elif gdv[0] != 0:  # Low wavelengths
        mag_ext[0:gdv[0]] = mag_ext[gdv[0]]
        msgs.warn("Extrapolating at low wavelengths using last valid value")
    elif gdv[-1] != (mag_ext.size - 1):  # High wavelengths
        mag_ext[gdv[-1] + 1:] = mag_ext[gdv[-1]]
        msgs.warn("Extrapolating at high wavelengths using last valid value")
    else:
        msgs.info("Extinction data covered the whole spectra. Applying correction...")
    # Evaluate
    flux_corr = 10.0 ** (0.4 * mag_ext * airmass)
    # Return
    return flux_corr


### Routines for standard sensfunc started from here
def find_standard(specobj_list):
    """
    Take the median boxcar and then the max object as the standard

    Parameters
    ----------
    specobj_list : list

    Returns
    -------
    mxix : int
        Index of the standard star

    """
    # Repackage as necessary (some backwards compatability)
    # Do it
    medfx = []
    for indx, spobj in enumerate(specobj_list):
        if spobj is None:
            medfx.append(0.)
        else:
            medfx.append(np.median(spobj.BOX_COUNTS))
    try:
        mxix = np.argmax(np.array(medfx))
    except:
        embed()
    msgs.info("Putative standard star {} has a median boxcar count of {}".format(specobj_list[mxix],
                                                                                 np.max(medfx)))
    # Return
    return mxix

#def apply_standard_sens(spec_obj, sens_dict, airmass, exptime, extinct_correct=True, telluric_correct = False,
#                        longitude=None, latitude=None):
#    """ Apply the sensitivity function to the data
#    We also correct for extinction.
#
#    Parameters
#    ----------
#    spec_obj : dict
#        SpecObj
#    sens_dict : dict
#        Sens Function dict
#    airmass : float
#        Airmass
#    exptime : float
#        Exposure time in seconds
#    longitude : float
#        longitude in degree for observatory
#    latitude: float
#        latitude in degree for observatory. Used for extinction
#        correction
#    """

def sensfunc(wave, counts, counts_ivar, counts_mask, exptime, airmass, std_dict, longitude, latitude, ech_orders=None,
             mask_abs_lines=True, polyorder=4, balm_mask_wid=10.0, nresln=20., resolution=3000.,
             trans_thresh=0.9,polycorrect=True, polyfunc=False, debug=False):
    """
    Function to generate the sensitivity function. This function fits
    a bspline to the 2.5*log10(flux_std/flux_counts). The break
    points spacing, which determines the scale of variation of the
    sensitivity function is determined by the nresln parameter. This
    code can work in different regimes, but NOTE THAT TELLURIC MODE
    IS DEPRECATED, use telluric.sensfunc_telluric instead

    Args:
        wave (`numpy.ndarray`_):
            Wavelength of the star. Shape (nspec,) or (nspec, norders)
        counts (ndarray):
            Flux (in counts) of the star. Shape (nspec,) or (nspec, norders)
        counts_ivar (`numpy.ndarray`_):
            Inverse variance of the star counts. Shape (nspec,) or (nspec, norders)
        counts_mask (`numpy.ndarray`_):
            Good pixel mask for the counts. Shape (nspec,) or (nspec, norders)
        exptime (float):
            Exposure time in seconds
        airmass (float):
            Airmass
        std_dict (dict):
            Dictionary containing information about the standard star returned by flux_calib.get_standard_spectrum
        longitude (float):
            Telescope longitude, used for extinction correction.
        latitude (float):
            Telescope latitude, used for extinction correction
        ech_orders (int `numpy.ndarray`_):
            If passed the echelle orders will be added to the meta_table. ech_orders must be a numpy array of integers
            with the shape (norders,) giving the order numbers
        mask_abs_lines (bool):
            If True, mask stellar absorption lines before fitting sensitivity function. Default = True
        balm_mask_wid (float):
            Parameter describing the width of the mask for or stellar absorption lines (i.e. mask_abs_lines=True). A region
            equal to balm_mask_wid*resln is masked where resln is the estimate for the spectral resolution in pixels
            per resolution element.
        polycorrect (bool):
            Whether you want to interpolate the sensfunc with polynomial in the stellar absortion line regions before
            fitting with the bspline
        nresln (float):
            Parameter governing the spacing of the bspline breakpoints. default = 20.0
        resolution (float):
            Expected resolution of the standard star spectrum. This should probably be determined from the grating, but is
            currently hard wired. default=3000.0
        trans_thresh (float):
            Parameter for selecting telluric regions which are masked. Locations below this transmission value are masked.
            If you have significant telluric absorption you should be using telluric.sensnfunc_telluric. default = 0.9

    Returns:
        Tuple: Returns:

            - meta_table (astropy.Table) -- Table containing meta data
              for the sensitivity function
            - out_table (astropy.Table) -- Table containing the
              sensitivity function

    """

    wave_arr, counts_arr, ivar_arr, mask_arr, nspec, norders = utils.spec_atleast_2d(wave, counts, counts_ivar, counts_mask)
    zeropoint_data = np.zeros_like(wave_arr)
    zeropoint_data_gpm = np.zeros_like(wave_arr, dtype=bool)
    zeropoint_fit = np.zeros_like(wave_arr)
    zeropoint_fit_gpm = np.zeros_like(wave_arr, dtype=bool)
    #mask_sens = np.ones_like(mask_arr)
    wave_min = np.zeros(norders)
    wave_max = np.zeros(norders)

    for iord in range(norders):
        zeropoint_data[:, iord], zeropoint_data_gpm[:, iord], zeropoint_fit[:, iord], zeropoint_fit_gpm[:, iord], = fit_zeropoint(
            wave_arr[:,iord], counts_arr[:,iord], ivar_arr[:,iord], mask_arr[:,iord], exptime, airmass, std_dict,
            longitude, latitude, mask_abs_lines=mask_abs_lines, polyorder=polyorder,
            balm_mask_wid=balm_mask_wid, nresln=nresln, resolution=resolution, trans_thresh=trans_thresh,
            polycorrect=polycorrect, polyfunc=polyfunc, debug=debug)
        wave_min[iord] = wave_arr[wave_arr[:,iord] > 1.0, iord].min()
        wave_max[iord] = wave_arr[wave_arr[:,iord] > 1.0, iord].max()

    # Allocate the meta parameter table, ext=1
    meta_table = table.Table(meta={'name': 'Parameter Values'})
    meta_table['EXPTIME'] = [exptime]
    meta_table['AIRMASS'] = [airmass]
    meta_table['STD_RA'] = [std_dict['std_ra']]
    meta_table['STD_DEC'] = [std_dict['std_dec']]
    meta_table['STD_NAME'] = [std_dict['name']]
    meta_table['CAL_FILE'] = [std_dict['cal_file']]
    if ech_orders is not None:
        meta_table['ECH_ORDERS'] = [ech_orders]
    # Allocate the output table, ext=2
    out_table = table.Table(meta={'name': 'Sensitivity Function'})
    # These are transposed because we need to store them in an astropy table, with number of rows = norders
    out_table['SENS_WAVE'] = wave_arr.T
    out_table['SENS_ZEROPOINT'] = zeropoint_data.T
    out_table['SENS_ZEROPOINT_GPM'] = zeropoint_data_gpm.T
    out_table['SENS_ZEROPOINT_FIT'] = zeropoint_fit.T
    out_table['SENS_ZEROPOINT_FIT_GPM'] = zeropoint_fit_gpm.T
    out_table['WAVE_MIN'] = wave_min
    out_table['WAVE_MAX'] = wave_max

    return meta_table, out_table

def get_sensfunc_factor(wave, wave_zp, zeropoint, exptime, tellmodel=None, extinct_correct=False,
                         airmass=None, longitude=None, latitude=None, extrap_sens=False):
    """
    Get the final sensitivity function factor that will be multiplied into a spectrum in units of counts to flux calibrate it.
    This code interpolates the sensitivity function and can also multiply in extinction and telluric corrections.

    FLAM, FLAM_SIG, and FLAM_IVAR are generated

    Args:
        wave (float `numpy.ndarray`_): shape = (nspec,)
           Senstivity
        wave_zp (float `numpy.ndarray`_):
           Zerooint wavelength vector shape = (nsens,)
        zeropoint (float `numpy.ndarray`_): shape = (nsens,)
           Zeropoint, i.e. sensitivity function
        exptime (float):
        tellmodel (float  `numpy.ndarray`_, optional): shape = (nspec,)
           Apply telluric correction if it is passed it. Note this is deprecated.
        extinct_correct (bool, optional)
           If True perform an extinction correction. Deafult = False
        airmass (float, optional):
           Airmass used if extinct_correct=True. This is required if extinct_correct=True
        longitude (float, optional):
            longitude in degree for observatory
            Required for extinction correction
        latitude:
            latitude in degree for observatory
            Required  for extinction correction
        extrap_sens (bool, optional):
            Extrapolate the sensitivity function (instead of crashing out)

    Returns:
        sensfunc_factor (`numpy.ndarray`_): shape = (nspec,)
            This quantity is defined to be sensfunc_interp/exptime/delta_wave

    """

    zeropoint_obs = np.zeros_like(wave)
    wave_mask = wave > 1.0  # filter out masked regions or bad wavelengths
    delta_wave = wvutils.get_delta_wave(wave, wave_mask)
    try:
        zeropoint_obs[wave_mask] = interpolate.interp1d(wave_zp, zeropoint, bounds_error=True)(wave[wave_mask])
    except ValueError:
        if extrap_sens:
            zeropoint_obs[wave_mask] = interpolate.interp1d(wave_zp, zeropoint, bounds_error=False)(wave[wave_mask])
            msgs.warn(
                "Your data extends beyond the bounds of your sensfunc. You should be adjusting "
                "the par['sensfunc']['extrap_blu'] and/or par['sensfunc']['extrap_red'] to extrapolate further "
                "and recreate your sensfunc. But we are extrapolating per your direction. Good luck!")
        else:
            msgs.error("Your data extends beyond the bounds of your sensfunc. " + msgs.newline() +
                       "Adjust the par['sensfunc']['extrap_blu'] and/or par['sensfunc']['extrap_red'] to extrapolate "
                       "further and recreate your sensfunc.")

    # This is the S_lam factor required to convert N_lam = counts/sec/Ang to F_lam = 1e-17 erg/s/cm^2/Ang, i.e.
    # F_lam = S_lam*N_lam
    sensfunc_obs = Nlam_to_Flam(wave, zeropoint_obs)

    # TODO Telluric corrections via this method are deprecated
    # Did the user request a telluric correction?
    if tellmodel is not None:
        # This assumes there is a separate telluric key in this dict.
        msgs.info('Applying telluric correction')
        sensfunc_obs = sensfunc_obs * (tellmodel > 1e-10) / (tellmodel + (tellmodel < 1e-10))


    if extinct_correct:
        if longitude is None or latitude is None:
            msgs.error('You must specify longitude and latitude if we are extinction correcting')
        # Apply Extinction if optical bands
        msgs.info("Applying extinction correction")
        msgs.warn("Extinction correction applyed only if the spectra covers <10000Ang.")
        extinct = load_extinction_data(longitude, latitude)
        ext_corr = extinction_correction(wave * units.AA, airmass, extinct)
        senstot = sensfunc_obs * ext_corr
    else:
        senstot = sensfunc_obs.copy()

    # senstot is the conversion from N_lam to F_lam, and the division by exptime and delta_wave are to convert
    # the spectrum in counts/pixel into units of N_lam = counts/sec/angstrom
    return senstot/exptime/delta_wave


# JFH TODO This code needs to be cleaned up. The telluric option should probably be removed. Logic is not easy to follow.
def fit_zeropoint(wave, counts, counts_ivar, counts_mask, exptime, airmass, std_dict, longitude, latitude,
                  mask_abs_lines=True, polyorder=4, balm_mask_wid=10.0, nresln=20., resolution=3000.,
                  trans_thresh=0.9, polycorrect=True, polyfunc=False, debug=False):

    """

    Function to generate the sensitivity function. This function fits
    a bspline to the 2.5*log10(flux_std/flux_counts). The break
    points spacing, which determines the scale of variation of the
    sensitivity function is determined by the nresln parameter. This
    code can work in different regimes, but NOTE THAT TELLURIC MODE
    IS DEPRECATED, use telluric.sensfunc_telluric instead.

        - If telluric=False, a sensfunc is generated by fitting a
          bspline to the using nresln=20.0 and masking out telluric
          regions.

        - If telluric=True, sensfunc is a pixelized sensfunc (not
          smooth) for correcting both throughput and telluric lines.
          if you set polycorrect=True, the sensfunc in the Hydrogen
          recombination line region (often seen in star spectra) will
          be replaced by a smoothed polynomial function.

    Args:
        wave (`numpy.ndarray`_):
            Wavelength of the star. Shape (nspec,)
        counts (`numpy.ndarray`_):
            Flux (in counts) of the star. Shape (nspec,)
        counts_ivar (`numpy.ndarray`_):
            Inverse variance of the star counts. Shape (nspec,)
        counts_mask (`numpy.ndarray`_):
            Good pixel mask for the counts.
        exptime (float):
            Exposure time in seconds
        airmass (float):
            Airmass
        std_dict (dict):
            Dictionary containing information about the standard star returned by flux_calib.get_standard_spectrum
        longitude (float):
            Telescope longitude, used for extinction correction.
        latitude (float):
            Telescope latitude, used for extinction correction
        mask_abs_lines (bool):
            If True, mask stellar absorption lines before fitting sensitivity function. Default = True
        balm_mask_wid (float):
            Parameter describing the width of the mask for or stellar absorption lines (i.e. mask_abs_lines=True). A region
            equal to balm_mask_wid*resln is masked where resln is the estimate for the spectral resolution in pixels
            per resolution element.
        polycorrect: bool
            Whether you want to interpolate the zeropoint with polynomial in the stellar absortion line regions before
            fitting with the bspline
        nresln (float):
            Parameter governing the spacing of the bspline breakpoints. default = 20.0
        resolution (float):
            Expected resolution of the standard star spectrum. This should probably be determined from the grating, but is
            currently hard wired. default=3000.0
        trans_thresh (float):
            Parameter for selecting telluric regions which are masked. Locations below this transmission value are masked.
            If you have significant telluric absorption you should be using telluric.sensnfunc_telluric. default = 0.9

    Returns:
            zeropoint (`numpy.ndarray`_): Sensitivity function with same shape as wave (nspec,)
            mask_sens (`numpy.ndarray`_): Good pixel mask for sensitivity function with same shape as wave (nspec,)

    """
    # Create copy of the arrays to avoid modification and convert to
    # Nlam = electrons/s/Angstrom
    delta_wave = wvutils.get_delta_wave(wave, (wave > 1.0))
    Nlam_star = counts/exptime/delta_wave
    Nlam_ivar_star = delta_wave**2*counts_ivar*exptime**2

    # Extinction correction
    msgs.info("Applying extinction correction")
    extinct = load_extinction_data(longitude,latitude)
    ext_corr = extinction_correction(wave * units.AA, airmass, extinct)
    # Correct for extinction
    Nlam_star = Nlam_star * ext_corr
    Nlam_ivar_star = Nlam_ivar_star / ext_corr ** 2
    gpm_star = counts_mask

    # Interpolate the standard star onto the current set of observed wavelengths
    flux_true = interpolate.interp1d(std_dict['wave'], std_dict['flux'], bounds_error=False,
                                     fill_value='extrapolate')(wave)
    # Do we need to extrapolate? TODO Replace with a model or a grey body?
    ## TODO This is an ugly hack. Why are we only triggering this if the extrapolated star is negative.
    if np.min(flux_true) <= 0.:
        msgs.warn('Your spectrum extends beyond calibrated standard star, extrapolating the spectra with polynomial.')
        mask_model = flux_true <= 0
        pypeitFit = fitting.robust_fit(std_dict['wave'].value, std_dict['flux'].value,8,function='polynomial',
                                                    maxiter=50, lower=3.0, upper=3.0, maxrej=3,
                                                    grow=0, sticky=True, use_mad=True)
        star_poly = pypeitFit.eval(wave)
        #flux_true[mask_model] = star_poly[mask_model]
        flux_true = star_poly.copy()
        if debug:
            plt.plot(std_dict['wave'], std_dict['flux'],'bo',label='Raw Star Model')
            plt.plot(std_dict['wave'],  pypeitFit.eval(std_dict['wave'].value),
                     'k-',label='robust_poly_fit')
            plt.plot(wave,flux_true,'r-',label='Your Final Star Model used for sensfunc')
            plt.show()

    # Get masks from observed star spectrum. True = Good pixels
    mask_bad, mask_balm, mask_tell = get_mask(wave, Nlam_star, Nlam_ivar_star, gpm_star, mask_abs_lines=mask_abs_lines,
                                              mask_telluric=True, balm_mask_wid=balm_mask_wid, trans_thresh=trans_thresh)

    # Get zeropoint
    zeropoint_data, zeropoint_data_gpm, zeropoint_fit, zeropoint_fit_gpm = standard_zeropoint(
        wave, Nlam_star, Nlam_ivar_star, mask_bad, flux_true, mask_balm=mask_balm,
        mask_tell=mask_tell, maxiter=35, upper=3.0, lower=3.0, polyorder=polyorder,
        balm_mask_wid=balm_mask_wid, nresln=nresln, resolution=resolution,
        polycorrect=polycorrect, polyfunc=polyfunc, debug=debug, show_QA=False)

    if debug:
        sensfactor = Nlam_to_Flam(wave, zeropoint_fit)
        plt.plot(wave[zeropoint_fit_gpm], flux_true[zeropoint_fit_gpm], color='k',lw=2, label='Reference Star')
        plt.plot(wave[zeropoint_fit_gpm], Nlam_star[zeropoint_fit_gpm]*sensfactor[zeropoint_fit_gpm], color='r', label='Fluxed Observed Star')
        plt.xlabel(r'Wavelength [$\AA$]')
        plt.ylabel('Flux [erg/s/cm2/Ang.]')
        plt.legend(fancybox=True, shadow=True)
        plt.show()


    return zeropoint_data, zeropoint_data_gpm, zeropoint_fit, zeropoint_fit_gpm




def get_mask(wave_star,flux_star, ivar_star, mask_star, mask_abs_lines=True, mask_telluric=True, balm_mask_wid=10., trans_thresh=0.9):
    '''
    Get a couple of masks from your observed standard spectrum.

    Args:
      wave_star: numpy array
        wavelength array of your spectrum
      flux_star: numpy array
        flux array of your spectrum
      ivar_star:
        ivar array of your spectrum
      mask_star: bool
        whether you need to mask Hydrogen recombination line region. If False, the returned msk_star are all good.
      mask_tell: bool
        whether you need to mask telluric region. If False, the returned msk_tell are all good.
      trans_thresh: float
        parameter for selecting telluric regions.

    Returns:
      msk_bad: bool type numpy array
        mask for bad pixels.
      msk_star: bool type numpy array
        mask for recombination lines in star spectrum.
      msk_tell: bool type numpy array
        mask for telluric regions.
    '''

    # Mask (True = good pixels)
    # mask for recombination lines
    mask_balm = np.ones_like(flux_star).astype(bool)
    # mask for telluric regions
    mask_tell = np.ones_like(flux_star).astype(bool)

    # masking bad entries
    msgs.info(" Masking bad pixels")
    mask_bad = mask_star.copy()
    mask_bad[ivar_star <= 0.] = False
    mask_bad[flux_star <= 0.] = False
    # Mask edges
    msgs.info(" Masking edges")
    mask_bad[[0, -1]] = False
    # Mask Atm. cutoff
    msgs.info(" Masking Below the atmospheric cutoff")
    atms_cutoff = wave_star <= 3000.0
    mask_bad[atms_cutoff] = False

    # TODO JFH replace with mask_star_lines from telluric.py
    if mask_abs_lines:
        # Mask Balmer, Paschen, Brackett, and Pfund recombination lines
        msgs.info("Masking recombination lines:")
        # Mask Balmer
        msgs.info(" Masking Balmer")
        lines_balm = np.array([3836.4, 3969.6, 3890.1, 4102.8, 4102.8, 4341.6, 4862.7, 5407.0,
                               6564.6, 8224.8, 8239.2])
        for line_balm in lines_balm:
            ibalm = np.abs(wave_star - line_balm) <= balm_mask_wid
            mask_balm[ibalm] = False
        # Mask Paschen
        msgs.info(" Masking Paschen")
        # air wavelengths from:
        # https://www.subarutelescope.org/Science/Resources/lines/hi.html
        lines_pasc = np.array([8203.6, 8440.3, 8469.6, 8504.8, 8547.7, 8600.8, 8667.4, 8752.9,
                               8865.2, 9017.4, 9229.0, 9546.0, 10049.4, 10938.1,
                               12818.1, 18751.0])
        for line_pasc in lines_pasc:
            ipasc = np.abs(wave_star - line_pasc) <= balm_mask_wid
            mask_balm[ipasc] = False
        # Mask Brackett
        msgs.info(" Masking Brackett")
        # air wavelengths from:
        # https://www.subarutelescope.org/Science/Resources/lines/hi.html
        lines_brac = np.array([14584.0, 18174.0, 19446.0, 21655.0,26252.0, 40512.0])
        for line_brac in lines_brac:
            ibrac = np.abs(wave_star - line_brac) <= balm_mask_wid
            mask_balm[ibrac] = False
        # Mask Pfund
        msgs.info(" Masking Pfund")
        # air wavelengths from:
        # https://www.subarutelescope.org/Science/Resources/lines/hi.html
        lines_pfund = np.array([22788.0, 32961.0, 37395.0, 46525.0,74578.0])
        for line_pfund in lines_pfund:
            ipfund = np.abs(wave_star - line_pfund) <= balm_mask_wid
            mask_balm[ipfund] = False

    if mask_telluric:
        ## Mask telluric region in the optical
        tell_opt = np.any([((wave_star >= 6270.00) & (wave_star <= 6290.00)), # H2O
                       ((wave_star >= 6850.00) & (wave_star <= 6960.00)), #O2 telluric band
                       ((wave_star >= 7580.00) & (wave_star <= 7750.00)), #O2 telluric band
                       ((wave_star >= 7160.00) & (wave_star <= 7340.00)), #H2O
                       ((wave_star >= 8150.00) & (wave_star <= 8250.00))],axis=0) #H2O
        mask_tell[tell_opt] = False
        ## Mask near-infrared telluric region
        if np.max(wave_star)>9100.0:
            # ToDo: should use the specific atmosphere transmission after FBD get the grid.
            ## Read atmosphere transmission
            #
            #if watervp <1.5:
            #    skytrans_file = resource_filename('pypeit', '/data/skisim/'+'mktrans_zm_10_10.dat')
            #elif (watervp>=1.5 and watervp<2.3):
            #    skytrans_file = resource_filename('pypeit', '/data/skisim/'+'mktrans_zm_16_10.dat')
            #elif (watervp>=2.3 and watervp<4.0):
            #    skytrans_file = resource_filename('pypeit', '/data/skisim/' + 'mktrans_zm_30_10.dat')
            #else:
            #    skytrans_file = resource_filename('pypeit', '/data/skisim/' + 'mktrans_zm_50_10.dat')
            #
            skytrans_file = resource_filename('pypeit', '/data/skisim/' + 'mktrans_zm_10_10.dat')
            skytrans = ascii.read(skytrans_file)
            wave_trans, trans = skytrans['wave'].data*10000.0, skytrans['trans'].data
            trans_use = (wave_trans>=np.min(wave_star)-100.0) & (wave_trans<=np.max(wave_star)+100.0)
            # Estimate the resolution of your spectra.
            # I assumed 3 pixels per resolution. This gives an approximate right resolution at the middle point.
            resolution = np.median(wave_star) / np.median(wave_star - np.roll(wave_star, 1)) / 3
            trans_convolved, px_sigma, px_bin = conv2res(wave_trans[trans_use], trans[trans_use], resolution,
                                                         central_wl='midpt', debug=False)
            trans_final = interpolate.interp1d(wave_trans[trans_use], trans_convolved,
                                               bounds_error=False,
                                               fill_value='extrapolate')(wave_star)
            tell_nir = (trans_final<trans_thresh) & (wave_star>9100.0)
            mask_tell[tell_nir] = False
        else:
            msgs.info('Your spectrum is bluer than 9100A, only optical telluric regions are masked.')

    return mask_bad, mask_balm, mask_tell


# These are physical limits on the allowed values of the zeropoint in magnitudes

def Nlam_to_Flam(wave, zeropoint, zp_min=5.0, zp_max=30.0):
    """
    The factor that when multiplied into N_lam converts to F_lam, i.e. S_lam where S_lam \equiv F_lam/N_lam

    Parameters
    ----------
    wave (`numpy.ndarray`_):
       Wavelength vector for zeropoint
    zeropoint (`numpy.ndarray`_):
       zeropoint
    zp_min (float, optional):
       Minimum allowed value of the ZP. For smaller values the S_lam factor is set to zero
    zp_max (float, optional):
       Maximum allowed value of the ZP. For larger values the S_lam factor is set to zero

    Returns
    -------

    """
    gpm = (wave > 1.0) & (zeropoint > zp_min) & (zeropoint < zp_max)
    factor = np.zeros_like(wave)
    factor[gpm] = np.power(10.0, -0.4*(zeropoint[gpm] - ZP_UNIT_CONST))/np.square(wave[gpm])
    return factor

def Flam_to_Nlam(wave, zeropoint, zp_min=5.0, zp_max=30.0):
    """
    The factor that when multiplied into F_lam converts to N_lam, i.e. 1/S_lam where S_lam \equiv F_lam/N_lam


    Parameters
    ----------
    wave (`numpy.ndarray`_):
       Wavelength array, float, shape (nspec,)
    zeropoint (`numpy.ndarray`_):
       zeropoint array, float, shape (nspec,)

    Returns:
    --------
    `numpy.ndarray`_:
        Factor that when multiplied into F_lam converts to N_lam

    """
    gpm = (wave > 1.0) & (zeropoint > zp_min) & (zeropoint < zp_max)
    factor = np.zeros_like(wave)
    factor[gpm] = np.power(10.0, 0.4*(zeropoint[gpm] - ZP_UNIT_CONST))*np.square(wave[gpm])
    return factor


def compute_zeropoint(wave, N_lam, N_lam_gpm, flam_std_star, tellmodel=None):
    """
    Routine to compute the zeropoint and zeropoint_gpm from the N_lam (counts/s/A) of a standard star


    Parameters
    ----------
    wave (`numpy.ndarray`_):
        Wavelength array, float, shape (nspec,)
    N_lam (`numpy.ndarray`_):
        N_lam spectrum of standard star, float, shape (nspec,)
    N_lam_gpm (`numpy.ndarray`_):
        N_lam mask, good pixel mask, boolean, shape (nspec,)
    flam_std_star (`numpy.ndarray`_):
        True standard star spectrum units set of PYPEIT_FLUX_SCALE erg/s/cm^2/sm/Angstrom
    tellmodel (`numpy.ndarray`_):
        Telluric absorption model, optional, shape (nspec,)

    Returns:
    --------
    zeropoint (`numpy.ndarray`_):
        Spectroscopic zeropoint, float, shape (nspec,)
    zeropoint_gpm (`numpy.ndarray`_):
        Zeropoint good pixel mask, bool, shape  (nspec,)
    """

    tellmodel = np.ones_like(N_lam) if tellmodel is None else tellmodel
    S_nu_dimless = np.square(wave)*tellmodel*flam_std_star*utils.inverse(N_lam)
    zeropoint = -2.5*np.log10(S_nu_dimless + (S_nu_dimless <= 0.0)) + ZP_UNIT_CONST
    zeropoint_gpm = N_lam_gpm & np.isfinite(zeropoint) & (N_lam > 0.0) & (S_nu_dimless > 0.0) & \
                    np.isfinite(flam_std_star) & (wave > 1.0)
    return zeropoint, zeropoint_gpm

#def throughput_from_sensfile(sensfile):
#
#    wave, zeropoint, meta_table, out_table, header_sens = sensfunc.SensFunc.load(sensfile)
#    spectrograph = util.load_spectrograph(header_sens['PYP_SPEC'])
#    throughput = zeropoint_to_thru(wave, zeropoint, spectrograph.telescope.eff_aperture())
#    return wave, throughput


def zeropoint_to_throughput(wave, zeropoint, eff_aperture):
    """
    Routine to compute the spectrograph throughput from the zeropoint and effective aperture.

    Parameters
    ----------
    wave (`numpy.ndarray`_):
         Wavelength array shape (nspec,) or (nspec, norders)
    zeropoint (`numpy.ndarray`_):
         Zeropoint array shape (nspec,) or (nspec, norders)
    eff_aperture (float):
         Effective aperture of the telescope in m^2. See spectrograph object

    Returns
    -------
       throughput (`numpy.ndarray`_):
           Throughput of the spectroscopic setup. Same shape as wave and zeropoint

    """

    eff_aperture_m2 = eff_aperture*u.m**2
    S_lam_units = 1e-17*u.erg/u.cm**2
    # Set the throughput to be -1 in places where it is not defined.
    throughput = np.full_like(zeropoint, -1.0)
    zeropoint_gpm = (zeropoint > 5.0) & (zeropoint < 30.0) & (wave > 1.0)
    inv_S_lam = Flam_to_Nlam(wave[zeropoint_gpm], zeropoint[zeropoint_gpm])/S_lam_units
    inv_wave = utils.inverse(wave[zeropoint_gpm])/u.angstrom
    thru = ((const.h*const.c)*inv_wave/eff_aperture_m2*inv_S_lam).decompose()
    throughput[zeropoint_gpm] = thru
    return throughput


def zeropoint_qa_plot(wave, zeropoint_data, zeropoint_data_gpm, zeropoint_fit, zeropoint_fit_gpm, title='Zeropoint QA', axis=None, show=False):
    """
    QA plot for zeropoint plotting

    Parameters
    ----------
    wave
    zeropoint_data
    zeropoint_data_gpm
    zeropoint_fit
    zeropoint_fit_gpm
    title
    order
    axis
    show

    Returns
    -------

    """

    wv_gpm = wave > 1.0
    if axis is None:
        plt.close()
        fig = plt.figure(figsize=(12,8))
        axis = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    rejmask = zeropoint_data_gpm[wv_gpm] & np.logical_not(zeropoint_fit_gpm[wv_gpm])
    axis.plot(wave[wv_gpm], zeropoint_data[wv_gpm], label='Zeropoint estimated', drawstyle='steps-mid', color='k', alpha=0.7, zorder=5, linewidth=1.0)
    axis.plot(wave[wv_gpm], zeropoint_fit[wv_gpm], label='Zeropoint fit', color='red', linewidth=2.0, zorder=7, alpha=0.7)
    axis.plot(wave[wv_gpm][rejmask], zeropoint_data[wv_gpm][rejmask], 's', zorder=10, mfc='None', mec='blue', mew=0.7, label='rejected pixels')
    axis.plot(wave[wv_gpm][np.logical_not(zeropoint_data_gpm[wv_gpm])], zeropoint_data[wv_gpm][np.logical_not(zeropoint_data_gpm[wv_gpm])], 'v',
             zorder=9, mfc='None', mec='orange', mew=0.7, label='originally masked')
    med_filt_mask = zeropoint_data_gpm[wv_gpm] & np.isfinite(zeropoint_data[wv_gpm])
    zp_med_filter = utils.fast_running_median(zeropoint_data[wv_gpm][med_filt_mask], 11)
    axis.set_ylim(0.95 * zp_med_filter.min(), 1.05 * zp_med_filter.max())
    axis.legend()
    axis.set_xlabel('Wavelength')
    axis.set_ylabel('Zeropoint (AB mag)')
    axis.set_title(title, fontsize=12)
    if show:
        plt.show()


def standard_zeropoint(wave, Nlam, Nlam_ivar, Nlam_gpm, flam_true, mask_balm=None, mask_tell=None,
                       maxiter=35, upper=3.0, lower=3.0, func = 'polynomial', polyorder=5, balm_mask_wid=50.,
                       nresln=20., resolution=2700., polycorrect=True, debug=False, polyfunc=False, show_QA=False):
    """
    Generate a sensitivity function based on observed flux and standard spectrum.

    Parameters
    ----------
    wave : `numpy.ndarray`_
      wavelength as observed
    Nlam : `numpy.ndarray`_
      counts/s/Angstrom as observed
    Nlam_ivar : `numpy.ndarray`_
      inverse variance of counts/s/Angstrom
    flam_true : Quantity array
      standard star true flux (erg/s/cm^2/A)
    msk_bad : `numpy.ndarray`_
      mask for bad pixels. True is good.
    msk_star: `numpy.ndarray`_
      mask for hydrogen recombination lines. True is good.
    msk_tell: `numpy.ndarray`_
      mask for telluric regions. True is good.
    maxiter : integer
      maximum number of iterations for polynomial fit
    upper : integer
      number of sigma for rejection in polynomial
    lower : integer
      number of sigma for rejection in polynomial
    polyorder : integer
      order of polynomial fit
    balm_mask_wid: float
      in units of angstrom
      Mask parameter for Balmer absorption. A region equal to
      balm_mask_wid is masked.
    resolution: integer/float.
      spectra resolution
      This paramters should be removed in the future. The resolution should be estimated from spectra directly.
    debug : bool
      if True shows some dubugging plots

    Returns
    -------
    zeropoint ( `numpy.ndarray`_):
      Spectroscopic zeropoint.
    """
    if np.any(np.invert(np.isfinite(Nlam_ivar))):
        msgs.warn("NaN are present in the inverse variance")

    # check masks
    if mask_tell is None:
        mask_tell = np.ones_like(wave,dtype=bool)
    if mask_balm is None:
        mask_balm = np.ones_like(wave, dtype=bool)

    #S_nu_dimless = np.square(wave)*flam_true*utils.inverse(Nlam)*(Nlam > 0.0)
    #zeropoint_data = -2.5*np.log10(S_nu_dimless) + telluric.zp_unit_const()
    # zeropoint_gpm is the pixels for which zp is not defined, zeropoint_fitmask includes additional Balmer/Telluric masking for polyfit
    #zeropoint_gpm = Nlam_gpm & np.isfinite(zeropoint_data) & (Nlam > 0.0) & np.isfinite(flam_true) & (wave > 1.0)
    zeropoint_data, zeropoint_data_gpm = compute_zeropoint(wave, Nlam, Nlam_gpm, flam_true)


    zeropoint_fitmask = zeropoint_data_gpm & mask_tell & mask_balm
    wave_min = wave[wave > 1.0].min()
    wave_max = wave[wave > 1.0].max()

    pypeitFit = fitting.robust_fit(wave, zeropoint_data, polyorder, function=func,
                                minx=wave_min, maxx=wave_max, in_gpm=zeropoint_fitmask,
                                lower=lower, upper=upper, groupbadpix=False,
                                grow=0, sticky=True, use_mad=True)

    zeropoint_poly = pypeitFit.eval(wave)
    # Robustly characterize the stanarad deviation for the b-spline fitting.
    zp_dev_mean, zp_dev_median, zp_std = stats.sigma_clipped_stats(zeropoint_data - zeropoint_poly, np.invert(zeropoint_fitmask),
                                                                   cenfunc='median', stdfunc=utils.nan_mad_std,
                                                                   maxiters=10, sigma_lower=lower, sigma_upper=upper)
    zeropoint_ivar = np.ones_like(zeropoint_data)/zp_std**2

    ZP_MAX = 40.0
    ZP_MIN = 5.0

    zeropoint_clean = zeropoint_data.copy()
    zeropoint_clean_gpm = zeropoint_data_gpm.copy()
    # Polynomial corrections on Hydrogen Recombination lines
    if ((np.sum(zeropoint_fitmask) > 0.5 * len(zeropoint_fitmask)) & polycorrect):
        ## Only correct Hydrogen Recombination lines with polyfit in the telluric free region
        balmer_clean = np.zeros_like(wave, dtype=bool)
        # Commented out the bluest recombination lines since they are weak for spectroscopic standard stars.
        #836.4, 3969.6, 3890.1, 4102.8, 4102.8, 4341.6, 4862.7,   \
        lines_hydrogen = np.array([5407.0, 6564.6, 8224.8, 8239.2, 8203.6, 8440.3, 8469.6, 8504.8, 8547.7, 8600.8, \
                                   8667.4, 8752.9, 8865.2, 9017.4, 9229.0, 10049.4, 10938.1, 12818.1, 21655.0])
        for line_hydrogen in lines_hydrogen:
            ihydrogen = np.abs(wave - line_hydrogen) <= balm_mask_wid
            balmer_clean[ihydrogen] = True
        # Clean pixels which hit Balmer lines or which have the zeropoint_data outside the min/max range
        # AND have polynomial values inside the min/max range
        msk_clean = ((balmer_clean) | (zeropoint_clean > ZP_MAX) | (zeropoint_clean < ZP_MIN)) & \
                    (zeropoint_poly > ZP_MIN) & (zeropoint_poly < ZP_MAX)
        zeropoint_clean[msk_clean] = zeropoint_poly[msk_clean]
        gpm = np.isfinite(Nlam_ivar) & (Nlam_ivar > 0)
        zeropoint_clean[np.invert(gpm)] = zeropoint_poly[np.invert(gpm)]
    else:
        ## if half more than half of your spectrum is masked (or polycorrect=False) then do not correct it with polyfit
        msgs.warn('No polynomial corrections performed on Hydrogen Recombination line regions')

    # ToDo
    # Compute an effective resolution for the standard. This could be improved
    # to setup an array of breakpoints based on the resolution. At the
    # moment we are using only one number
    msgs.work("Should pull resolution from arc line analysis")
    msgs.work("At the moment the resolution is taken as the PixelScale")
    msgs.work("This needs to be changed!")
    std_pix = np.median(np.abs(wave - np.roll(wave, 1)))
    std_res = np.median(wave/resolution) # median resolution in units of Angstrom.
    if (nresln * std_res) < std_pix:
        msgs.warn("Bspline breakpoints spacing shoud be larger than 1pixel")
        msgs.warn("Changing input nresln to fix this")
        nresln = std_res / std_pix

    # Fit zeropoint with bspline
    kwargs_bspline = {'bkspace': std_res * nresln}
    kwargs_reject = {'maxrej': 5}
    msgs.info("Initialize bspline for flux calibration")
    init_bspline = bspline.bspline(wave, bkspace=kwargs_bspline['bkspace'])
    fullbkpt = init_bspline.breakpoints

    # remove masked regions from breakpoints
    msk_bkpt = interpolate.interp1d(wave, zeropoint_clean_gpm, kind='nearest', fill_value='extrapolate')(fullbkpt)
    init_breakpoints = fullbkpt[msk_bkpt > 0.999]

    # init_breakpoints = fullbkpt
    msgs.info("Bspline fit on zeropoint. ")
    bset1, bmask = fitting.iterfit(wave, zeropoint_clean, invvar=zeropoint_ivar, inmask=zeropoint_fitmask, upper=upper, lower=lower,
                                fullbkpt=init_breakpoints, maxiter=maxiter, kwargs_bspline=kwargs_bspline,
                                kwargs_reject=kwargs_reject)
    zeropoint_bspl, zeropoint_fit_gpm = bset1.value(wave)
    zeropoint_bspl_bkpt, _ = bset1.value(init_breakpoints)

    if debug:
        # Check for calibration
        plt.figure(1)
        plt.plot(wave, zeropoint_data, drawstyle='steps-mid', color='black', label='Zeropoint Data')
        plt.plot(wave, zeropoint_bspl, color='cornflowerblue', label='Bspline fit')
        plt.plot(wave[np.invert(zeropoint_fitmask)], zeropoint_data[np.invert(zeropoint_fitmask)], '+', color='red', markersize=5.0,
                 label='masked zeropoint')
        plt.plot(wave[np.invert(zeropoint_fitmask)], zeropoint_bspl[np.invert(zeropoint_fitmask)], '+', color='red', markersize=5.0,
                 label='masked zeropoint_bspl_fit')
        plt.plot(init_breakpoints, zeropoint_bspl_bkpt, '.', color='green', markersize=4.0, label='breakpoints')
        plt.plot(init_breakpoints, np.interp(init_breakpoints, wave, zeropoint_data), '.', color='green',
                 markersize=4.0,
                 label='data interpolated onto breakpoints')
        plt.plot(wave, 1.0 / np.sqrt(zeropoint_ivar), color='orange', label='sigma used for fits')
        plt.legend()
        plt.xlabel('Wavelength [ang]')
        med_filt_mask = zeropoint_data_gpm & np.isfinite(zeropoint_data)
        zp_med_filter = utils.fast_running_median(zeropoint_data[med_filt_mask], 11)
        plt.ylim(0.95 * zp_med_filter.min(), 1.05 * zp_med_filter.max())
        plt.title('Bspline fit')
        plt.show()

    if ((np.sum(zeropoint_fitmask) > 0.5 * len(zeropoint_fitmask)) & polycorrect):
        msk_clean = ((balmer_clean) | (zeropoint_data > ZP_MAX) | (zeropoint_data < ZP_MIN)) & \
                    (zeropoint_poly > ZP_MIN) & (zeropoint_poly < ZP_MAX)
        zeropoint_bspl_clean = zeropoint_bspl.copy()
        zeropoint_bspl_clean[msk_clean] = zeropoint_poly[msk_clean]
        msk_badpix = np.isfinite(Nlam_ivar) & (Nlam_ivar>0)
        zeropoint_bspl_clean[np.invert(msk_badpix)] = zeropoint_poly[np.invert(msk_badpix)]
    else:
        ## if half more than half of your spectrum is masked (or polycorrect=False) then do not correct it with polyfit
        zeropoint_bspl_clean = zeropoint_bspl.copy()
        msgs.warn('No polynomial corrections performed on Hydrogen Recombination line regions')

    # Calculate zeropoint
    zeropoint_fit = zeropoint_poly if polyfunc else zeropoint_bspl_clean


    # TODO Should we return the bspline fitmask here?
    return zeropoint_data, zeropoint_data_gpm, zeropoint_fit, zeropoint_fit_gpm

def load_filter_file(filter):
    """
    Load a system response curve for a given filter

    Args:
        filter (str): Name of filter

    Returns:
        `numpy.ndarray`_: wavelength, instrument throughput


    # Optical filters
    BASS_MZLS_filters = ['BASS-MZLS-{}'.format(i) for i in ['G', 'R','Z']]
    CFHT_filters = ['CFHT-{}'.format(i) for i in ['U', 'G', 'R', 'I', 'Z']]
    DECAM_filters = ['DECAM-{}'.format(i) for i in ['U', 'G', 'R', 'I', 'Z', 'Y']]
    HSC_filters = ['HSC-{}'.format(i) for i in ['G', 'R', 'I', 'Z', 'Y']]
    LSST_filters = ['LSST-{}'.format(i) for i in ['U', 'G', 'R', 'I', 'Z', 'Y']]
    PS1_filters = ['PS1-{}'.format(i) for i in ['G', 'R', 'I', 'Z', 'Y']]
    SDSS_filters = ['SDSS-{}'.format(i) for i in ['U', 'G', 'R', 'I', 'Z']]

    # NIR filters
    UKIDSS_filters = ['UKIRT-{}'.format(i) for i in ['Y', 'J', 'H', 'K']]
    VISTA_filters = ['VISTA-{}'.format(i) for i in ['Z', 'Y', 'J', 'H', 'K']]
    TMASS_filters = ['TMASS-{}'.format(i) for i in ['J', 'H', 'K']]

    # Other filters
    GAIA_filters = ['GAIA-{}'.format(i) for i in ['G', 'B', 'R']]
    GALEX_filters = ['GALEX-{}'.format(i) for i in ['F', 'N']]
    WISE_filters = ['WISE-{}'.format(i) for i in ['W1', 'W2', 'W3', 'W4']]

    allowed_options = BASS_MZLS_filters + CFHT_filters + DECAM_filters + HSC_filters \
                      + LSST_filters + PS1_filters + SDSS_filters + UKIDSS_filters\
                      + VISTA_filters + TMASS_filters + GAIA_filters + GALEX_filters + WISE_filters
    """

    filter_file = resource_filename('pypeit', os.path.join('data', 'filters', 'filter_list.ascii'))
    tbl = table.Table.read(filter_file, format='ascii')

    allowed_options = tbl['filter'].data

    # Check
    if filter not in allowed_options:
        msgs.error("PypeIt is not ready for filter = {}".format(filter))

    trans_file = resource_filename('pypeit', os.path.join('data', 'filters', 'filtercurves.fits'))
    trans = io.fits_open(trans_file)
    wave = trans[filter].data['lam']  # Angstroms
    instr = trans[filter].data['Rlam']  # Am keeping in atmospheric terms
    keep = instr > 0.
    # Parse
    wave = wave[keep]
    instr = instr[keep]

    # Return
    return wave, instr

# TODO Replace this stuff wth calls to the astropy speclite package.
def scale_in_filter(wave, flux, gpm, scale_dict):
    """
    Scale spectra to input magnitude in given filter

    scale_dict has data model:
      - 'filter' (str): name of filter
      - 'mag' (float): magnitude
      - 'mag_type' (str, optional): type of magnitude.  Assumed 'AB'
      - 'masks' (list, optional): Wavelength ranges to mask in calculation

    Args:
        wave (`numpy.ndarray`_):
        flux (`numpy.ndarray`_):
        gpm (`numpy.ndarray`_):
            True is good
        scale_dict (dict like):
            Usually is a Coadd1DPar() object
            Requires mag_type, filter, filter_mag, and filter_mask

    Returns:
        float: scale value for the flux, i.e. newflux = flux * scale
    """

    # Mask further?
    if scale_dict['filter_mask'] is not None:
        # Funny formatting
        if isinstance(scale_dict['filter_mask'], str):
            regions = scale_dict['filter_mask'].split(',')
        else:
            regions = scale_dict['filter_mask']
        for region in regions:
            mask = region.split(':')
            gpm[(wave > float(mask[0])) & (wave < float(mask[1]))] = False
    mag_type = scale_dict['mag_type']

    # Parse the spectrum
    wave = wave[gpm]
    flux = flux[gpm]

    # Grab the instrument response function
    msgs.info("Integrating spectrum in filter: {}".format(scale_dict['filter']))
    fwave, trans = load_filter_file(scale_dict['filter'])
    tfunc = interpolate.interp1d(fwave, trans, bounds_error=False, fill_value=0.)

    # TODO this expression below is incorrect for irregular gridded wavelengths. FIX
    # Convolve
    allt = tfunc(wave)
    wflam = np.sum(flux*allt)/np.sum(allt)* PYPEIT_FLUX_SCALE*units.erg/units.s/units.cm**2/units.AA

    mean_wv = np.sum(fwave*trans)/np.sum(trans) * units.AA

    #
    if mag_type == 'AB':
        # Convert flam to AB magnitude
        fnu = wflam * mean_wv**2 / constants.c
        # Apparent AB
        AB = -2.5 * np.log10(fnu.to('erg/s/cm**2/Hz').value) - 48.6
        # Scale factor
        Dm = AB - scale_dict['filter_mag']
        scale = np.power(10.0,(Dm/2.5))
        msgs.info("Scaling spectrum by {}".format(scale))
    else:
        msgs.error("Bad magnitude type")

    return scale

