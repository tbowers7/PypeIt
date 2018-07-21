from __future__ import (print_function, absolute_import, division, unicode_literals)

import copy

import inspect

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import gridspec, font_manager

from scipy import interpolate
from pkg_resources import resource_filename

from astropy import units
from astropy.coordinates import SkyCoord, solar_system, EarthLocation, ICRS
from astropy.coordinates import UnitSphericalRepresentation, CartesianRepresentation
from astropy.time import Time

from linetools.spectra import xspectrum1d

from pypeit import msgs
from pypeit.core import arc
from pypeit.core import qa
from pypeit import utils

from pypeit import debugger

def flex_shift(obj_skyspec, arx_skyspec, mxshft=None):
    """ Calculate shift between object sky spectrum and archive sky spectrum

    Parameters
    ----------
    obj_skyspec
    arx_skyspec

    Returns
    -------
    flex_dict
    """
    if mxshft is None:
        mxshft = 20
    #Determine the brightest emission lines
    msgs.warn("If we use Paranal, cut down on wavelength early on")
    arx_amp, arx_cent, arx_wid, arx_w, arx_yprep = arc.detect_lines(arx_skyspec.flux.value)
    obj_amp, obj_cent, obj_wid, obj_w, obj_yprep = arc.detect_lines(obj_skyspec.flux.value)

    #Keep only 5 brightest amplitude lines (xxx_keep is array of indices within arx_w of the 5 brightest)
    arx_keep = np.argsort(arx_amp[arx_w])[-5:]
    obj_keep = np.argsort(obj_amp[obj_w])[-5:]

    #Calculate wavelength (Angstrom per pixel)
    arx_disp = np.append(arx_skyspec.wavelength.value[1]-arx_skyspec.wavelength.value[0],
                         arx_skyspec.wavelength.value[1:]-arx_skyspec.wavelength.value[:-1])
    #arx_disp = (np.amax(arx_sky.wavelength.value)-np.amin(arx_sky.wavelength.value))/arx_sky.wavelength.size
    obj_disp = np.append(obj_skyspec.wavelength.value[1]-obj_skyspec.wavelength.value[0],
                         obj_skyspec.wavelength.value[1:]-obj_skyspec.wavelength.value[:-1])
    #obj_disp = (np.amax(obj_sky.wavelength.value)-np.amin(obj_sky.wavelength.value))/obj_sky.wavelength.size

    #Calculate resolution (lambda/delta lambda_FWHM)..maybe don't need this? can just use sigmas
    arx_idx = (arx_cent+0.5).astype(np.int)[arx_w][arx_keep]   # The +0.5 is for rounding
    arx_res = arx_skyspec.wavelength.value[arx_idx]/\
              (arx_disp[arx_idx]*(2*np.sqrt(2*np.log(2)))*arx_wid[arx_w][arx_keep])
    obj_idx = (obj_cent+0.5).astype(np.int)[obj_w][obj_keep]   # The +0.5 is for rounding
    obj_res = obj_skyspec.wavelength.value[obj_idx]/ \
              (obj_disp[obj_idx]*(2*np.sqrt(2*np.log(2)))*obj_wid[obj_w][obj_keep])
    #obj_res = (obj_sky.wavelength.value[0]+(obj_disp*obj_cent[obj_w][obj_keep]))/(
    #    obj_disp*(2*np.sqrt(2*np.log(2)))*obj_wid[obj_w][obj_keep])
    if not np.all(np.isfinite(obj_res)):
        msgs.error("Failure to measure the resolution of the object spectrum.  Probably an error in the wavelength image.")
    msgs.info("Resolution of Archive={:g} and Observation={:g}".format(
        np.median(arx_res), np.median(obj_res)))

    #Determine sigma of gaussian for smoothing
    arx_sig2 = (arx_disp[arx_idx]*arx_wid[arx_w][arx_keep])**2.
    obj_sig2 = (obj_disp[obj_idx]*obj_wid[obj_w][obj_keep])**2.

    arx_med_sig2 = np.median(arx_sig2)
    obj_med_sig2 = np.median(obj_sig2)

    if obj_med_sig2 >= arx_med_sig2:
        smooth_sig = np.sqrt(obj_med_sig2-arx_med_sig2)  # Ang
        smooth_sig_pix = smooth_sig / np.median(arx_disp[arx_idx])
        arx_skyspec = arx_skyspec.gauss_smooth(smooth_sig_pix*2*np.sqrt(2*np.log(2)))
    else:
        msgs.warn("Prefer archival sky spectrum to have higher resolution")
        smooth_sig_pix = 0.
        msgs.warn("New Sky has higher resolution than Archive.  Not smoothing")
        #smooth_sig = np.sqrt(arx_med_sig**2-obj_med_sig**2)

    #Determine region of wavelength overlap
    min_wave = max(np.amin(arx_skyspec.wavelength.value), np.amin(obj_skyspec.wavelength.value))
    max_wave = min(np.amax(arx_skyspec.wavelength.value), np.amax(obj_skyspec.wavelength.value))

    #Smooth higher resolution spectrum by smooth_sig (flux is conserved!)
#    if np.median(obj_res) >= np.median(arx_res):
#        msgs.warn("New Sky has higher resolution than Archive.  Not smoothing")
        #obj_sky_newflux = ndimage.gaussian_filter(obj_sky.flux, smooth_sig)
#    else:
        #tmp = ndimage.gaussian_filter(arx_sky.flux, smooth_sig)
#        arx_skyspec = arx_skyspec.gauss_smooth(smooth_sig_pix*2*np.sqrt(2*np.log(2)))
        #arx_sky.flux = ndimage.gaussian_filter(arx_sky.flux, smooth_sig)

    # Define wavelengths of overlapping spectra
    keep_idx = np.where((obj_skyspec.wavelength.value>=min_wave) &
                         (obj_skyspec.wavelength.value<=max_wave))[0]
    #keep_wave = [i for i in obj_sky.wavelength.value if i>=min_wave if i<=max_wave]

    #Rebin both spectra onto overlapped wavelength range
    if len(keep_idx) <= 50:
        msgs.error("Not enough overlap between sky spectra")
    else: #rebin onto object ALWAYS
        keep_wave = obj_skyspec.wavelength[keep_idx]
        arx_skyspec = arx_skyspec.rebin(keep_wave)
        obj_skyspec = obj_skyspec.rebin(keep_wave)
        # Trim edges (rebinning is junk there)
        arx_skyspec.data['flux'][0,:2] = 0.
        arx_skyspec.data['flux'][0,-2:] = 0.
        obj_skyspec.data['flux'][0,:2] = 0.
        obj_skyspec.data['flux'][0,-2:] = 0.

    #   Normalize spectra to unit average sky count
    norm = np.sum(obj_skyspec.flux.value)/obj_skyspec.npix
    obj_skyspec.flux = obj_skyspec.flux / norm
    norm2 = np.sum(arx_skyspec.flux.value)/arx_skyspec.npix
    arx_skyspec.flux = arx_skyspec.flux / norm2
    if (norm < 0.):
        msgs.warn("Bad normalization of object in flexure algorithm")
        msgs.warn("Will try the median")
        norm = np.median(obj_skyspec.flux.value)
        if (norm < 0.):
            msgs.error("Improper sky spectrum for flexure.  Is it too faint??")
    if (norm2 < 0.):
        msgs.error("Bad normalization of archive in flexure. You are probably using wavelengths well beyond the archive.")

    #deal with bad pixels
    msgs.work("Need to mask bad pixels")

    #deal with underlying continuum
    msgs.work("Consider taking median first [5 pixel]")
    everyn = obj_skyspec.npix // 20
    bspline_par = dict(everyn=everyn)
    mask, ct = utils.robust_polyfit(obj_skyspec.wavelength.value, obj_skyspec.flux.value, 3, function='bspline',
                                  sigma=3., bspline_par=bspline_par)# everyn=everyn)
    obj_sky_cont = utils.func_val(ct, obj_skyspec.wavelength.value, 'bspline')
    obj_sky_flux = obj_skyspec.flux.value - obj_sky_cont
    mask, ct_arx = utils.robust_polyfit(arx_skyspec.wavelength.value, arx_skyspec.flux.value, 3, function='bspline',
                                      sigma=3., bspline_par=bspline_par)# everyn=everyn)
    arx_sky_cont = utils.func_val(ct_arx, arx_skyspec.wavelength.value, 'bspline')
    arx_sky_flux = arx_skyspec.flux.value - arx_sky_cont

    # Consider shaprness filtering (e.g. LowRedux)
    msgs.work("Consider taking median first [5 pixel]")

    #Cross correlation of spectra
    #corr = np.correlate(arx_skyspec.flux, obj_skyspec.flux, "same")
    corr = np.correlate(arx_sky_flux, obj_sky_flux, "same")

    #Create array around the max of the correlation function for fitting for subpixel max
    # Restrict to pixels within maxshift of zero lag
    lag0 = corr.size//2
    #mxshft = settings.argflag['reduce']['flexure']['maxshift']
    max_corr = np.argmax(corr[lag0-mxshft:lag0+mxshft]) + lag0-mxshft
    subpix_grid = np.linspace(max_corr-3., max_corr+3., 7.)

    #Fit a 2-degree polynomial to peak of correlation function
    fit = utils.func_fit(subpix_grid, corr[subpix_grid.astype(np.int)], 'polynomial', 2)
    max_fit = -0.5*fit[1]/fit[2]

    #Calculate and apply shift in wavelength
    shift = float(max_fit)-lag0
    msgs.info("Flexure correction of {:g} pixels".format(shift))
    #model = (fit[2]*(subpix_grid**2.))+(fit[1]*subpix_grid)+fit[0]

    #if msgs._debug['flexure']:
    #    debugger.plot1d(arx_skyspec.wavelength, arx_sky_flux, xtwo=np.roll(obj_skyspec.wavelength,int(-1*shift)), ytwo=obj_sky_flux)
    #    #debugger.xplot(arx_sky.wavelength, arx_sky.flux, xtwo=np.roll(obj_sky.wavelength.value,9), ytwo=obj_sky.flux*100)
    #    debugger.set_trace()

    flex_dict = dict(polyfit=fit, shift=shift, subpix=subpix_grid,
                     corr=corr[subpix_grid.astype(np.int)],
                     sky_spec=obj_skyspec,
                     arx_spec=arx_skyspec,
                     corr_cen=corr.size/2, smooth=smooth_sig_pix)
    # Return
    return flex_dict

# This has been moved into the spectrographs class
#def flexure_archive(spectrograph=None, skyspec_fil=None):
#    """  Load archived sky spectrum
#    """
#    #   latitude = settings.spect['mosaic']['latitude']
#    #   longitude = settings.spect['mosaic']['longitude']
#    root = resource_filename('pypeit', 'data/sky_spec/')
#    if skyspec_fil is None: #settings.argflag['reduce']['flexure']['spectrum'] is None:
#        # Red or blue?
#        if spectrograph in ['shane_kast_blue']:
#            skyspec_fil = 'sky_kastb_600.fits'
#        elif spectrograph in ['keck_lris_blue']:
#            skyspec_fil = 'sky_LRISb_600.fits'
#        else:
#            skyspec_fil = 'paranal_sky.fits'
#    #
#    msgs.info("Using {:s} file for Sky spectrum".format(skyspec_fil))
#    arx_sky = xspectrum1d.XSpectrum1D.from_file(root+skyspec_fil)
#    #hdu = fits.open(root+'/data/sky_spec/'+skyspec_fil)
#    #archive_wave = hdu[0].data
#    #archive_flux = hdu[1].data
#    #arx_sky = xspectrum1d.XSpectrum1D.from_tuple((archive_wave, archive_flux))
#    # Return
#    return skyspec_fil, arx_sky


'''
def flexure_slit():
    """Correct wavelength down slit center for flexure

    Parameters:
    ----------
    slf :
    det : int
    """
    debugger.set_trace()  # THIS METHOD IS NOT BEING USED THESE DAYS
    # Load Archive
    skyspec_fil, arx_sky = flexure_archive()

    # Extract
    censpec_wv = arextract.boxcar_cen(slf, det, slf._mswave[det-1])
    censpec_fx = arextract.boxcar_cen(slf, det, slf._bgframe[det-1])
    cen_sky = xspectrum1d.XSpectrum1D.from_tuple((censpec_wv, censpec_fx))
    # Find shift
    fdict = flex_shift(slf, det, cen_sky, arx_sky)
    msgs.work("Flexure shift = {:g} down slit center".format(fdict['shift']))
    # Refit
    #  What if xfit shifts outside of 0-1?
    xshift = fdict['shift']/(slf._msarc[det-1].shape[0]-1)
    mask, fit = utils.robust_polyfit(np.array(slf._wvcalib[det-1]['xfit'])+xshift,
                                       np.array(slf._wvcalib[det-1]['yfit']),
                                       len(slf._wvcalib[det-1]['fitc']),
                                       function=slf._wvcalib[det-1]['function'], sigma=slf._wvcalib[det-1]['nrej'], minv=slf._wvcalib[det-1]['fmin'], maxv=slf._wvcalib[det-1]['fmax'])
    # Update wvcalib
    slf._wvcalib[det-1]['shift'] = fdict['shift']  # pixels
    slf._wvcalib[det-1]['fitc'] = fit
    msgs.work("Add another QA for wavelengths?")
    # Update mswave
    wv_calib = slf._wvcalib[det-1]
    slf._mswave[det-1] = rutils.func_val(wv_calib['fitc'], slf._tilts[det-1], wv_calib['function'], minv=wv_calib['fmin'], maxv=wv_calib['fmax'])
    # Write to Masters?  Not for now
    # For QA (kludgy..)
    censpec_wv = arextract.boxcar_cen(slf, det, slf._mswave[det-1])
    fdict['sky_spec'] = xspectrum1d.XSpectrum1D.from_tuple((censpec_wv, censpec_fx))
    flex_dict = dict(polyfit=[], shift=[], subpix=[], corr=[],
                     corr_cen=[], spec_file=skyspec_fil, smooth=[],
                     arx_spec=[], sky_spec=[])
    #debugger.set_trace()
    #debugger.xplot(censpec_wv, censpec_fx, xtwo=fdict['arx_spec'].wavelength, ytwo=fdict['arx_spec'].flux*50)
    for key in ['polyfit', 'shift', 'subpix', 'corr', 'corr_cen', 'smooth', 'sky_spec', 'arx_spec']:
        flex_dict[key].append(fdict[key])
    return flex_dict
'''


def flexure_obj(specobjs, maskslits, method, sky_spectrum, sky_file=None, mxshft=None):
    """Correct wavelengths for flexure, object by object

    Parameters:
    ----------
    method : str
      'boxcar' -- Recommneded
      'slitpix' --

    Returns:
    ----------
    flex_list: list
      list of dicts containing flexure results
        Aligned with specobjs
        Filled with a basically empty dict if the slit is skipped or there is no object

    """
    msgs.work("Consider doing 2 passes in flexure as in LowRedux")
    # Load Archive
#    skyspec_fil, arx_sky = flexure_archive(spectrograph=spectrograph, skyspec_fil=skyspec_fil)

    # Loop on objects
    flex_list = []

    gdslits = np.where(~maskslits)[0]
    for sl in range(len(specobjs)):
        # Reset
        flex_dict = dict(polyfit=[], shift=[], subpix=[], corr=[],
                         corr_cen=[], spec_file=sky_file, smooth=[],
                         arx_spec=[], sky_spec=[])
        if sl not in gdslits:
            flex_list.append(flex_dict.copy())
            continue
        msgs.info("Working on flexure in slit (if an object was detected): {:d}".format(sl))
        for specobj in specobjs[sl]:  # for convenience
            if specobj is None:
                continue

            # Using boxcar
            if method in ['boxcar', 'slitcen']:
                sky_wave = specobj.boxcar['wave'].to('AA').value
                sky_flux = specobj.boxcar['sky']
            else:
                msgs.error("Not ready for this flexure method: {}".format(method))

            # Generate 1D spectrum for object
            obj_sky = xspectrum1d.XSpectrum1D.from_tuple((sky_wave, sky_flux))

            # Calculate the shift
            fdict = flex_shift(obj_sky, sky_spectrum, mxshft=mxshft)

            # Simple interpolation to apply
            npix = len(sky_wave)
            x = np.linspace(0., 1., npix)
            # Apply
            for attr in ['boxcar', 'optimal']:
                if not hasattr(specobj, attr):
                    continue
                if 'wave' in getattr(specobj, attr).keys():
                    msgs.info("Applying flexure correction to {0:s} extraction for object:".format(attr) +
                              msgs.newline() + "{0:s}".format(str(specobj)))
                    f = interpolate.interp1d(x, sky_wave, bounds_error=False, fill_value="extrapolate")
                    getattr(specobj, attr)['wave'] = f(x+fdict['shift']/(npix-1))*units.AA
            # Shift sky spec too
            cut_sky = fdict['sky_spec']
            x = np.linspace(0., 1., cut_sky.npix)
            f = interpolate.interp1d(x, cut_sky.wavelength.value, bounds_error=False, fill_value="extrapolate")
            twave = f(x + fdict['shift']/(cut_sky.npix-1))*units.AA
            new_sky = xspectrum1d.XSpectrum1D.from_tuple((twave, cut_sky.flux))

            # Update dict
            for key in ['polyfit', 'shift', 'subpix', 'corr', 'corr_cen', 'smooth', 'arx_spec']:
                flex_dict[key].append(fdict[key])
            flex_dict['sky_spec'].append(new_sky)
        flex_list.append(flex_dict.copy())
    return flex_list


def geomotion_calculate(fitstbl, idx, time, longitude, latitude, altitude, refframe):
    """
    Correct the wavelength calibration solution to the desired reference frame
    """
    loc = (longitude * units.deg, latitude * units.deg, altitude * units.m,)
    radec = SkyCoord(fitstbl["ra"][idx], fitstbl["dec"][idx], unit=(units.hourangle, units.deg),
                     frame='fk5')
    obstime = Time(time.value, format=time.format, scale='utc', location=loc)
    return geomotion_velocity(obstime, radec, frame=refframe)


def geomotion_correct(specobjs, maskslits, fitstbl, scidx, time, longitude, latitude, elevation,
                      refframe):
    """ Correct the wavelength of every pixel to a barycentric/heliocentric frame.

    Parameters
    ----------
    specobjs
    maskslits
    fitstbl : Table
      Containing the properties of every fits file
    scidx
    time
    settings_mosaic
    refframe

    Returns
    -------
    vel : float
      The velocity correction that should be applied to the wavelength array.
    vel_corr : float
      The relativistic velocity correction that should be multiplied by the
      wavelength array to convert each wavelength into the user-specified
      reference frame.

    """
    # Calculate
    vel = geomotion_calculate(fitstbl, scidx, time, longitude, latitude, elevation, refframe)
    vel_corr = np.sqrt((1. + vel/299792.458) / (1. - vel/299792.458))

    gdslits = np.where(~maskslits)[0]
    # Loop on slits to apply
    for sl in range(len(specobjs)):
        if sl not in gdslits:
            continue
        # Loop on objects
        for specobj in specobjs[sl]:
            if specobj is None:
                continue
            # Loop on extraction methods
            for attr in ['boxcar', 'optimal']:
                if not hasattr(specobj, attr):
                    continue
                if 'wave' in getattr(specobj, attr).keys():
                    msgs.info('Applying {0} correction to '.format(refframe)
                              + '{0} extraction for object:'.format(attr)
                              + msgs.newline() + "{0}".format(str(specobj)))
                    getattr(specobj, attr)['wave'] = getattr(specobj, attr)['wave'] * vel_corr
    # Return
    return vel, vel_corr  # Mainly for debugging


def geomotion_velocity(time, skycoord, frame="heliocentric"):
    """ Perform a barycentric/heliocentric velocity correction.

    For the correciton, this routine uses the ephemeris:  astropy.coordinates.solar_system_ephemeris.set
    For more information see `~astropy.coordinates.solar_system_ephemeris`.

    Parameters
    ----------
    time : astropy.time.Time
      The time of observation, including the location.
    skycoord: astropy.coordinates.SkyCoord
      The RA and DEC of the pointing, as a SkyCoord quantity.
    frame : str
      The reference frame that should be used for the calculation.

    Returns
    -------
    vcorr : float
      The velocity correction that should be added to the original velocity.
    """

    # Check that the RA/DEC of the object is ICRS compatible
    if not skycoord.is_transformable_to(ICRS()):
        msgs.error("Cannot transform RA/DEC of object to the ICRS")

    # Calculate ICRS position and velocity of Earth's geocenter
    ep, ev = solar_system.get_body_barycentric_posvel('earth', time)
    # Calculate GCRS position and velocity of observatory
    op, ov = time.location.get_gcrs_posvel(time)
    # ICRS and GCRS are axes-aligned. Can add the velocities
    velocity = ev + ov
    if frame == "heliocentric":
        # ICRS position and velocity of the Sun
        sp, sv = solar_system.get_body_barycentric_posvel('sun', time)
        velocity += sv

    # Get unit ICRS vector in direction of SkyCoord
    sc_cartesian = skycoord.icrs.represent_as(UnitSphericalRepresentation).represent_as(CartesianRepresentation)
    return sc_cartesian.dot(velocity).to(units.km / units.s).value


def airtovac(wave):
    """ Convert air-based wavelengths to vacuum

    Parameters:
    ----------
    wave: Quantity array
      Wavelengths 

    Returns:
    ----------
    wave: Quantity array
      Wavelength array corrected to vacuum wavelengths
    """
    # Convert to AA
    wave = wave.to(units.AA)
    wavelength = wave.value

    # Standard conversion format
    sigma_sq = (1.e4/wavelength)**2. #wavenumber squared
    factor = 1 + (5.792105e-2/(238.0185-sigma_sq)) + (1.67918e-3/(57.362-sigma_sq))
    factor = factor*(wavelength>=2000.) + 1.*(wavelength<2000.) #only modify above 2000A

    # Convert
    wavelength = wavelength*factor
    # Units
    new_wave = wavelength*units.AA
    new_wave.to(wave.unit)

    return new_wave


def vactoair(wave):
    """Convert to air-based wavelengths from vacuum

    Parameters:
    ----------
    wave: Quantity array
      Wavelengths 

    Returns:
    ----------
    wave: Quantity array
      Wavelength array corrected to air
    """
    # Convert to AA
    wave = wave.to(units.AA)
    wavelength = wave.value

    # Standard conversion format
    sigma_sq = (1.e4/wavelength)**2. #wavenumber squared
    factor = 1 + (5.792105e-2/(238.0185-sigma_sq)) + (1.67918e-3/(57.362-sigma_sq))
    factor = factor*(wavelength>=2000.) + 1.*(wavelength<2000.) #only modify above 2000A

    # Convert
    wavelength = wavelength/factor
    new_wave = wavelength*units.AA
    new_wave.to(wave.unit)

    return new_wave


def flexure_qa(specobjs, maskslits, basename, det, flex_list, slit_cen=False):
    """ QA on flexure measurement

    Parameters
    ----------
    det
    flex_list : list
      list of dict containing flexure results
    slit_cen : bool, optional
      QA on slit center instead of objects

    Returns
    -------

    """
    plt.rcdefaults()
    plt.rcParams['font.family']= 'times new roman'

    # Grab the named of the method
    method = inspect.stack()[0][3]
    #
    gdslits = np.where(~maskslits)[0]
    for sl in range(len(specobjs)):
        if sl not in gdslits:
            continue
        if specobjs[sl][0] is None:
            continue
        # Setup
        if slit_cen:
            nobj = 1
            ncol = 1
        else:
            nobj = len(specobjs[sl])
            ncol = min(3, nobj)
        #
        if nobj==0:
            continue
        nrow = nobj // ncol + ((nobj % ncol) > 0)

        # Get the flexure dictionary
        flex_dict = flex_list[sl]

        # Outfile
        outfile = qa.set_qa_filename(basename, method+'_corr', det=det,
                                       slit=specobjs[sl][0].slitid)

        plt.figure(figsize=(8, 5.0))
        plt.clf()
        gs = gridspec.GridSpec(nrow, ncol)

        # Correlation QA
        for o in range(nobj):
            ax = plt.subplot(gs[o//ncol, o % ncol])
            # Fit
            fit = flex_dict['polyfit'][o]
            xval = np.linspace(-10., 10, 100) + flex_dict['corr_cen'][o] #+ flex_dict['shift'][o]
            #model = (fit[2]*(xval**2.))+(fit[1]*xval)+fit[0]
            model = utils.func_val(fit, xval, 'polynomial')
            mxmod = np.max(model)
            ylim = [np.min(model/mxmod), 1.3]
            ax.plot(xval-flex_dict['corr_cen'][o], model/mxmod, 'k-')
            # Measurements
            ax.scatter(flex_dict['subpix'][o]-flex_dict['corr_cen'][o],
                       flex_dict['corr'][o]/mxmod, marker='o')
            # Final shift
            ax.plot([flex_dict['shift'][o]]*2, ylim, 'g:')
            # Label
            if slit_cen:
                ax.text(0.5, 0.25, 'Slit Center', transform=ax.transAxes, size='large', ha='center')
            else:
                ax.text(0.5, 0.25, '{:s}'.format(specobjs[sl][o].idx), transform=ax.transAxes, size='large', ha='center')
            ax.text(0.5, 0.15, 'flex_shift = {:g}'.format(flex_dict['shift'][o]),
                    transform=ax.transAxes, size='large', ha='center')#, bbox={'facecolor':'white'})
            # Axes
            ax.set_ylim(ylim)
            ax.set_xlabel('Lag')

        # Finish
        plt.tight_layout(pad=0.2, h_pad=0.0, w_pad=0.0)
        plt.savefig(outfile, dpi=600)
        plt.close()

        # Sky line QA (just one object)
        if slit_cen:
            o = 0
        else:
            o = 0
            specobj = specobjs[sl][o]
        sky_spec = flex_dict['sky_spec'][o]
        arx_spec = flex_dict['arx_spec'][o]

        # Sky lines
        sky_lines = np.array([3370.0, 3914.0, 4046.56, 4358.34, 5577.338, 6300.304,
                              7340.885, 7993.332, 8430.174, 8919.610, 9439.660,
                              10013.99, 10372.88])*units.AA
        dwv = 20.*units.AA
        gdsky = np.where((sky_lines > sky_spec.wvmin) & (sky_lines < sky_spec.wvmax))[0]
        if len(gdsky) == 0:
            msgs.warn("No sky lines for Flexure QA")
            return
        if len(gdsky) > 6:
            idx = np.array([0, 1, len(gdsky)//2, len(gdsky)//2+1, -2, -1])
            gdsky = gdsky[idx]

        # Outfile
        outfile = qa.set_qa_filename(basename, method+'_sky', det=det,
                                       slit=specobjs[sl][0].slitid)
        # Figure
        plt.figure(figsize=(8, 5.0))
        plt.clf()
        nrow, ncol = 2, 3
        gs = gridspec.GridSpec(nrow, ncol)
        if slit_cen:
            plt.suptitle('Sky Comparison for Slit Center', y=1.05)
        else:
            plt.suptitle('Sky Comparison for {:s}'.format(specobj.idx), y=1.05)

        for ii, igdsky in enumerate(gdsky):
            skyline = sky_lines[igdsky]
            ax = plt.subplot(gs[ii//ncol, ii % ncol])
            # Norm
            pix = np.where(np.abs(sky_spec.wavelength-skyline) < dwv)[0]
            f1 = np.sum(sky_spec.flux[pix])
            f2 = np.sum(arx_spec.flux[pix])
            norm = f1/f2
            # Plot
            ax.plot(sky_spec.wavelength[pix], sky_spec.flux[pix], 'k-', label='Obj',
                    drawstyle='steps-mid')
            pix2 = np.where(np.abs(arx_spec.wavelength-skyline) < dwv)[0]
            ax.plot(arx_spec.wavelength[pix2], arx_spec.flux[pix2]*norm, 'r-', label='Arx',
                    drawstyle='steps-mid')
            # Axes
            ax.xaxis.set_major_locator(plt.MultipleLocator(dwv.value))
            ax.set_xlabel('Wavelength')
            ax.set_ylabel('Counts')

        # Legend
        plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
                   handletextpad=0.3, fontsize='small', numpoints=1)

        # Finish
        plt.savefig(outfile, dpi=800)
        plt.close()
        #plt.close()

    plt.rcdefaults()

    return
