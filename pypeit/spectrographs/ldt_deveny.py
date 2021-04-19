"""
Module for LDT/DeVeny specific methods.

.. include:: ../include/links.rst
"""
from pkg_resources import resource_filename

import numpy as np

from astropy.time import Time

from pypeit import msgs
from pypeit import telescopes
from pypeit import io
from pypeit.core import framematch
from pypeit.spectrographs import spectrograph
from pypeit.core import parse
from pypeit.images import detector_container


class LDTDeVenySpectrograph(spectrograph.Spectrograph):
    """
    Child to handle LDT/DeVeny specific code
    """
    ndet = 1
    name = 'ldt_deveny'
    telescope = telescopes.LDTTelescopePar()
    camera = 'deveny'
    comment = 'LDT DeVeny Optical Spectrograph'

    # Parameters equal to the PypeIt defaults, shown here for completeness
    # supported = False
    # pypeline = 'MultiSlit'

    def get_detector_par(self, hdu, det):
        """
        Return metadata for the selected detector.

        Args:
            hdu (`astropy.io.fits.HDUList`_):
                The open fits file with the raw image of interest.
            det (:obj:`int`):
                1-indexed detector number.

        Returns:
            :class:`~pypeit.images.detector_container.DetectorContainer`:
            Object with the detector metadata.
        """
        header = hdu[0].header
        
        # Binning
        binning = self.get_meta_value(self.get_headarr(hdu), 'binning')  # Could this be detector dependent??

        # Detector
        detector_dict = dict(
            binning         = binning,
            det             = 1,
            dataext         = 0,
            specaxis        = 1,
            specflip        = True,
            spatflip        = False,
            platescale      = 0.34,
            darkcurr        = 0.0,    # Still need to measure this.
            saturation      = 65535.,
            nonlinear       = 1.0,
            mincounts       = -1e10,
            numamplifiers   = 1,
            gain            = np.atleast_1d(header['GAIN']),
            ronoise         = np.atleast_1d(header['RDNOISE']),
            # Data & Overscan Sections deliberately cut in spatial direction
            #   to exclue the edges of the slit and weird light spillover.
            datasec         = np.atleast_1d('[25:495,54:2096]'),
            oscansec        = np.atleast_1d('[25:495,2101:2144]')
            )
        return detector_container.DetectorContainer(**detector_dict)

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the ``PypeIt``-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}
        
        # Required (core)
        self.meta['ra'] = dict(ext=0, card='RA')
        self.meta['dec'] = dict(ext=0, card='DEC')
        self.meta['target'] = dict(ext=0, card='OBJNAME')
        self.meta['dispname'] = dict(card=None, compound=True)
        self.meta['decker'] = dict(card=None, compound=True)
        self.meta['binning'] = dict(card=None, compound=True)
        self.meta['mjd'] = dict(card=None, compound=True)
        self.meta['airmass'] = dict(ext=0, card='AIRMASS')
        self.meta['exptime'] = dict(ext=0, card='EXPTIME')
        
        # Extras for config and frametyping
        self.meta['idname'] = dict(ext=0, card='OBSTYPE')
        self.meta['dispangle'] = dict(ext=0, card='GRANGLE', rtol=1e-3)
        self.meta['filter1'] = dict(card=None, compound=True)
        self.meta['slitwid'] = dict(ext=0, card='SLITASEC')
        self.meta['lampstat01'] = dict(card=None, compound=True)
        

    def compound_meta(self, headarr, meta_key):
        """
        Methods to generate metadata requiring interpretation of the header
        data, instead of simply reading the value of a header card.

        Args:
            headarr (:obj:`list`):
                List of `astropy.io.fits.Header`_ objects.
            meta_key (:obj:`str`):
                Metadata keyword to construct.

        Returns:
            object: Metadata value read from the header(s).
        """
        if meta_key == 'binning':
            """
            Binning in lois headers is space-separated rather than comma-separated.
            """
            binspec, binspatial = headarr[0]['CCDSUM'].split()
            binning = parse.binning2string(binspec, binspatial)
            return binning
        
        elif meta_key == 'mjd':
            """
            Use astropy to convert 'DATE-OBS' into a mjd.
            """
            ttime = Time(headarr[0]['DATE-OBS'], format='isot')
            return ttime.mjd
        
        elif meta_key == 'lampstat01':
            """
            The spectral comparison lamps turned on are listed in `LAMPCAL`, but
            if no lamps are on, then this string is blank.  Return either the
            populated `LAMPCAL` string, or 'off' to ensure a positive entry for
            `lampstat01`.
            """
            lampcal = headarr[0]['LAMPCAL'].strip()
            if lampcal == '':
                return 'off'
            else:
                return lampcal
        
        elif meta_key == 'dispname':
            """
            Convert older FITS keyword GRATING (gpmm/blaze) into the newer
            Grating ID names (DVx) for easier identification of disperser.
            """
            gratings = {"DV1": "150/5000", "DV2": "300/4000", "DV3": "300/6750",
                        "DV4": "400/8500", "DV5": "500/5500", "DV6": "600/4900",
                        "DV7": "600/6750", "DV8": "831/8000", "DV9": "1200/5000",
                        "DV10": "2160/5000", "DVxx": "UNKNOWN"}
            ids = list(gratings.keys())
            kwds = list(gratings.values())
            idx = kwds.index(headarr[0]['GRATING'])
            return f"{ids[idx]} ({kwds[idx]})"
        
        elif meta_key == 'decker':
            """
            Provide a stub for future inclusion of a decker on LDT/DeVeny.
            """
            if "DECKER" in headarr[0].keys():
                decker = headarr[0]['DECKER']
            else:
                decker = "None"
            return decker

        elif meta_key == 'filter1':
            """
            Remove the parenthetical knob position to leave just the filter name
            """
            return headarr[0]['FILTREAR'].split()[0]

        else:
            msgs.error("Not ready for this compound meta for LDT/DeVeny")

    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.
        
        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of ``PypeIt`` methods.
        """
        par = super().default_pypeit_par()

        # Calibration Parameters
        # Turn off illumflat and darkimage; turn on overscan
        set_use = dict(use_illumflat=False, use_darkimage=False, use_overscan=True)
        par.reset_all_processimages_par(**set_use)

        # Need to specify this for long-slit data
        par['calibrations']['slitedges']['sync_predict'] = 'nearest'
        par['calibrations']['slitedges']['bound_detector'] = True
    
        # Make a bad pixel mask
        par['calibrations']['bpm_usebias'] = True
        # Set pixel flat combination method
        par['calibrations']['pixelflatframe']['process']['combine'] = 'median'

        # Wavelengths
        # Change the wavelength calibration method
        par['calibrations']['wavelengths']['method'] = 'holy-grail'
        # Include the lamps available on DeVeny
        par['calibrations']['wavelengths']['lamps'] = ['NeI', 'ArI', 'ArII', 'CdI', 'HgI']
        # 1D wavelength solution
        par['calibrations']['wavelengths']['rms_threshold'] = 0.5
        par['calibrations']['wavelengths']['sigdetect'] = 5.
        par['calibrations']['wavelengths']['fwhm']= 3.0
        par['calibrations']['wavelengths']['n_first'] = 3
        par['calibrations']['wavelengths']['n_final'] = 5
        par['calibrations']['wavelengths']['rms_threshold'] = 0.2
        par['calibrations']['wavelengths']['nlocal_cc'] = 13
        
        
        # # Do not flux calibrate
        # par['fluxcalib'] = None
        # # Set the default exposure time ranges for the frame typing
        # par['calibrations']['biasframe']['exprng'] = [None, 1]
        # par['calibrations']['darkframe']['exprng'] = [999999, None]     # No dark frames
        # par['calibrations']['pinholeframe']['exprng'] = [999999, None]  # No pinhole frames
        # par['calibrations']['arcframe']['exprng'] = [None, 120]
        # par['calibrations']['standardframe']['exprng'] = [None, 120]
        # par['scienceframe']['exprng'] = [90, None]

        # # Extraction
        # par['reduce']['skysub']['bspline_spacing'] = 0.8
        # par['reduce']['skysub']['no_poly'] = True
        # par['reduce']['skysub']['bspline_spacing'] = 0.6
        # par['reduce']['skysub']['joint_fit'] = False
        # par['reduce']['skysub']['global_sky_std']  = False

        par['reduce']['extraction']['sn_gauss'] = 4.0
        par['reduce']['findobj']['sig_thresh'] = 5.0
        par['reduce']['skysub']['sky_sigrej'] = 5.0
        par['reduce']['findobj']['find_trim_edge'] = [5,5]

        # # cosmic ray rejection parameters for science frames
        # par['scienceframe']['process']['sigclip'] = 5.0
        # par['scienceframe']['process']['objlim'] = 2.0

        # Sensitivity function parameters
        par['sensfunc']['polyorder'] = 7

        # # Do not correct for flexure
        # par['flexure']['spec_method'] = 'skip'

        return par

    def bpm(self, filename, det, shape=None, msbias=None):
        """
        Generate a default bad-pixel mask.

        Even though they are both optional, either the precise shape for
        the image (``shape``) or an example file that can be read to get
        the shape (``filename`` using :func:`get_image_shape`) *must* be
        provided.

        Args:
            filename (:obj:`str` or None):
                An example file to use to get the image shape.
            det (:obj:`int`):
                1-indexed detector number to use when getting the image
                shape from the example file.
            shape (tuple, optional):
                Processed image shape
                Required if filename is None
                Ignored if filename is not None
            msbias (`numpy.ndarray`_, optional):
                Master bias frame used to identify bad pixels

        Returns:
            `numpy.ndarray`_: An integer array with a masked value set
            to 1 and an unmasked value set to 0.  All values are set to
            0.
        """

        # Call the base-class method to generate the empty bpm
        bpm_img = super().bpm(filename, det, shape=shape, msbias=msbias)

        if det == 1:
            msgs.info("Using hard-coded BPM for DeVeny")

            bpm_img[:, -1] = 1

        else:
            msgs.error(f"Invalid detector number, {det}, for LDT/DeVeny (only one detector).")

        return bpm_img

    def configuration_keys(self):
        """
        Return the metadata keys that define a unique instrument
        configuration.

        This list is used by :class:`~pypeit.metadata.PypeItMetaData` to
        identify the unique configurations among the list of frames read
        for a given reduction.

        Returns:
            :obj:`list`: List of keywords of data pulled from file headers
            and used to constuct the :class:`~pypeit.metadata.PypeItMetaData`
            object.
        """
        return ['dispname', 'dispangle', 'filter1']

    def check_frame_type(self, ftype, fitstbl, exprng=None):
        """
        Check for frames of the provided type.

        Args:
            ftype (:obj:`str`):
                Type of frame to check. Must be a valid frame type; see
                frame-type :ref:`frame_type_defs`.
            fitstbl (`astropy.table.Table`_):
                The table with the metadata for one or more frames to check.
            exprng (:obj:`list`, optional):
                Range in the allowed exposure time for a frame of type
                ``ftype``. See
                :func:`pypeit.core.framematch.check_frame_exptime`.

        Returns:
            `numpy.ndarray`_: Boolean array with the flags selecting the
            exposures in ``fitstbl`` that are ``ftype`` type frames.
        """
        good_exp = framematch.check_frame_exptime(fitstbl['exptime'], exprng)
        if ftype in ['bias']:
            return (fitstbl['idname'] == 'BIAS')
        if ftype in ['arc', 'tilt']:
            # FOCUS frames should have frametype None
            return good_exp & (fitstbl['lampstat01'] != 'off') & (fitstbl['idname'] != 'FOCUS')
        if ftype in ['trace', 'pixelflat']:
            return good_exp & (fitstbl['idname'] == 'DOME FLAT') & (fitstbl['lampstat01'] == 'off')
        if ftype in ['illumflat']:
            return good_exp & (fitstbl['idname'] == 'SKY FLAT') & (fitstbl['lampstat01'] == 'off')
        if ftype in ['science', 'standard']:
            return good_exp & (fitstbl['idname'] == 'OBJECT') & (fitstbl['lampstat01'] == 'off')
        if ftype in ['dark']:
            return good_exp & (fitstbl['idname'] == 'DARK') & (fitstbl['lampstat01'] == 'off')
        if ftype in ['pinhole','align']:
            # Don't types pinhole or align frames
            return np.zeros(len(fitstbl), dtype=bool)
        msgs.warn('Cannot determine if frames are of type {0}.'.format(ftype))
        return np.zeros(len(fitstbl), dtype=bool)

    def pypeit_file_keys(self):
        """
        Define the list of keys to be output into a standard ``PypeIt`` file.

        Returns:
            :obj:`list`: The list of keywords in the relevant
            :class:`~pypeit.metadata.PypeItMetaData` instance to print to the
            :ref:`pypeit_file`.
        """
        return super().pypeit_file_keys() + ['slitwid','lampstat01']

    def config_specific_par(self, scifile, inp_par=None):
        """
        Modify the ``PypeIt`` parameters to hard-wired values used for
        specific instrument configurations.

        Args:
            scifile (:obj:`str`):
                File to use when determining the configuration and how
                to adjust the input parameters.
            inp_par (:class:`~pypeit.par.parset.ParSet`, optional):
                Parameter set used for the full run of PypeIt.  If None,
                use :func:`default_pypeit_par`.

        Returns:
            :class:`~pypeit.par.parset.ParSet`: The PypeIt parameter set
            adjusted for configuration specific parameter values.
        """
        # Start with instrument wide
        par = super().config_specific_par(scifile, inp_par=inp_par)
    
        # Wavelength calibrations
        if self.get_meta_value(scifile, 'dispname') == 'DV4 (400/8000)':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'ldt_deveny_DV4.fits'

        return par



