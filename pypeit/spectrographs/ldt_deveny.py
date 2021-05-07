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
            darkcurr        = 0.0,      # Still need to measure this.
            saturation      = 65535.,
            nonlinear       = 1.0,      # Still need to measure this.
            mincounts       = -1e10,
            numamplifiers   = 1,
            gain            = np.atleast_1d(header['GAIN']),
            ronoise         = np.atleast_1d(header['RDNOISE']),
            # Data & Overscan Sections -- Edge tracing can handle slit edges
            datasec         = np.atleast_1d('[5:512,54:2096]'),
            oscansec        = np.atleast_1d('[5:512,2101:2144]')
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
        # Turn off illumflat -- other defaults OK (as of v1.4.1)
        set_use = dict(use_illumflat=False)
        par.reset_all_processimages_par(**set_use)

        # Use median combine (rather than weighted mean) for all cals
        par['calibrations']['biasframe']['process']['combine'] = 'median'
        par['calibrations']['arcframe']['process']['combine'] = 'median'
        par['calibrations']['tiltframe']['process']['combine'] = 'median'
        par['calibrations']['pixelflatframe']['process']['combine'] = 'median'
        par['calibrations']['illumflatframe']['process']['combine'] = 'median'

        # For science, standard, and sky frames, combine using weighted mean (Default)
        #par['calibrations']['standardframe']['process']['combine'] = 'weightmean'
        #par['calibrations']['skyframe']['process']['combine'] = 'weightmean'
        #par['scienceframe']['process']['combine'] = 'weightmean'

        # Make a bad pixel mask
        par['calibrations']['bpm_usebias'] = True

        # Wavelength Calibration Parameters
        # Include all the lamps available on DeVeny
        par['calibrations']['wavelengths']['lamps'] = ['NeI', 'ArI', 'ArII', 'CdI', 'HgI']
        #par['calibrations']['wavelengths']['method'] = 'reidentify'
        # These are changes from defaults from another spectrograph...
        # TODO: Not sure if we will need to adjust these at some point
        #par['calibrations']['wavelengths']['n_first'] = 3  # Default: 2
        #par['calibrations']['wavelengths']['n_final'] = 5  # Default: 4
        #par['calibrations']['wavelengths']['nlocal_cc'] = 13  # Default: 11
        par['calibrations']['wavelengths']['fwhm']= 3.0  # Default: 4.0
        par['calibrations']['wavelengths']['rms_threshold'] = 0.5  # Default: 0.15
        par['calibrations']['wavelengths']['sigdetect'] = 10.  # Default: 5.0
        # Needed to address ISSUE #1155 when non-echelle spectrographs use
        #  the wavelength calibration method `reidentify`.
        par['calibrations']['wavelengths']['ech_fix_format'] = False

        # Slit-edge setting for long-slit data
        par['calibrations']['slitedges']['bound_detector'] = True
        par['calibrations']['slitedges']['sync_predict'] = 'nearest'
    
        # Set the default exposure time ranges for the frame typing
        par['calibrations']['biasframe']['exprng'] = [None, 1]
        par['calibrations']['arcframe']['exprng'] = [None, 60]
        
        # Reduction and Extraction Parameters
        par['reduce']['findobj']['sig_thresh'] = 5.0   # Default: [10.0]
        
        # Sensitivity function parameters
        par['sensfunc']['polyorder'] = 7  # Default: 5
        
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

        # msgs.info("Using hard-coded BPM for DeVeny")
        # bpm_img[:, 0] = 1

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
    
        # Set parameters based on grating used:
        grating = self.get_meta_value(scifile, 'dispname')
        if grating == 'DV1 (150/5000)':
            pass
        elif grating == 'DV2 (300/4000)':
            pass
        elif grating == 'DV3 (300/6750)':
            pass
        elif grating == 'DV4 (400/8000)':
            # Wavelength calibrations
            par['calibrations']['wavelengths']['reid_arxiv'] = 'ldt_deveny_DV4.fits'
        elif grating == 'DV5 (500/5500)':
            pass
        elif grating == 'DV6 (600/4900)':
            pass
        elif grating == 'DV7 (600/6750)':
            pass
        elif grating == 'DV8 (831/8000)':
            pass
        elif grating == 'DV9 (1200/5000)':
            pass
        elif grating == 'DV10 (2160/5000)':
            pass
        else:
            pass

        return par



