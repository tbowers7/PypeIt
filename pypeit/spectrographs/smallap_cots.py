"""
Module for small-aperture telescopes running commercial equipment

This module is designed to encompass the configuration parameters for small-
aperture telesceopes equiped with commercial spectrographs.  The aim is to
open use of PypeIt to research institutions running this type of smaller, 
non-custom equipment and the amateur community.

While most of the major astronomical observatories supported in PypeIt have
custom spectrographs where the detector is inseparable from the instrument
and telescope, this application is quite the opposite.  Any number of observers
could be using the same, `e.g.`, Shelyak LISA spectrograph with a variety of
commercial CCD or CMOS cameras on telescopes of varying dimension and focal
ratio. Because of this, the spectrgraph class has been teased apart to include
portions for the detector, instrument, and telescope separately.

At present, the module begins with a series of user-adjusted parameters for
the three components of the spectrograph system.  It will likely be useful to
offload these parameters into a separate configuration file elsewhere on the
user's machine (possibly installed via the caching mechanism) so that the
user need not dig into the guts of the installed PypeIt package to find these
parameters for adjustment.

For now, assuming MaxIm DL as the image-capture software, but will need to
generalize as other software is supported here.

.. include:: ../include/links.rst
"""
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u
import numpy as np

from pypeit import io
from pypeit import msgs
from pypeit import data
from pypeit.core import framematch
from pypeit.core import parse
from pypeit.images import detector_container
from pypeit.par.pypeitpar import TelescopePar
from pypeit.spectrographs import spectrograph

u.add_enabled_units(u.imperial.ft)

# ======== USER TO FILL IN PARAMETERS HERE FOR THEIR OBSERVATORY ========#

# Observatory and Telescope Parameters
ADDRESS = "1400 W Mars Hill Rd.  Flagstaff, AZ 86001"  # Observatory address
ELEVATION = 1200 * u.imperial.ft  # Observatory elevation in feet
TEL_APERTURE = 0.508  # Telescope Aperture in meters
TEL_FOCLEN = 3.454  # Telesceop Focal Length in meters
TEL_CENOBSTR = 0.39  # Central Obscuration Fraction by Diameter

# Spectrograph Parameters
HORIZONTAL_SPECTRUM = True  # Spectrum is horizontal on chip
RED_TO_RIGHT = True  # Wavelength increases to the right on chip
GRATING_NAME = "Shelyak LISA 300/5000"  # Something to identify the grating
GRATING_ANGLE = 45  # Something to identify the grating tilt
SLIT_WIDTH = 23.0  # Slit width in microns
SPEC_CAMERA_DEMAG = 0.68  # The demagnification factor by the spectral camera optics
ARC_LAMPS = ["NeI", "ArI"]  # The reference arc line lists to use

# Camera Parameters
INSTRUME_KWD = "Atik Cameras"  # The INSTRUME keyword from the FITS headers
DARK_CURRENT = 0.001  # electrons per second
GAIN = 0.28  # e-/ADU (measured)
READ_NOISE = 4.0  # e-
PIXEL_SIZE = 6.45  # microns


# ======== THE REMAINDER OF THIS FILE SHOULD REMAIN UNTOUCHED ========#


# Telescope Parameter Class ====================#
class SmallApTelescopePar(TelescopePar):
    def __init__(self):
        self.loc = EarthLocation.of_address(ADDRESS)
        super(SmallApTelescopePar, self).__init__(
            longitude=self.loc.lon.to(u.si.deg).value,
            latitude=self.loc.lat.to(u.si.deg).value,
            elevation=ELEVATION.to(u.si.m).value,
            fratio=TEL_FOCLEN / TEL_APERTURE,
            diameter=TEL_APERTURE,
            eff_aperture=np.pi * TEL_APERTURE**2 / 4.0 * (1.0 - TEL_CENOBSTR**2),
        )


class COTSSpectrograph(spectrograph.Spectrograph):
    """
    Child to handle COTS spectrograph specific code
    """

    ndet = 1
    name = "smallap_cots"
    telescope = SmallApTelescopePar()
    camera = INSTRUME_KWD.replace(" ", "")
    header_name = INSTRUME_KWD
    comment = "Commercial-Off-The-Shelf Spectrographs"
    supported = False

    PLATE_SCALE = 206265.0 / TEL_FOCLEN / 1.0e3  # arcsec / mm at the focal plane
    PIXEL_SCALE = (
        PLATE_SCALE * (PIXEL_SIZE / 1.0e3) / SPEC_CAMERA_DEMAG
    )  # arcsec / pixel

    # Parameters equal to the PypeIt defaults, shown here for completeness
    # pypeline = 'MultiSlit'

    def get_detector_par(self, det, hdu=None):
        """
        Return metadata for the selected detector.

        .. warning::

            Some of the necessary detector parameters are read from the file
            header, meaning the ``hdu`` argument is effectively **required** for
            this class.  The optional use of ``hdu`` is only viable for
            automatically generated documentation.

        Args:
            det (:obj:`int`):
                1-indexed detector number.
            hdu (`astropy.io.fits.HDUList`_, optional):
                The open fits file with the raw image of interest.

        Returns:
            :class:`~pypeit.images.detector_container.DetectorContainer`:
            Object with the detector metadata.
        """
        if hdu is None:
            binning = "1,1"  # Most common use mode
        else:
            binning = self.get_meta_value(self.get_headarr(hdu), "binning")

        # Detector
        detector_dict = dict(
            binning=binning,
            det=1,
            dataext=0,
            specaxis=int(HORIZONTAL_SPECTRUM),  # Native spectrum axis (1 = x)
            specflip=not RED_TO_RIGHT,  # Flip the spectrum if blue to right
            spatflip=False,
            platescale=self.PIXEL_SCALE,  # Arcsec / pixel
            darkcurr=DARK_CURRENT * 3600,  # Electrons per hour
            saturation=65535.0,  # 16-bit ADC
            nonlinear=0.95,  # Linear to 95% of saturation
            mincounts=-1e10,
            numamplifiers=1,
            gain=np.atleast_1d(GAIN),  # e-/ADU (measured)
            ronoise=np.atleast_1d(READ_NOISE),  # e-
            datasec=np.atleast_1d("[:,:]"),  # The whole thing
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
        self.meta["ra"] = dict(card=None, compound=True)
        self.meta["dec"] = dict(card=None, compound=True)
        self.meta["target"] = dict(card=None, compound=True)
        self.meta["dispname"] = dict(card=None, compound=True)
        self.meta["decker"] = dict(card=None, compound=True)
        self.meta["binning"] = dict(card=None, compound=True)
        self.meta["mjd"] = dict(card=None, compound=True)
        self.meta["airmass"] = dict(card=None, compound=True)
        self.meta["exptime"] = dict(ext=0, card="EXPTIME")
        self.meta["instrument"] = dict(ext=0, card="INSTRUME")

        # Extras for config and frametyping
        self.meta["idname"] = dict(ext=0, card="IMAGETYP")
        self.meta["dispangle"] = dict(card=None, compound=True)
        self.meta["slitwid"] = dict(card=None, compound=True)

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
        # If RA/Dec/airmass not in the FITS header, set to zenith
        if meta_key == "ra":
            lst = Time(
                headarr[0]["DATE-OBS"], format="isot", location=self.telescope.loc
            ).sidereal_time("mean")
            return headarr[0].get("OBJCTRA", lst.degree)

        if meta_key == "dec":
            return headarr[0].get("OBJCTDEC", self.telescope["latitude"])

        if meta_key == "airmass":
            return headarr[0].get("AIRMASS", 1.0)

        if meta_key == "target":
            # Replace any spaces with underscores in the object name
            return headarr[0]["OBJECT"].replace(" ", "_")

        if meta_key == "dispname":
            return GRATING_NAME

        if meta_key == "decker":
            return "None"

        if meta_key == "binning":
            binspec = headarr[0]["XBINNING"]
            binspatial = headarr[0]["YBINNING"]
            return parse.binning2string(binspec, binspatial)

        if meta_key == "mjd":
            # Use astropy to convert 'DATE-OBS' into a mjd.
            ttime = Time(headarr[0]["DATE-OBS"], format="isot")
            return ttime.mjd

        if meta_key == "dispangle":
            return GRATING_ANGLE

        if meta_key == "slitwid":
            return np.round(self.PLATE_SCALE * (SLIT_WIDTH / 1.0e3), 2)

        msgs.error(f"Not ready for compound meta {meta_key} for COTS Spectrograph")

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
        return ["dispname", "binning"]

    def raw_header_cards(self):
        """
        Return additional raw header cards to be propagated in
        downstream output files for configuration identification.
        The list of raw data FITS keywords should be those used to populate
        the :meth:`~pypeit.spectrograph.Spectrograph.configuration_keys`
        or are used in :meth:`~pypeit.spectrograph.Spectrograph.config_specific_par`
        for a particular spectrograph, if different from the name of the
        PypeIt metadata keyword.
        This list is used by :meth:`~pypeit.spectrograph.Spectrograph.subheader_for_spec`
        to include additional FITS keywords in downstream output files.
        Returns:
            :obj:`list`: List of keywords from the raw data files that should
            be propagated in output files.
        """
        return ["XBINNING", "YBINNING"]

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
        good_exp = framematch.check_frame_exptime(fitstbl["exptime"], exprng)
        if ftype == "bias":
            return fitstbl["idname"] == "Bias Frame"
        if ftype in ["arc", "tilt"]:
            return good_exp & (["_NeAr" in target for target in fitstbl["target"]])
        if ftype in ["trace", "pixelflat"]:
            return good_exp & (fitstbl["idname"] == "Flat Field")
        if ftype == "science":
            return (
                good_exp
                & (fitstbl["idname"] == "Light Frame")
                & np.logical_not(["_NeAr" in target for target in fitstbl["target"]])
            )
        if ftype == 'dark':
            return good_exp & (fitstbl['idname'] == 'Dark Frame')
        if ftype in ["pinhole", "align", "illumflat", "sky", "lampoffflats", "standard"]:
            # Don't types pinhole or align frames
            return np.zeros(len(fitstbl), dtype=bool)
        msgs.warn(f"Cannot determine if frames are of type {ftype}")
        return np.zeros(len(fitstbl), dtype=bool)

    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of ``PypeIt`` methods.
        """
        par = super().default_pypeit_par()

        # Turn off illumflat and overscan
        set_use = dict(use_illumflat=False, use_overscan=False)
        par.reset_all_processimages_par(**set_use)

        # Make a bad pixel mask
        par["calibrations"]["bpm_usebias"] = True

        # Wavelength Calibration Parameters
        # Do not sigmaclip the arc frames for better MasterArc and better wavecalib
        par["calibrations"]["arcframe"]["process"]["clip"] = False
        # Do not sigmaclip the tilt frames
        par["calibrations"]["tiltframe"]["process"]["clip"] = False
        # Arc lamps used
        par["calibrations"]["wavelengths"]["lamps"] = ARC_LAMPS
        # Set this as default... but use `holy-grail` for DV4, DV8
        par["calibrations"]["wavelengths"]["method"] = "holy-grail"
        # The DeVeny arc line FWHM varies based on slitwidth used
        par["calibrations"]["wavelengths"]["fwhm_fromlines"] = True
        par["calibrations"]["wavelengths"]["nsnippet"] = 1  # Default: 2
        # Because of the wide wavelength range, solution more non-linear; user higher orders
        par["calibrations"]["wavelengths"]["n_first"] = 3  # Default: 2
        par["calibrations"]["wavelengths"]["n_final"] = 5  # Default: 4

        # Slit-edge settings for long-slit data (DeVeny's slit is > 90" long)
        par["calibrations"]["slitedges"]["bound_detector"] = True
        par["calibrations"]["slitedges"]["sync_predict"] = "nearest"
        par["calibrations"]["slitedges"]["minimum_slit_length"] = 90.0

        # For the tilts, our lines are not as well-behaved as others',
        #   possibly due to the Wynne type E camera.
        par["calibrations"]["tilts"]["spat_order"] = 4  # Default: 3
        par["calibrations"]["tilts"]["spec_order"] = 5  # Default: 4

        # Cosmic ray rejection parameters for science frames
        par["scienceframe"]["process"]["sigclip"] = 5.0  # Default: 4.5
        par["scienceframe"]["process"]["objlim"] = 2.0  # Default: 3.0
        par["scienceframe"]["process"]["use_darkimage"] = True  # Default: False
        par["scienceframe"]["process"]["dark_expscale"] = True  # Default: False

        # Reduction and Extraction Parameters -- Look for fainter objects
        par["reduce"]["findobj"]["snr_thresh"] = 5.0  # Default: 10.0

        # Flexure Correction Parameters
        par["flexure"]["spec_method"] = "boxcar"  # Default: 'skip'

        # Sensitivity Function Parameters
        par["sensfunc"]["polyorder"] = 7  # Default: 5

        return par

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
        return super().config_specific_par(scifile, inp_par=inp_par)

    def pypeit_file_keys(self):
        """
        Define the list of keys to be output into a standard ``PypeIt`` file.

        Returns:
            :obj:`list`: The list of keywords in the relevant
            :class:`~pypeit.metadata.PypeItMetaData` instance to print to the
            :ref:`pypeit_file`.
        """
        return super().pypeit_file_keys() + ["dispangle", "slitwid"]

    def get_rawimage(self, raw_file, det):
        """
        Read raw images and generate a few other bits and pieces
        that are key for image processing.

        For LDT/DeVeny, the LOIS control system automatically adjusts the
        DATASEC and OSCANSEC regions if the CCD is used in a binning other
        than 1x1.  The get_rawimage() method in the base class assumes these
        sections are fixed and adjusts them based on the binning -- incorrect
        for this instrument.

        This method is a stripped-down version of the base class method and
        additionally does NOT send the binning to parse.sec2slice().

        Parameters
        ----------
        raw_file : :obj:`str`
            File to read
        det : :obj:`int`
            1-indexed detector to read

        Returns
        -------
        detector_par : :class:`pypeit.images.detector_container.DetectorContainer`
            Detector metadata parameters.
        raw_img : `numpy.ndarray`_
            Raw image for this detector.
        hdu : `astropy.io.fits.HDUList`_
            Opened fits file
        exptime : :obj:`float`
            Exposure time *in seconds*.
        rawdatasec_img : `numpy.ndarray`_
            Data (Science) section of the detector as provided by setting the
            (1-indexed) number of the amplifier used to read each detector
            pixel. Pixels unassociated with any amplifier are set to 0.
        oscansec_img : `numpy.ndarray`_
            Overscan section of the detector as provided by setting the
            (1-indexed) number of the amplifier used to read each detector
            pixel. Pixels unassociated with any amplifier are set to 0.
        """
        # Open
        hdu = io.fits_open(raw_file)

        # Grab the DetectorContainer and extract the raw image
        detector = self.get_detector_par(det, hdu=hdu)
        raw_img = hdu[detector["dataext"]].data.astype(float)

        # Exposure time (used by RawImage) from the header
        headarr = self.get_headarr(hdu)
        exptime = self.get_meta_value(headarr, "exptime")

        for section in ["datasec", "oscansec"]:
            # Get the data section from Detector
            image_sections = detector[section]

            # Initialize the image (0 means no amplifier)
            pix_img = np.zeros(raw_img.shape, dtype=int)
            for i in range(detector["numamplifiers"]):
                if image_sections is not None:
                    # Convert the (FITS) data section from a string to a slice
                    # DO NOT send the binning (default: None)
                    datasec = parse.sec2slice(image_sections[i], one_indexed=True,
                                              include_end=True, require_dim=2)
                    # Assign the amplifier
                    pix_img[datasec] = i + 1

            # Finish
            if section == "datasec":
                rawdatasec_img = pix_img.copy()
            else:
                oscansec_img = pix_img.copy()

        # Return
        return detector, raw_img, hdu, exptime, rawdatasec_img, oscansec_img
