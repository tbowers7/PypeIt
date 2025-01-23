r"""
Module for LDT/NIHTS specific methods.

The Near-Infrared High-Throughput Spectrograph (NIHTS, pronounced "nights")
is a low-resolution (:math`R$\sim$200`) near-infrared prism spectrograph,
covering 0.86-2.4 microns in a single order.  NIHTS achieved first light in
November 2015.  NIHTS contains no moving parts and employs a single slit mask
with 7 different slit widths (4.03", 1.34", 0.81", 0.27", 0.54", 1.07" and
1.61"), each approximately 12" in length.  During early commissioning NIHTS
was fed by a fold mirror in the instrument cube, which was replaced by a
dichroic on 13 December 2017 to allow for simultaneous LMI (optical) imaging
and NIHTS spectroscopy.  Current instrumentation plans have NIHTS available on
the telescope every night.

NIHTS commissioning continued through the 2018A semester with the instrument
operating in shared risk mode.  The instrument has been available for normal
science observing since 01 July 2018.  The NIHTS Commissioning work has been
published in `Gustafsson et al.\ (2021) in PASP
<https://ui.adsabs.harvard.edu/abs/2021PASP..133c5001G/abstract>`_.  

.. include:: ../include/links.rst
"""

import astropy.coordinates
import astropy.io.fits
import astropy.table
import astropy.time
import numpy as np

from pypeit import msgs
from pypeit import telescopes
from pypeit.core import framematch
from pypeit.core import parse
from pypeit.images import detector_container
from pypeit.par import pypeitpar
from pypeit.spectrographs import spectrograph


class LDTNIHTSSpectrograph(spectrograph.Spectrograph):
    """
    Child to handle LDT/NIHTS specific code
    """

    ndet = 1
    name = "ldt_nihts"
    telescope = telescopes.LDTTelescopePar()
    camera = "nihts"
    url = "https://lowell.edu/research/telescopes-and-facilities/ldt/nihts/"
    header_name = "NIHTS"
    comment = "LDT NIHTS IR Spectrograph, 2015 - present"
    supported = True

    # Parameters equal to the PypeIt defaults, shown here for completeness
    # pypeline = 'MultiSlit'

    def get_detector_par(
        self, det: int, hdu: astropy.io.fits.HDUList = None
    ) -> detector_container.DetectorContainer:
        """
        Return metadata for the selected detector.

        .. warning::

            Many of the necessary detector parameters are read from the file
            header, meaning the ``hdu`` argument is effectively **required** for
            LTD/NIHTS.  The optional use of ``hdu`` is only viable for
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
            dataext = 0  # Raw data
            numamps = 4  # Most common use mode
            binning = "1,1"  # Most common use mode
            gainarr = np.full(numamps, 24.0)  # Hardcoded in the header
            ronarr = np.full(numamps, 100.0)  # Hardcoded in the header

        else:
            # If file is post-processed, data extension is specified.  Raw is 0.
            dataext = hdu[0].header.get("POST_EXT", 0)
            numamps = hdu[0].header.get("NUMAMP", 4)
            binning = self.get_meta_value(self.get_headarr(hdu), "binning")
            gainarr = np.zeros(numamps, dtype=float)
            ronarr = np.zeros(numamps, dtype=float)

            for ii in range(numamps):
                # Assign the gain for this amplifier
                gainarr[ii] = hdu[0].header[f"GAIN_{ii+1:02d}"]
                # Assign the readout noise for this amplifier
                ronarr[ii] = hdu[0].header[f"RN_{ii+1:02d}"]

        # Detector
        detector_dict = dict(
            binning=binning,
            det=1,  # NIHTS has but one detector
            dataext=dataext,
            specaxis=1,  # Native spectrum is along the x-axis
            specflip=False,  # NIHTS H1 chip has blue at the left
            spatflip=False,
            platescale=0.13,  # Arcsec / pixel
            darkcurr=0.0,  # e-/pixel/hour
            saturation=16000.0,  # From the NIHTS manual
            nonlinear=0.97,  # Linear to ~97% of saturation
            mincounts=-1e10,
            numamplifiers=numamps,
            gain=gainarr,  # See above
            ronoise=ronarr,  # See above
            # Manually set the data and overscan sections because the values
            #   in the header are incorrect
            datasec=np.atleast_1d(
                [
                    "[:512,:512]",
                    "[513:,:512]",
                    "[:512,513:]",
                    "[513:,513:]",
                ]
            ),
            oscansec=np.atleast_1d(
                [
                    "[:4,:4]",
                    "[1022:,:4]",
                    "[:4,1022:]",
                    "[1022:,1022:]",
                ]
            ),
        )
        return detector_container.DetectorContainer(**detector_dict)

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the PypeIt-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}

        # Required (core)
        self.meta["ra"] = dict(ext=0, card="RA")
        self.meta["dec"] = dict(ext=0, card="DEC")
        self.meta["target"] = dict(card=None, compound=True)
        self.meta["dispname"] = dict(ext=0, card="INSTRUME")
        self.meta["decker"] = dict(card=None, compound=True)
        self.meta["binning"] = dict(card=None, compound=True)
        self.meta["mjd"] = dict(card=None, compound=True)
        self.meta["airmass"] = dict(ext=0, card="AIRMASS")
        self.meta["exptime"] = dict(ext=0, card="EXPTIME")
        self.meta["instrument"] = dict(ext=0, card="INSTRUME")

        # Extras for config, frametyping, and nodding
        # NOTE: `rtol` is _relative_ tolerance (e.g. 1 part in 1,000)
        self.meta["idname"] = dict(ext=0, card="IMAGETYP")
        self.meta["slitwid"] = dict(card=None, compound=True)
        self.meta["lampstat01"] = dict(card=None, compound=True)
        self.meta["frameno"] = dict(ext=0, card="OBSERNO")
        self.meta["dithpos"] = dict(card=None, compound=True)
        self.meta["utc"] = dict(ext=0, card="UTCSTART")

    def compound_meta(self, headarr: list, meta_key: str) -> object:
        """
        Methods to generate metadata requiring interpretation of the header
        data, instead of simply reading the value of a header card.

        Args:
            headarr (:obj:`list`):
                List of `astropy.io.fits.Header`_ objects.
            meta_key (:obj:`str`):
                Metadata keyword to construct.

        Returns:
            :obj:`object`: Metadata value read from the header(s).
        """
        if meta_key == "binning":
            # Binning in lois headers is space-separated, spec x spat
            binspec, binspatial = parse.parse_binning(headarr[0]["CCDSUM"])
            return parse.binning2string(binspec, binspatial)

        if meta_key == "mjd":
            # Use custom scrubber + AstroPy to convert 'DATE-OBS' into a mjd.
            ttime = self.scrub_isot_dateobs(headarr[0]["DATE-OBS"])
            return ttime.mjd

        if meta_key == "lampstat01":
            # NIHTS uses only the Xe lamp attached to the bottom of the
            #  instrument cube.  The way the scripts are written, the Xe lamp
            #  is only turned on when the target name "Comparison" with frame
            #  type "COMPARISON" is used.  There are lamp-off subtraction
            #  frames taken with target name "Comparison - No Lamps" and frame
            #  type "COMPARISON".
            return (
                "Xe"
                if (
                    headarr[0]["OBSTYPE"] == "COMPARISON"
                    and headarr[0]["OBJNAME"] == "Comparison"
                )
                else "off"
            )

        if meta_key == "decker":
            # NIHTS has no decker
            return "None"

        if meta_key == "slitwid":
            # The width of the slitlet selected in the XCAM GUI for the target
            #   Stored in the first comment card (Slit:n.nn, Position:N)
            try:
                return float(headarr[0]["COMMENT"][0].split(",")[0].split(":")[1])
            except ValueError:
                # For calibration frames or SED1/2, use the width of the end slitlets
                return 4.03

        if meta_key == "dithpos":
            # The nodding position of the target along the slit
            #   Stored in the first comment card (Slit:n.nn, Position:N)
            return headarr[0]["COMMENT"][0].split(",")[1].split(":")[1]

        if meta_key == "target":
            # Revert to TCS's SCITARG if target not set in LOUI for OBJECT frames
            return (
                headarr[0]["SCITARG"].strip()
                if (
                    headarr[0]["IMAGETYP"].strip() == "OBJECT"
                    and headarr[0]["OBJNAME"].strip() in ["UNKNOWN", ""]
                )
                else headarr[0]["OBJNAME"].strip()
            )

        msgs.error(f'Not ready for compound meta "{meta_key}" for LDT/NIHTS')

    def configuration_keys(self) -> list[str]:
        """
        Return the metadata keys that define a unique instrument
        configuration.

        This list is used by :class:`~pypeit.metadata.PypeItMetaData` to
        identify the unique configurations among the list of frames read
        for a given reduction.

        For NIHTS, there is only one possible configuration (no moving parts),
        so this method returns an empty list.

        Returns:
            :obj:`list`: List of keywords of data pulled from file headers
            and used to constuct the :class:`~pypeit.metadata.PypeItMetaData`
            object.
        """
        return []

    def pypeit_file_keys(self) -> list[str]:
        """
        Define the list of keys to be output into a standard PypeIt file.

        Returns:
            :obj:`list` : The list of keywords in the relevant
            :class:`~pypeit.metadata.PypeItMetaData` instance to print to the
            :ref:`pypeit_file`.
        """
        return super().pypeit_file_keys() + ["utc", "slitwid", "lampstat01", "dithpos"]

    @classmethod
    def default_pypeit_par(cls) -> pypeitpar.PypeItPar:
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of PypeIt methods.
        """
        par = super().default_pypeit_par()

        # No bias or overscan for IRFPAs
        par.reset_all_processimages_par(
            use_biasimage=False, use_illumflat=False, use_overscan=False
        )

        # Slit-edge settings for NIHTS' slitlets
        par["calibrations"]["slitedges"]["edge_thresh"] = 15.0  # Default: 20.0
        par["calibrations"]["slitedges"]["fit_order"] = 2  # Default: 5
        par["calibrations"]["slitedges"]["max_nudge"] = 5  # Default: None
        par["calibrations"]["slitedges"]["minimum_slit_length"] = 6.0  # Default: None
        par["calibrations"]["slitedges"]["smash_range"] = [0.2, 0.5]  # Default: None
        par["calibrations"]["slitedges"]["sync_predict"] = "nearest"  # Default: 'pca'
        par["calibrations"]["slitedges"]["trace_thresh"] = 50  # Default: None
        par["calibrations"]["slitedges"]["trim_spec"] = [0, 50]  # Default: None

        # Only use LONG arc frames
        par["calibrations"]["arcframe"]["exprng"] = [30, None]
        # For processing the arc frame, these settings allow for the combination of
        #   of frames from different lamps into a comprehensible Master
        par["calibrations"]["arcframe"]["process"]["clip"] = False
        par["calibrations"]["arcframe"]["process"]["combine"] = "mean"
        # par['calibrations']['arcframe']['process']['subtract_continuum'] = True
        par["calibrations"]["tiltframe"]["process"]["clip"] = False
        par["calibrations"]["tiltframe"]["process"]["combine"] = "mean"
        # par['calibrations']['tiltframe']['process']['subtract_continuum'] = True

        # Wavelength Calibration Parameters
        # Arc lamps list from header -- instead of defining the full list here
        par["calibrations"]["wavelengths"]["lamps"] = ["XeI"]
        # Set this as default... but use `holy-grail` for DV4, DV8
        par["calibrations"]["wavelengths"][
            "method"
        ] = "holy-grail"  #'full_template'  # Default: 'holy-grail'
        # Reidentification parameters
        par["calibrations"]["wavelengths"]["reid_arxiv"] = "ldt_nihts.fits"
        # The DeVeny arc line FWHM varies based on slitwidth used
        par["calibrations"]["wavelengths"]["fwhm_fromlines"] = True  # Default: True
        par["calibrations"]["wavelengths"]["nsnippet"] = 1  # Default: 2

        # # For the tilts, our lines are not as well-behaved as others',
        # #   possibly due to the Wynne version E camera.
        # par["calibrations"]["tilts"]["spat_order"] = 4  # Default: 3
        # par["calibrations"]["tilts"]["spec_order"] = 5  # Default: 4

        # Flat-field parameter modification
        par["calibrations"]["flatfield"]["pixelflat_min_wave"] = 3000.0  # Default: None
        par["calibrations"]["flatfield"]["slit_illum_finecorr"] = False  # Default: True
        par["calibrations"]["flatfield"]["spec_samp_fine"] = 30  # Default: 1.2
        par["calibrations"]["flatfield"]["tweak_slits"] = False  # Default: True

        # Cosmic ray rejection parameters for science frames
        par["scienceframe"]["process"]["sigclip"] = 5.0  # Default: 4.5
        par["scienceframe"]["process"]["objlim"] = 2.0  # Default: 3.0

        # Object Finding, Extraction, and Sky Subtraction Parameters
        assumed_seeing = 1.5  # arcsec
        par["reduce"]["findobj"]["trace_npoly"] = 3  # Default: 5
        par["reduce"]["findobj"]["snr_thresh"] = 50.0  # Default: 10.0
        par["reduce"]["findobj"]["maxnumber_std"] = 1  # Default: 5
        par["reduce"]["findobj"]["maxnumber_sci"] = 5  # Default: 10
        par["reduce"]["findobj"]["find_fwhm"] = np.round(
            assumed_seeing / 0.34, 1
        )  # Default: 5.0 pix
        par["reduce"]["findobj"]["find_trim_edge"] = [0, 0]  # Default: [5, 5]
        # Boxcar width = ±3σ of Gaussian profile = >99% enclosed flux; radius = 1.28 * seeing
        par["reduce"]["extraction"]["boxcar_radius"] = np.round(
            assumed_seeing * 1.28, 1
        )  # Default: 1.5"
        par["reduce"]["extraction"]["use_2dmodel_mask"] = False  # Default: True
        par["reduce"]["skysub"]["sky_sigrej"] = 4.0  # Default: 3.0

        # Flexure Correction Parameters
        par["flexure"]["spec_method"] = "boxcar"  # Default: 'skip'
        par["flexure"]["spec_maxshift"] = 30  # Default: 20

        # Sensitivity Function Parameters
        par["sensfunc"]["UVIS"]["nresln"] = 15  # Default: 20
        par["sensfunc"]["UVIS"]["polycorrect"] = False  # Default: True

        return par

    def check_frame_type(
        self, ftype: str, fitstbl: astropy.table.Table, exprng: list = None
    ) -> np.ndarray:
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
                :func:`~pypeit.core.framematch.check_frame_exptime`.

        Returns:
            `numpy.ndarray`_: Boolean array with the flags selecting the
            exposures in ``fitstbl`` that are ``ftype`` type frames.
        """
        good_exp = framematch.check_frame_exptime(fitstbl["exptime"], exprng)
        if ftype in ["arc", "tilt"]:
            # FOCUS frames should have frametype None
            return (
                good_exp
                & (fitstbl["lampstat01"] != "off")
                & (fitstbl["idname"] != "FOCUS")
            )
        if ftype in ["trace", "pixelflat"]:
            return (
                good_exp
                & (fitstbl["idname"] == "DOME FLAT")
                & (fitstbl["lampstat01"] == "off")
                & (["- No Lamp" not in objname for objname in fitstbl["target"]])
            )
        if ftype == "dark":
            return (
                good_exp
                & (fitstbl["lampstat01"] == "off")
                & (["- No Lamp" in objname for objname in fitstbl["target"]])
            )
        if ftype == "illumflat":
            return (
                good_exp
                & (fitstbl["idname"] == "SKY FLAT")
                & (fitstbl["lampstat01"] == "off")
            )
        if ftype == "science":
            return (
                good_exp
                & (fitstbl["idname"] == "OBJECT")
                & (fitstbl["lampstat01"] == "off")
            )
        if ftype == "standard":
            return (
                good_exp
                & (fitstbl["idname"] == "STANDARD")
                & (fitstbl["lampstat01"] == "off")
            )
        if ftype in [
            "bias",
            "lampoffflats",
            "pinhole",
            "align",
            "sky",
            "scattlight",
            "slitless_pixflat",
        ]:
            # NIHTS doesn't have any of these types of frames
            return np.zeros(len(fitstbl), dtype=bool)
        msgs.warn(f"Cannot determine if frames are of type {ftype}")
        return np.zeros(len(fitstbl), dtype=bool)

    def tweak_standard(
        self,
        wave_in,
        counts_in,
        counts_ivar_in,
        gpm_in,
        meta_table,
        log10_blaze_function=None,
    ):
        """
        This routine is for performing instrument- and/or disperser-specific
        tweaks to standard stars so that sensitivity function fits will be
        well behaved.

        These are tweaks needed by LDT/DeVeny for smooth sensfunc sailing.

        Parameters
        ----------
        wave_in: `numpy.ndarray`_
            Input standard star wavelengths (:obj:`float`, ``shape = (nspec,)``)
        counts_in: `numpy.ndarray`_
            Input standard star counts (:obj:`float`, ``shape = (nspec,)``)
        counts_ivar_in: `numpy.ndarray`_
            Input inverse variance of standard star counts (:obj:`float`, ``shape = (nspec,)``)
        gpm_in: `numpy.ndarray`_
            Input good pixel mask for standard (:obj:`bool`, ``shape = (nspec,)``)
        meta_table: :obj:`dict`
            Table containing meta data that is slupred from the :class:`~pypeit.specobjs.SpecObjs`
            object.  See :meth:`~pypeit.specobjs.SpecObjs.unpack_object` for the
            contents of this table.
        log10_blaze_function: `numpy.ndarray`_ or None
            Input blaze function to be tweaked, optional. Default=None.

        Returns
        -------
        wave_out: `numpy.ndarray`_
            Output standard star wavelengths (:obj:`float`, ``shape = (nspec,)``)
        counts_out: `numpy.ndarray`_
            Output standard star counts (:obj:`float`, ``shape = (nspec,)``)
        counts_ivar_out: `numpy.ndarray`_
            Output inverse variance of standard star counts (:obj:`float`, ``shape = (nspec,)``)
        gpm_out: `numpy.ndarray`_
            Output good pixel mask for standard (:obj:`bool`, ``shape = (nspec,)``)
        log10_blaze_function_out: `numpy.ndarray`_ or None
            Output blaze function after being tweaked.
        """
        # First, simply chop off the wavelengths outside physical limits:
        valid_wave = (wave_in >= 2900.0) & (wave_in <= 11000.0)
        wave_out = wave_in[valid_wave]
        counts_out = counts_in[valid_wave]
        counts_ivar_out = counts_ivar_in[valid_wave]
        gpm_out = gpm_in[valid_wave]

        if log10_blaze_function is not None:
            log10_blaze_function_out = log10_blaze_function[valid_wave]
        else:
            log10_blaze_function_out = None

        # Next, build a gpm based on other reasonable wavelengths and filters
        edge_region = (wave_out < 3000.0) | (wave_out > 10200.0)
        neg_counts = counts_out <= 0

        # If an order-blocking filter was in use, mask blocked region
        #  at "nominal" cutoff value
        if "FILTER1" in meta_table.keys():
            rearfilt = meta_table["FILTER1"].strip()
            if rearfilt == "OG570":
                block_region = wave_out < 5700.0
            elif rearfilt == "GG495":
                block_region = wave_out < 4950.0
            elif rearfilt == "GG420":
                block_region = wave_out < 4200.0
            elif rearfilt == "WG360":
                block_region = wave_out < 3600.0
            else:
                block_region = wave_out < 0
        # In case the filter didn't make it into the header
        else:
            block_region = wave_out < 0

        # Build up the OUTPUT GOOD PIXEL MASK
        gpm_out = (
            gpm_out
            & np.logical_not(edge_region)
            & np.logical_not(neg_counts)
            & np.logical_not(block_region)
        )

        return wave_out, counts_out, counts_ivar_out, gpm_out, log10_blaze_function_out

    @staticmethod
    def rotate_trimsections(
        section_string: str, str_only: bool = False
    ) -> np.ndarray | str:
        """
        In order to orient LDT/NIHTS images into the PypeIt-standard
        configuration, frames are flipped over the x=y line.  As such,
        :math:`x' = y` and :math:`y' = x`.

        The ``TRIMSEC`` / ``BIASSEC`` FITS keywords in LDT/NIHTS data specify
        the proper regions to be trimmed for the data and overscan arrays,
        respectively, in the native orientation.  This method performs the
        rotation and returns the slices for the Numpy image section required
        by the PypeIt processing routines.

        The LDT/NIHTS FITS header lists the sections as ``'[SPEC_SEC,SPAT_SEC]'``.

        Args:
            section_string (:obj:`str`):
                The FITS keyword string to be parsed / translated
            str_only (:obj:`bool`, optional):
                Return the string only without formatting it for PypeIt's
                needed numpy image section
        Returns:
            section (`numpy.ndarray`_ or :obj:`str`):
                Numpy image section needed by PypeIt
        """
        # Split out the input section into spectral and spatial pieces
        spec_sec, spat_sec = section_string.strip("[]").split(",")

        # Both the spatial and spectral sections are unchanged.
        str_section = f"[{spat_sec},{spec_sec}]"
        # Return the PypeIt-standard Numpy array if not just the string
        return str_section if str_only else np.atleast_1d(str_section)

    @staticmethod
    def scrub_isot_dateobs(dt_str: str) -> astropy.time.Time:
        """Scrub the input ``DATE-OBS`` for ingestion by AstroPy Time

        The main issue this method addresses is that sometimes the LOIS
        software at LDT has roundoff abnormalities in the time string written
        to the ``DATE-OBS`` header keyword.  For example, in one header
        ``2020-01-30T13:17:010.0`` was written, where the seconds has 3 digits
        -- presumably the seconds field was constructed with a leading zero
        because ``sec`` was < 10, but when rounded for printing
        yielded "10.00", producing a complete seconds field of ``010.00``.

        This abnormality, along with a seconds field equaling ``60.00``, causes
        AstroPy's Time parser to freak out with a ``ValueError``.  This
        method attempts to return the `astropy.time.Time`_ object directly, but
        then scrubs any values that cause a ``ValueError``.

        The scrubbing consists of deconstructing the string into its components,
        then carefully reconstructing it into proper ISO 8601 format.  Also,
        some recursive edge-case catching is done, but at some point you just
        have to give up and go buy a lottery ticket.

        If you have a truly bizarre ``DATE-OBS`` string, simply edit that keyword
        in the FITS header and then re-run PypeIt.

        Parameters
        ----------
        dt_str : :obj:`str`
            Input datetime string from the ``DATE-OBS`` header keyword

        Returns
        -------
        `astropy.time.Time`_
            The AstroPy Time object corresponding to the ``DATE-OBS`` input string
        """
        # Clean all leading / trailing whitespace
        dt_str = dt_str.strip()

        # Attempt to directly return the AstroPy Time object
        try:
            return astropy.time.Time(dt_str, format="isot")
        except ValueError:
            # Split out all pieces of the datetime, and recompile
            date, time = dt_str.split("T")
            yea, mon, day = date.split("-")
            hou, mnt, sec = time.split(":")
            # Check if the seconds is exactly equal to 60... increment minute
            if sec == "60.00":
                sec = "00.00"
                if mnt != "59":
                    mnt = int(mnt) + 1
                else:
                    mnt = "00"
                    if hou != "23":
                        hou = int(hou) + 1
                    else:
                        hou = "00"
                        # If the edge cases go past here, go buy a lottery ticket!
                        day = int(day) + 1
            # Reconstitute the DATE-OBS string, and return the Time() object
            date = f"{int(yea):04d}-{int(mon):02d}-{int(day):02d}"
            time = f"{int(hou):02d}:{int(mnt):02d}:{float(sec):09.6f}"
            return astropy.time.Time(f"{date}T{time}", format="isot")
