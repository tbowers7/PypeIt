"""
Implements the objects used to hold slit edge data.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst

"""
import inspect

from IPython import embed

import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from astropy import units
from astropy.stats import sigma_clipped_stats

from pypeit import msgs
from pypeit import datamodel
from pypeit import specobj
from pypeit.bitmask import BitMask
from pypeit.spectrographs import slitmask


class SlitTraceBitMask(BitMask):
    """
    Mask bits used during slit tracing.
    """
    version = '1.0.0'

    def __init__(self):
        mask = dict([
            ('SHORTSLIT', 'Slit formed by left and right edge is too short. Not ignored for flexure'),
            ('BOXSLIT', 'Slit formed by left and right edge is valid (large enough to be a valid '
                        'slit), but too short to be a science slit'),
            ('USERIGNORE', 'User has specified to ignore this slit. Not ignored for flexure.'),
            ('BADWVCALIB', 'Wavelength calibration failed for this slit'),
            ('BADTILTCALIB', 'Tilts analysis failed for this slit'),
            ('SKIPFLATCALIB', 'Flat field generation failed for this slit. Skip flat fielding'),
            ('BADFLATCALIB', 'Flat field generation failed for this slit. Ignore it fully.'),
            ('BADREDUCE', 'Skysub/extraction failed for this slit'),
        ])
        super(SlitTraceBitMask, self).__init__(list(mask.keys()), descr=list(mask.values()))

    @property
    def exclude_for_reducing(self):
        # Ignore these flags when reducing or considering reduced slits
        return ['SKIPFLATCALIB']

    @property
    def exclude_for_flexure(self):
        # Ignore these flags when performing a flexure calculation
        #  Currently they are *all* of the flags..
        return ['SHORTSLIT', 'USERIGNORE', 'BADWVCALIB', 'BADTILTCALIB',
                'SKIPFLATCALIB', 'BADFLATCALIB', 'BADREDUCE']



class SlitTraceSet(datamodel.DataContainer):
    """
    Defines a generic class for holding and manipulating image traces
    organized into left-right slit pairs.

    Instantiation arguments map directly to the object
    :attr:`datamodel`. The only additional argument is ``load``,
    described below.

    :class:`SlitTraceSet` objects can be instantiated with only the
    master-frame arguments. If this is done, it's expected that
    you'll attempt to load an existing master frame, either using
    ``load=True`` on instantiation or by a call to :func:`load`.
    Otherwise, all the elements of the data model will be empty.

    Args:
        load (:obj:`bool`, optional):
            Attempt to load an existing master frame with the slit
            trace data. WARNING: If ``load`` is True, all of the slit
            information provided to the initialization is ignored!
            Only the arguments relevant to the
            :class:`pypeit.masterframe.MasterFrame` components of
            :class:`SlitTraceSet` are used.

    Attributes:
        left_flexure (`numpy.ndarray`_):  Convenient spot to hold flexure corrected left
        right_flexure (`numpy.ndarray`_):  Convenient spot to hold flexure corrected right
        master_key (:obj:`str`):
        master_dir (:obj:`str`):

    """
    frametype = 'slits'
    master_type = 'Slits'
    """Name for type of master frame."""
    master_file_format = 'fits.gz'
    """File format for the master frame file."""
    minimum_version = '1.1.0'
    version = '1.1.3'
    """SlitTraceSet data model version."""

    hdu_prefix = None
    output_to_disk = None

    bitmask = SlitTraceBitMask()

    # Define the data model
    datamodel = {'PYP_SPEC': dict(otype=str, descr='PypeIt spectrograph name'),
                 'pypeline': dict(otype=str, descr='PypeIt pypeline name'),
                 'det': dict(otype=int, descr='Detector'),
                 'nspec': dict(otype=int,
                               descr='Number of pixels in the image spectral direction.'),
                 'nspat': dict(otype=int,
                               descr='Number of pixels in the image spatial direction.'),
                 'binspec': dict(otype=int,
                                 descr='Number of pixels binned in the spectral direction.'),
                 'binspat': dict(otype=int,
                                 descr='Number of pixels binned in the spatial direction.'),
                 'pad': dict(otype=int,
                             descr='Integer number of pixels to consider beyond the slit edges.'),
                 'spat_id': dict(otype=np.ndarray, atype=(int,np.integer),
                                 descr='Slit ID number from SPAT measured at half way point.'),
                 'maskdef_id': dict(otype=np.ndarray, atype=(int,np.integer),
                                    descr='Slit ID number slitmask'),
                 'maskdef_designtab': dict(otype=Table, descr='Table with slitmask design and object info'),
                 'maskfile': dict(otype=str, descr='Data file that yielded the slitmask info'),
                 'maskdef_posx_pa': dict(otype=float, descr='PA that aligns with spatial dimension of the detector'),
                 'mask_median_off': dict(otype=float, descr='Median offset (pixels) of slitmask from expected position'),
                 'ech_order': dict(otype=np.ndarray, atype=(int,np.integer),
                                   descr='Slit ID number echelle order'),
                 'nslits': dict(otype=int,
                                descr='Total number of slits, derived from shape of left_init.'),
                 'left_init': dict(otype=np.ndarray, atype=np.floating,
                              descr='Spatial coordinates (pixel indices) of all left edges, one '
                                    'per slit.  Derived from the TraceImage. Shape is Nspec by '
                                    'Nslits.'),
                 'right_init': dict(otype=np.ndarray, atype=np.floating,
                              descr='Spatial coordinates (pixel indices) of all right edges, one '
                                    'per slit.  Derived from the TraceImage. Shape is Nspec by '
                                    'Nslits.'),
                 'left_tweak': dict(otype=np.ndarray, atype=np.floating,
                                    descr='Spatial coordinates (pixel indices) of all left '
                                          'edges, one per slit.  These traces have been adjusted '
                                          'by the flat-field.  Shape is Nspec by Nslits.'),
                 'right_tweak': dict(otype=np.ndarray, atype=np.floating,
                                     descr='Spatial coordinates (pixel indices) of all right '
                                           'edges, one per slit.  These traces have been adjusted '
                                           'by the flat-field.  Shape is Nspec by Nslits.'),
                 'center': dict(otype=np.ndarray, atype=np.floating,
                               descr='Spatial coordinates of the slit centers from left_init, '
                                     'right_init.  Shape is Nspec by Nslits.'),
                 'mask_init': dict(otype=np.ndarray, atype=np.integer,
                                   descr='Bit mask for slits at instantiation.  Used to reset'),
                 'mask': dict(otype=np.ndarray, atype=np.integer,
                              descr='Bit mask for slits (fully good slits have 0 value).  Shape '
                                    'is Nslits.'),
                'slitbitm': dict(otype=str, desc='List of BITMASK keys from SlitTraceBitMask'),
                'specmin': dict(otype=np.ndarray, atype=np.floating,
                                descr='Minimum spectral position allowed for each slit/order.  '
                                      'Shape is Nslits.'),
                'specmax': dict(otype=np.ndarray, atype=np.floating,
                                descr='Maximum spectral position allowed for each slit/order.  '
                                      'Shape is Nslits.')}
    """Provides the class data model."""

    # TODO: Allow tweaked edges to be arguments?
    # TODO: May want nspat to be a required argument.
    # The INIT must contain every datamodel item or risk fail on I/O when it is a nested container
    def __init__(self, left_init, right_init, pypeline, det=None, nspec=None, nspat=None, PYP_SPEC=None,
                 mask_init=None, specmin=None, specmax=None, binspec=1, binspat=1, pad=0,
                 spat_id=None, maskdef_id=None, maskdef_designtab=None, maskfile=None,
                 maskdef_posx_pa=None, mask_median_off=None,
                 ech_order=None, nslits=None, left_tweak=None,
                 right_tweak=None, center=None, mask=None, slitbitm=None):

        # Instantiate the DataContainer
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        _d = dict([(k,values[k]) for k in args[1:]])
        # The dictionary passed to DataContainer.__init__ does not
        # contain self or the MasterFrame arguments.
        # TODO: Does it matter if the calling function passes the
        # keyword arguments in a different order?
        datamodel.DataContainer.__init__(self, d=_d)

    def _validate(self):
        """
        Validate the slit traces.
        """
        # Allow the object to be empty
        if self.left_init is None or self.right_init is None:
            return
        if self.left_init.shape != self.right_init.shape:
            raise ValueError('Input left_init and right_init traces should have the same shape.')
        if self.left_init.ndim == 1:
            # Object contains a single slit.  Expand the dimensions.
            self.left_init = np.expand_dims(self.left_init, 1)
            self.right_init = np.expand_dims(self.right_init, 1)

        # Do the same for the tweaked traces. NOTE: Tweaked traces will
        # only exist if they were read from an output file; i.e., the
        # current init does not allow for the tweaked traces to be
        # defined directly.
        if self.left_tweak is not None:
            if self.left_tweak.shape != self.right_tweak.shape:
                # TODO: Shouldn't get here, but just in case the init changes...
                raise ValueError('Tweaked left and right traces should have the same shape.')
            if self.left_tweak.ndim == 1:
                # Object contains a single slit.  Expand the dimensions.
                self.left_tweak = np.expand_dims(self.left_tweak, 1)
                self.right_tweak = np.expand_dims(self.right_tweak, 1)

        self.nspec, self.nslits = self.left_init.shape

        # Center is always defined by the original traces, not the
        # tweaked ones. This means that the slit IDs are always tied
        # to the original traces, not the tweaked ones.
        self.center = (self.left_init+self.right_init)/2

        if self.nspat is None:
            # TODO: May want nspat to be a required argument given the
            # only other option is this kludge, which should basically
            # never be useful.
            self.nspat = int(np.amax(np.append(self.left_init, self.right_init)))
        if self.spat_id is None:
            self.spat_id = np.round(self.center[int(np.round(0.5 * self.nspec)), :]).astype(int)
        if self.PYP_SPEC is None:
            self.PYP_SPEC = 'unknown'
        if self.mask_init is None:
            self.mask_init = np.zeros(self.nslits, dtype=self.bitmask.minimum_dtype())
        if self.specmin is None:
            self.specmin = np.full(self.nslits, -1, dtype=float)
        if self.specmax is None:
            self.specmax = np.full(self.nslits, self.nspec, dtype=float)

        # If the echelle order is provided, check that the number of
        # orders matches the number of provided "slits"
        if self.ech_order is not None and len(self.ech_order) != self.nslits:
            msgs.error('Number of provided echelle orders does not match the number of '
                       'order traces.')

        # Make sure mask, specmin, and specmax are at least 1D arrays.
        # TODO: Is there a way around this?
        self.mask_init = np.atleast_1d(self.mask_init)
        self.specmin = np.atleast_1d(self.specmin)
        self.specmax = np.atleast_1d(self.specmax)
        if self.slitbitm is None:
            self.slitbitm = ','.join(list(self.bitmask.keys()))
        else:
            # Validate
            if self.slitbitm != ','.join(list(self.bitmask.keys())):
                msgs.error("Input BITMASK keys differ from current data model!")
        # Mask
        if self.mask is None:
            self.mask = self.mask_init.copy()

    def _init_internals(self):
        self.left_flexure = None
        self.right_flexure = None
        # Master stuff
        self.master_key = None
        self.master_dir = None

    def _bundle(self):
        """
        Bundle the data in preparation for writing to a fits file.

        See :func:`pypeit.datamodel.DataContainer._bundle`. Data is
        always written to a 'SLITS' extension.
        """
        bndl = super(SlitTraceSet, self)._bundle(ext='SLITS', transpose_arrays=True)
        if self.maskdef_designtab is not None:
            # save the table
            tab_detached = bndl[0]['SLITS']['maskdef_designtab']
            # remove `tab_detached` from the dict
            bndl[0]['SLITS'].pop('maskdef_designtab')
            # create a dict for the `tab_detached`
            tab_dict = {'maskdef_designtab': tab_detached}
            return [bndl[0], tab_dict]
        else:
            return bndl

    @classmethod
    def _parse(cls, hdu, hdu_prefix=None):
        """
        Parse the data that was previously written to a fits file.

        See :func:`pypeit.datamodel.DataContainer._parse`. Data is
        always read from the 'SLITS' extension.
        """
        try:
            return super(SlitTraceSet, cls)._parse(hdu, ext=['SLITS', 'MASKDEF_DESIGNTAB'],
                                                   transpose_table_arrays=True)
        except KeyError:
            return super(SlitTraceSet, cls)._parse(hdu, ext='SLITS',
                                                   transpose_table_arrays=True)

    def init_tweaked(self):
        """
        Initialize the tweaked slits.
        """
        self.left_tweak = self.left_init.copy()
        self.right_tweak = self.right_init.copy()

    def rm_tweaked(self):
        """
        Delete (set to None) the tweaked traces.
        """
        self.left_tweak = None
        self.right_tweak = None

    @property
    def slit_info(self):
        """

        Returns:
            `numpy.ndarray`_:

        """
        #
        info = np.vstack([self.spat_id, self.mask])
        info = np.vstack([info, np.zeros_like(self.spat_id)]) \
                    if self.maskdef_id is None else np.vstack([info, self.maskdef_id])
        return info.astype(int).T

    # TODO: Do we need both of these? I.e., can the 'spat_id' for
    # echelle spectrographs just be the echelle order?
    @property
    def slitord_id(self):
        """
        Return array of slit_spatId (MultiSlit, IFU) or ech_order (Echelle) values

        Returns:
            `numpy.ndarray`_:

        """
        if self.pypeline in ['MultiSlit', 'IFU']:
            return self.spat_id
        elif self.pypeline == 'Echelle':
            return self.ech_order
        else:
            msgs.error('Unrecognized Pypeline {:}'.format(self.pypeline))

    def spatid_to_zero(self, spat_id):
        """
        Convert input spat_id into a zero-based index

        Args:
            spat_id (int):

        Returns:
            int: zero-based index of the input spat_id

        """
        return np.where(self.spat_id == spat_id)[0][0]

    def slitord_to_zero(self, slitord):
        """
        Convert input slitord into a zero-based index

        Args:
            slitord (int):

        Returns:
            int: zero-based index of the input spat_id

        """
        if self.pypeline in ['MultiSlit', 'IFU']:
            return np.where(self.spat_id == slitord)[0][0]
        elif self.pypeline in ['Echelle']:
            return np.where(self.ech_order == slitord)[0][0]
        else:
            msgs.error('Unrecognized Pypeline {:}'.format(self.pypeline))

    def get_slitlengths(self, initial=False, median=False):
        """
        Get the length of each slit in pixels.

        By default, the method will return the tweaked slit lengths
        if they have been defined. If they haven't been defined the
        nominal edges (:attr:`left` and :attr:`right`) are returned.
        Use ``initial=True`` to return the nominal edges regardless
        of the presence of the tweaked edges.

        Args:
            initial (:obj:`bool`, optional):
                To use the initial edges regardless of the presence of
                the tweaked edges, set this to True.
            median (:obj:`bool`, optional):
                The default is to return the slit length as a function
                of the spectral coordinate. If median is set to true,
                the median slit length of each slit is returned.

        Returns:
            `numpy.ndarray`_: Slit lengths.
        """
        left, right, _ = self.select_edges(initial=initial)
        slitlen = right - left
        if median is True:
            slitlen = np.median(slitlen, axis=1)
        return slitlen

    def get_radec_image(self, wcs, initial=True, flexure=None, trace_cen=None):
        """Generate an RA and DEC image for every pixel in the frame

        Parameters
        ----------
        wcs : astropy.wcs
            The World Coordinate system of a science frame
        maxslitlen : int
            This is the slit length in pixels, and it should be the same
            value that was passed to get_wcs() to generate the WCS that
            is passed into this function as an argument.
        initial : bool
            Select the initial slit edges?
        flexure : float, optional
            If provided, offset each slit by this amount.
        trace_cen : `numpy.ndarray`_, optional
            Central traces of each slit. Shape should be (slits.nspec, slits.nslits).
            If None, the average of the left and right slit edges will be used

        Returns
        -------
        tuple : There are three elements in the tuple. The first two are 2D numpy arrays
                of shape (nspec, nspat), where the first ndarray is the RA image, and the
                second ndarray is the DEC image. RA and DEC are in units degrees. The third
                element of the tuple stores the minimum and maximum difference (in pixels)
                between the WCS reference (usually the centre of the slit) and the edge of
                the slits. The third array has a shape of (nslits, 2).
        """
        # Grab the central trace, if none was provided
        if trace_cen is None:
            left, right, _ = self.select_edges(initial=initial, flexure=flexure)
            trace_cen = 0.5 * (left + right)

        # Initialise the output
        raimg = np.zeros((self.nspec, self.nspat))
        decimg = np.zeros((self.nspec, self.nspat))
        minmax = np.zeros((self.nslits, 2))

        # Get the slit information
        slitid_img_init = self.slit_img(pad=0, initial=initial, flexure=flexure)
        for slit_idx, spatid in enumerate(self.spat_id):
            onslit = (slitid_img_init == spatid)
            onslit_init = np.where(onslit)
            evalpos = onslit_init[1] - trace_cen[onslit_init[0], slit_idx]
            minmax[:, 0] = np.min(evalpos)
            minmax[:, 1] = np.max(evalpos)
            slitID = np.ones(evalpos.size) * slit_idx - wcs.wcs.crpix[0]
            world_ra, world_dec, _ = wcs.wcs_pix2world(slitID, evalpos, onslit_init[0], 0)
            # Set the RA first and DEC next
            raimg[onslit] = world_ra.copy()
            decimg[onslit] = world_dec.copy()
        return raimg, decimg, minmax

    def select_edges(self, initial=False, flexure=None):
        """
        Select between the initial or tweaked slit edges and allow for
        flexure correction.

        By default, the method will return the tweaked slits if they
        have been defined. If they haven't been defined the nominal
        edges (:attr:`left` and :attr:`right`) are returned. Use
        ``original=True`` to return the nominal edges regardless of
        the presence of the tweaked edges.

        Args:
            initial (:obj:`bool`, optional):
                To use the initial edges regardless of the presence of
                the tweaked edges, set this to True.
            flexure (:obj:`float`, optional):
                If provided, offset each slit by this amount

        Returns:
            tuple: Returns the full arrays containing the left and right
            edge coordinates and the mask, respectively.
            These are returned as copies.
        """
        # TODO: Add a copy argument?
        if self.left_tweak is not None and self.right_tweak is not None and not initial:
            left, right = self.left_tweak, self.right_tweak
        else:
            left, right = self.left_init, self.right_init

        # Add in spatial flexure?
        if flexure:
            self.left_flexure = left + flexure
            self.right_flexure = right + flexure
            left, right = self.left_flexure, self.right_flexure

        # Return
        return left.copy(), right.copy(), self.mask.copy()

    def slit_img(self, pad=None, slitidx=None, initial=False, flexure=None,
                 exclude_flag=None, use_spatial=True):
        r"""
        Construct an image identifying each pixel with its associated
        slit.

        The output image has the same shape as the original trace
        image. Each pixel in the image is set to the index
        of its associated slit (i.e, the pixel value is
        :math:`0..N_{\rm slit}-1`). Pixels not associated with any
        slit are given values of -1.

        The width of the slit is extended at either edge by a fixed
        number of pixels using the `pad` parameter in :attr:`par`.
        This value can be overridden using the method keyword
        argument.

        .. warning::

            - The function does not check that pixels end up in
              multiple pixels or that the padding is sensible given
              the separation between slit edges!
            - All slits are identified in the image, even if they are
              masked with :attr:`mask`.

        Args:
            pad (:obj:`float`, :obj:`int`, :obj:`tuple`, optional):
                The number of pixels used to pad (extend) the edge of
                each slit. This can be a single scale to pad both
                left and right edges equally or a 2-tuple that
                provides separate padding for the left (first
                element) and right (2nd element) edges separately. If
                not None, this overrides the value in :attr:`par`.
                The value can be negative, which means that the
                widths are **trimmed** instead of padded.
            slitidx (:obj:`int`, array_like, optional):
                List of indexes (zero-based) to include in the image.
                If None, all slits not flagged are included.
            initial (:obj:`bool`, optional):
                By default, the method will use the tweaked slit
                edges if they have been defined. If they haven't
                been, the initial edges (:attr:`left_init` and
                :attr:`right_init`) are used. To use the nominal edges
                regardless of the presence of the tweaked edges, set
                this to True. See :func:`select_edges`.
            exclude_flag (:obj:`str`, optional):
                Bitmask flag to ignore when masking
                Warning -- This could conflict with input slitids, i.e. avoid using both
            use_spatial (bool, optional):
                If True, use self.spat_id value instead of 0-based indices


        Returns:
            `numpy.ndarray`_: The image with the slit index
            identified for each pixel.
        """
        #
        if slitidx is not None and exclude_flag is not None:
            msgs.error("Cannot pass in both slitidx and exclude_flag!")
        # Check the input
        if pad is None:
            pad = self.pad
        _pad = pad if isinstance(pad, tuple) else (pad,pad)
        if len(_pad) != 2:
            msgs.error('Padding for both left and right edges should be provided as a 2-tuple!')

        # Pixel coordinates
        spat = np.arange(self.nspat)
        spec = np.arange(self.nspec)

        left, right, _ = self.select_edges(initial=initial, flexure=flexure)

        # Choose the slits to use
        if slitidx is not None:
            slitidx = np.atleast_1d(slitidx).ravel()
        else:
            bpm = self.mask.astype(bool)
            if exclude_flag:
                bpm &= np.invert(self.bitmask.flagged(self.mask, flag=exclude_flag))
            slitidx = np.where(np.invert(bpm))[0]

        # TODO: When specific slits are chosen, need to check that the
        # padding doesn't lead to slit overlap.

        # Find the pixels in each slit, limited by the minimum and
        # maximum spectral position.
        slitid_img = np.full((self.nspec,self.nspat), -1, dtype=int)
        for i in slitidx:
            slit_id = self.spat_id[i] if use_spatial else i
            indx = (spat[None,:] > left[:,i,None] - _pad[0]) \
                        & (spat[None,:] < right[:,i,None] + _pad[1]) \
                        & (spec > self.specmin[i])[:,None] & (spec < self.specmax[i])[:,None]
            slitid_img[indx] = slit_id
        # Return
        return slitid_img

    def spatial_coordinate_image(self, slitidx=None, full=False, slitid_img=None,
                                 pad=None, initial=False, flexure_shift=None):
        r"""
        Generate an image with the normalized spatial coordinate
        within each slit.

        Args:
            slitidx (:obj:`int`, array_like, optional):
                List of indices to include in the image. If None,
                all slits are included.
            full (:obj:`bool`, optional):
                If True, return a full image with the coordinates
                based on the single provided slit. In this case,
                slitids must be provided and it can be only one slit!
                If False, only those coordinates falling in the slit
                (as determined by ``slitid_image`` or by a call to
                :func:`slit_img`) will be returned.
            slitid_img (`numpy.ndarray`_, optional):
                Image identifying the slit associated with each
                pixel. Should have the same shape as expected by the
                object (:attr:`nspec` by :attr:`nspat`). Pixels not
                associated with any pixel have a value of -1. This is
                a convenience parameter that allows previous
                executions of :func:`slit_img` to be passed in
                directly instead of having to repeat the operation. A
                slit ID image is only required if ``full`` is False.
            pad (:obj:`float`, :obj:`int`, optional):
                When constructing the slit ID image, use this value
                to override the value of `pad` in :attr:`par`. Only
                used if ``slitid_img`` is not provided directly and
                ``full`` is False.
            initial (:obj:`bool`, optional):
                By default, the method will use the tweaked slit
                edges if they have been defined. If they haven't
                been, the nominal edges (:attr:`left` and
                :attr:`right`) are used. To use the nominal edges
                regardless of the presence of the tweaked edges, set
                this to True. See :func:`select_edges`.

        Returns:
            `numpy.ndarray`_: Array specifying the spatial coordinate
            of pixel in its own slit, scaled to go from 0 to 1. If
            ``full`` is True, the image provides the coordinates
            relative to the left edge for the provided slit over the
            full image.
        """
        # Slit indices to include
        _slitidx = np.arange(self.nslits) if slitidx is None else np.atleast_1d(slitidx).ravel()
        if full and len(_slitidx) > 1:
            msgs.error('For a full image with the slit coordinates, must select a single slit.')

        # Generate the slit ID image if it wasn't provided
        if not full:
            if slitid_img is None:
                slitid_img = self.slit_img(pad=pad, slitidx=_slitidx, initial=initial)
            if slitid_img.shape != (self.nspec,self.nspat):
                msgs.error('Provided slit ID image does not have the correct shape!')

        # Choose the slit edges to use
        left, right, _ = self.select_edges(initial=initial, flexure=flexure_shift)

        # Slit width
        slitwidth = right - left

        # TODO: This check should go in it's own function and/or
        # checked when instantiating the object.
        # Slit widths have to be larger than 0
        indx = slitwidth <= 0.
        if np.any(indx[:,_slitidx]):
            bad_slits = np.where(np.any(indx, axis=0))[0]
            # TODO: Shouldn't this fault?
            msgs.warn('Slits {0} have negative (or 0) slit width!'.format(bad_slits))

        # Output image
        coo_img = np.zeros((self.nspec,self.nspat), dtype=float)
        spat = np.arange(self.nspat)
        for i in _slitidx:
            coo = (spat[None,:] - left[:,i,None])/slitwidth[:,i,None]
            if not full:
                indx = slitid_img == self.spat_id[i]
                coo_img[indx] = coo[indx]
            else:
                coo_img = coo
        return coo_img

    def spatial_coordinates(self, initial=False, flexure=None):
        """
        Return a fiducial coordinate for each slit.

        This is a simple wrapper for :func:`select_edges` and
        :func:`slit_spat_pos`.

        Args:
            original (:obj:`bool`, optional):
                By default, the method will use the tweaked slit
                edges if they have been defined. If they haven't
                been, the nominal edges (:attr:`left` and
                :attr:`right`) are used. To use the nominal edges
                regardless of the presence of the tweaked edges, set
                this to True. See :func:`select_edges`.

        Returns:
            `numpy.ndarray`_: Vector with the list of floating point
            spatial coordinates.
        """
        # TODO -- Confirm it makes sense to pass in flexure
        left, right, _ = self.select_edges(initial=initial, flexure=flexure)
        return SlitTraceSet.slit_spat_pos(left, right, self.nspat)

    @staticmethod
    def slit_spat_pos(left, right, nspat):
        r"""
        Return a fidicial, normalized spatial coordinate for each slit.

        The fiducial coordinates are given by::
   
            nspec = left.shape[0]
            (left[nspec//2,:] + right[nspec//2,:])/2/nspat

        Args:
            left (`numpy.ndarray`_):
                Array with left slit edges. Shape is :math:`(N_{\rm
                spec},N_{\rm slits})`.
            right (`numpy.ndarray`_):
                Array with right slit edges. Shape is :math:`(N_{\rm
                spec},N_{\rm slits})`.
            nspat (:obj:`int`):
                Number of pixels in the spatial direction in the image
                used to trace the slit edges.

        Returns:
            `numpy.ndarray`_: Vector with the list of floating point
            spatial coordinates.
        """
        if left.shape != right.shape:
            msgs.error('Left and right traces must have the same shape.')
        nspec = left.shape[0]
        return (left[nspec//2,:] + right[nspec//2,:])/2/nspat

    def mask_add_missing_obj(self, sobjs, expected_objpos, fwhm, median_off, slits_left, slits_right):
        """
        Generate new SpecObj and add them into the SpecObjs object for any slits missing the targeted source.

        Args:
            sobjs (:class:`pypeit.specobjs.SpecObjs`): List of SpecObj that have been found and traced
            expected_objpos (`numpy.ndarray`_): Array with expected positions of all objects from slitmask design info.
            fwhm (:obj:`float`): FWHM in pixels to be used in the optimal extraction
            median_off (:obj:`float`): User provided median offset in pixels of the slitmask
            slits_left (`numpy.ndarray`_): Array with left slit edges.
            slits_right (`numpy.ndarray`_): Array with right slit edges.

        Returns:
            tuple:
            sobjs (:class:`pypeit.specobjs.SpecObjs`): Updated list of SpecObj that have been found and traced
            nobj (:obj:`int`): Number of objects identified

        """

        if fwhm is None:
            msgs.error('A FWHM for the optimal extraction must be provided. See `find_fwhm` in `FindObjPar`.')

        if median_off is not None:
            self.mask_median_off = median_off
            msgs.info('Slitmask median offset: {} pixels'.format(round(self.mask_median_off, 2)))
        else:
            self.mask_median_off = 0.
            msgs.info('Slitmask median offset is assumed to be zero. '
                      'To change this see `slitmask_offset` in `SlitMaskPar`.')

        # midpoint in the spectral direction
        specmid = slits_left[:,0].size//2
        # slit center trace
        slit_cen = (slits_left + slits_right) / 2.

        # Loop on all the good slits
        gd_slit = self.mask == 0
        for islit in np.where(gd_slit)[0]:
            # Check for assigned obj
            if len(sobjs) > 0:
                target_obj_in_slit = (sobjs.MASKDEF_ID == self.maskdef_id[islit]) & \
                                     (sobjs.MASKDEF_OBJNAME != 'SERENDIP')
                if np.any(target_obj_in_slit):
                    continue

            # Object index
            oidx = np.where(self.maskdef_designtab['SLITID'].data == self.maskdef_id[islit])[0][0]
            #
            # Do it
            SPAT_PIXPOS = float((expected_objpos[oidx]-self.mask_median_off) + slits_left[specmid, islit])

            # TODO -- DO THIS RIGHT FOR SLITS (LIKELY) OFF THE DETECTOR/SLIT
            #  If we keep what follows, probably should add some tolerance to be off the edge
            #  Otherwise things break in skysub
            if (SPAT_PIXPOS > slits_right[specmid, islit]) or (SPAT_PIXPOS < slits_left[specmid, islit]):
                msgs.warn("Targeted object is off the detector")
                continue

            # Generate a new specobj
            specobj_dict = {'SLITID': self.spat_id[islit], # Confirm this
                            'DET': self.det, 
                            'OBJTYPE': 'science',  # Confirm this is ok
                            'PYPELINE': self.pypeline}
            thisobj = specobj.SpecObj(**specobj_dict)

            # Fill me up
            if sobjs.nobj == 0:
                # This uses slit edge trace
                xoff = SPAT_PIXPOS - slit_cen[specmid, islit]
                thisobj.TRACE_SPAT = slit_cen[:, islit] + xoff
                thisobj.SPAT_PIXPOS = SPAT_PIXPOS
                thisobj.SPAT_FRACPOS = (SPAT_PIXPOS - slits_left[specmid, islit]) / \
                                       (slits_right[specmid, islit]-slits_left[specmid, islit])
                thisobj.trace_spec = np.arange(slits_left.shape[0])
                # OBJID
                thisobj.OBJID = 1
            else:
                # This uses an object
                idx_nearest = np.argmin(np.abs(sobjs.SPAT_PIXPOS-SPAT_PIXPOS))
                specmid = sobjs[idx_nearest].TRACE_SPAT.size//2
                xoff = SPAT_PIXPOS - sobjs[idx_nearest].TRACE_SPAT[specmid]
                thisobj.TRACE_SPAT = sobjs[idx_nearest].TRACE_SPAT + xoff
                thisobj.SPAT_PIXPOS = SPAT_PIXPOS
                thisobj.trace_spec = sobjs[idx_nearest].trace_spec
                thisobj.SPAT_FRACPOS = (SPAT_PIXPOS - slits_left[specmid, islit]) / \
                                       (slits_right[specmid, islit]-slits_left[specmid, islit])
                # OBJID
                any_obj_in_slit = sobjs.MASKDEF_ID == self.maskdef_id[islit]
                if np.any(any_obj_in_slit):
                    thisobj.OBJID = np.max(sobjs[any_obj_in_slit].OBJID) + 1
                else:
                    thisobj.OBJID = 1

            # FWHM
            thisobj.FWHM = fwhm  # pixels
            thisobj.maskwidth = 4. * fwhm  # matches objfind() in extract.py
            # Finishing up
            thisobj.set_name()
            # Mask info
            thisobj.RA = self.maskdef_designtab['OBJRA'][oidx]
            thisobj.DEC = self.maskdef_designtab['OBJDEC'][oidx]
            thisobj.MASKDEF_OBJNAME = self.maskdef_designtab['OBJNAME'][oidx]
            thisobj.MASKDEF_ID = self.maskdef_designtab['SLITID'][oidx]
            thisobj.MASKDEF_EXTRACT = True
            thisobj.hand_extract_flag = False
            # Add to SpecObjs
            sobjs.add_sobj(thisobj)

        # Sort objects according to their spatial location
        spat_pixpos = sobjs.SPAT_PIXPOS
        sobjs = sobjs[spat_pixpos.argsort()]

        # Vette
        for sobj in sobjs:
            if not sobj.ready_for_extraction():
                msgs.error("Bad SpecObj.  Can't proceed")

        # Return
        return sobjs

    def assign_maskinfo(self, sobjs, plate_scale, slits_left, slits_right, det_buffer, TOLER=1.):
        """
        Assign RA, DEC, Name to objects
        Modified in place

        Args:
            sobjs (:class:`pypeit.specobjs.SpecObjs`): List of SpecObj that have been found and traced
            plate_scale (:obj:`float`): platescale for the current detector
            slits_left (`numpy.ndarray`_): Array with left slit edges.
            slits_right (`numpy.ndarray`_): Array with right slit edges.
            det_buffer (:obj:`int`): Minimum separation between detector edges and a slit edge
            TOLER (:obj:`float`, optional): Matching tolerance in arcsec

        Returns:
            Array with expected positions of all objects from slitmask design info.
        """

        # midpoint in the spectral direction
        specmid = slits_left[:,0].size//2
        # slit center trace
        slit_cen = (slits_left + slits_right) / 2.

        # Unpack -- Remove this once we have a DataModel
        obj_maskdef_id = self.maskdef_designtab['SLITID'].data
        obj_slit_coords = SkyCoord(ra=self.maskdef_designtab['SLITRA'],
                                   dec=self.maskdef_designtab['SLITDEC'], frame='fk5', unit='deg')
        obj_slit_pa = self.maskdef_designtab['SLITPA']
        # Distance (arcsec) of the object from the left edge
        obj_topdist = self.maskdef_designtab['OBJ_TOPDIST'].data
        obj_botdist = self.maskdef_designtab['OBJ_BOTDIST'].data

        # slit lengths
        expected_slitlen = (obj_topdist + obj_botdist) / plate_scale  # pixels
        measured_slitlen = slits_right[specmid, :] - slits_left[specmid, :]  # pixels
        # difference between measured and expected slit length (but only for the left side).
        left_edgeloss = np.zeros(self.maskdef_id.size)
        # define a new slit center to take into account the slits that are smaller than what
        # should be because they are cut by the detector edges.
        new_slitcen = slit_cen.copy()
        for i, maskid in enumerate(self.maskdef_id):
            if maskid == -99:
                left_edgeloss[i] = -9999.9
            else:
                if (slits_left[specmid, i] != det_buffer) & (slits_right[specmid, i] != self.nspat - det_buffer):
                    # for all but the slits that fall outside the detector, we assume the edge loss happens
                    # equally in both sides of the slit
                    left_edgeloss[i] = (expected_slitlen[obj_maskdef_id == maskid] - measured_slitlen[i]) / 2.  # pixels
                    # because edge loss happens equally in both sides, we assume no shift in slit center

        # Stats (typical edge loss across the detector)
        if left_edgeloss[(left_edgeloss != 0)&(left_edgeloss != -9999.9)].size > 3:
            _, median_edgeloss, _ = sigma_clipped_stats(left_edgeloss[(left_edgeloss != 0)&(left_edgeloss != -9999.9)],
                                                        sigma=2.)
        else:
            median_edgeloss = 0.

        expected_objpos_all = np.zeros(obj_maskdef_id.size)
        for i, maskid in enumerate(obj_maskdef_id):
            # fix the edge loss for slits that fall off the detector
            if slits_left[specmid, i] == det_buffer:
                # for the leftmost slit that is cut by the detector edge, we assume that the edge loss
                # happens mostly in the left side and we assume a value (median_edgeloss) for the right side
                left_edgeloss[i] = expected_slitlen[obj_maskdef_id == maskid] - measured_slitlen[i] \
                                   - median_edgeloss  # pixels
                # shift in slit center due to slit partially falling outside the detector
                censhift = left_edgeloss / 2.
                new_slitcen[i] -= censhift
            if slits_right[specmid, i] == self.nspat - det_buffer:
                # for the rightmost slit that is cut by the detector edge, we assume that the edge loss
                # happens mostly in the right side, and `left_edgeloss` is assumed to be equal to `median_edgeloss`.
                left_edgeloss[i] = median_edgeloss
                # shift in slit center due to slit partially falling outside the detector
                censhift = (expected_slitlen[obj_maskdef_id == maskid] - measured_slitlen[i] - median_edgeloss) / 2.
                new_slitcen[i] += censhift

            # Expected objects position (distance from left edge) in pixels, corrected for edge loss
            expected_objpos_all[i] = obj_topdist[i] / plate_scale - left_edgeloss[self.maskdef_id == maskid][0]

        if sobjs.nobj == 0:
            msgs.warn('NO detected objects')
            return expected_objpos_all
        else:
            # Restrict to objects on this detector
            on_det = sobjs.DET == self.det
            cut_sobjs = sobjs[on_det]

            # First pass
            measured, expected = [], []
            for sobj in cut_sobjs:
                # Set MASKDEF_ID
                sobj.MASKDEF_ID = self.maskdef_id[self.spat_id == sobj.SLITID][0]
                # object ID
                # TODO -- Add to SpecObj DataModel?
                # There is small chance that self.maskdef_id=-99. This would definitely happen if the user
                # add a custom slit. If maskdef_id=-99 for a certain object, we cannot assign OBJECT, RA, DEC
                oidx = np.where(obj_maskdef_id == sobj.MASKDEF_ID)[0]
                if oidx.size == 0:
                    # In this case I have to give a fake value for index reasons later in the code.
                    measured.append(-9999.9)
                    expected.append(9999.9)
                else:
                    oidx = oidx[0]
                    # Expected object position (distance from left edge) in pixels, corrected for edge loss
                    expected_objpos = expected_objpos_all[oidx]
                    # Measured object position (distance from left edge) in pixels
                    measured_objpos = sobj.SPAT_PIXPOS - slits_left[specmid, self.maskdef_id == sobj.MASKDEF_ID][0]
                    # Finish
                    measured.append(measured_objpos)
                    expected.append(expected_objpos)
            measured = np.array(measured)
            expected = np.array(expected)

            # # Stats (offset between expected and measured object position in pixels)
            # if len(expected) > 3:
            #     mean, median_off, std = sigma_clipped_stats(
            #         expected[expected != 9999.9] - measured[measured != -9999.9], sigma=2.)
            # else:
            #     median_off = 0.

            # Assign
            # Loop on slits to deal with multiple sources within TOLER
            # Exclude the objects that have maskdef_id=-99
            uni_maskid = np.unique(cut_sobjs.MASKDEF_ID[cut_sobjs.MASKDEF_ID!=-99])
            for maskid in uni_maskid:
                # Index for SpecObjs on this slit
                idx = np.where(cut_sobjs.MASKDEF_ID == maskid)[0]
                # Index for slitmask
                sidx = np.where(self.maskdef_designtab['SLITID'] == maskid)[0][0]
                # Within TOLER?
                # separation in pixels
                separ = expected[idx]-measured[idx]
                msgs.info('MASKDEF_ID:{}'.format(maskid))
                msgs.info('Difference between expected and detected object '
                          'positions: {} arcsec'.format(np.round(separ*plate_scale, 3)))
                in_toler = np.abs(separ*plate_scale) < TOLER
                if np.any(in_toler):
                    # Parse the peak fluxes
                    peak_flux = cut_sobjs[idx].smash_peakflux[in_toler]
                    imx_peak = np.argmax(peak_flux)
                    imx_idx = idx[in_toler][imx_peak]
                    # Object in Mask Definition
                    oidx = np.where(obj_maskdef_id == maskid)[0][0]
                    # Assign
                    sobj = cut_sobjs[imx_idx]
                    sobj.RA = self.maskdef_designtab['OBJRA'][oidx]
                    sobj.DEC = self.maskdef_designtab['OBJDEC'][oidx]
                    sobj.MASKDEF_OBJNAME = self.maskdef_designtab['OBJNAME'][oidx]
                    sobj.MASKDEF_EXTRACT = False
                    sobj.hand_extract_flag = False
                    # Remove that idx value
                    idx = idx.tolist()
                    idx.remove(imx_idx)
                    idx = np.array(idx)
                # Fill in the rest
                for ss in idx:
                    sobj = cut_sobjs[ss]
                    # Measured coordinates
                    offset = measured[ss] - (new_slitcen - slits_left)[specmid, self.spat_id == sobj.SLITID][0]
                    new_obj_coord = obj_slit_coords[sidx].directional_offset_by(
                        np.radians(obj_slit_pa[sidx]), (offset*plate_scale)*units.arcsec)
                    # Assign
                    sobj.RA = new_obj_coord.ra.value
                    sobj.DEC = new_obj_coord.dec.value
                    sobj.MASKDEF_OBJNAME = 'SERENDIP'
                    sobj.MASKDEF_EXTRACT = False
                    sobj.hand_extract_flag = False
            # Give fake values of RA, DEC, and MASKDEF_OBJNAME for object with maskdef_id=-99.
            noidx = np.where(cut_sobjs.MASKDEF_ID == -99)[0]
            if noidx.size > 0:
                for sobj in cut_sobjs[noidx]:
                    # Assign
                    sobj.RA = 0.0
                    sobj.DEC = 0.0
                    sobj.MASKDEF_OBJNAME = 'NONE'
                    sobj.MASKDEF_EXTRACT = False
                    sobj.hand_extract_flag = False

        # Return
        return expected_objpos_all

    def user_mask(self, det, slitspatnum):
        """
        Mask all but the input slit

        Args:
            det (:obj:`int`): Detector number
            slitspatnum (:obj:`str` or :obj:`list`):
        """
        # Parse
        dets, spat_ids = parse_slitspatnum(slitspatnum)
        if det not in dets:
            return
        # Cut down for convenience
        indet = dets == det
        spat_ids = spat_ids[indet]
        #
        msk = np.ones(self.nslits, dtype=bool)
        for slit_spat in spat_ids:
            #TODO -- Consider putting in a tolerance which if not met causes a crash
            idx = np.argmin(np.abs(self.spat_id - slit_spat))
            msk[idx] = False
        self.mask[msk] = self.bitmask.turn_on(self.mask[msk], 'USERIGNORE')

    def mask_flats(self, flatImages):
        """
        Mask from a :class:`pypeit.flatfield.Flats` object

        Args:
            flatImages (:class:`pypeit.flatfield.FlatImages`):

        """
        # Loop on all the FLATFIELD BPM keys
        for flag in ['SKIPFLATCALIB', 'BADFLATCALIB']:
            bad_flats = self.bitmask.flagged(flatImages.get_bpmflats(), flag)
            if np.any(bad_flats):
                self.mask[bad_flats] = self.bitmask.turn_on(self.mask[bad_flats], flag)

    def mask_wvcalib(self, wv_calib):
        """
        Mask from a WaveCalib object

        Args:
            wv_calib (:obj:`dict`):

        """
        for islit in range(self.nslits):
            if wv_calib.wv_fits[islit] is None or wv_calib.wv_fits[islit].pypeitfit is None:
                self.mask[islit] = self.bitmask.turn_on(self.mask[islit], 'BADWVCALIB')

    def mask_wavetilts(self, waveTilts):
        """
        Mask from a :class:`pypeit.wavetilts.WaveTilts` object

        Args:
            waveTilts (:class:`pypeit.wavetilts.WaveTilts`):

        """
        # There is only one BPM for Tilts (so far)
        bad_tilts = waveTilts.bpmtilts > 0
        if np.any(bad_tilts):
            self.mask[bad_tilts] = self.bitmask.turn_on(self.mask[bad_tilts], 'BADTILTCALIB')


def parse_slitspatnum(slitspatnum):
    """
    Parse the slitspatnum into a list of detectors and SPAT_IDs

    Args:
        slitspatnum (:obj:`str` or :obj:`list`:

    Returns:
        tuple:  dets, spat_ids  (each is an `numpy.ndarray`_ of int's)

    """
    dets = []
    spat_ids = []
    for item in slitspatnum.split(','):
        spt = item.split(':')
        dets.append(int(spt[0]))
        spat_ids.append(int(spt[1]))
    # Return
    return np.array(dets).astype(int), np.array(spat_ids).astype(int)
