.. include:: ../include/links.rst

.. _deimos_slitmask_ids_report:

DEIMOS slitmask ID assignment and missing slits
===============================================

Version History
---------------

=========   ================   =========== ===========
*Version*   *Author*           *Date*      ``PypeIt``
=========   ================   =========== ===========
1.0         Debora Pelliccia   9 Oct 2020  1.1.2dev
1.1         Debora Pelliccia   5 Nov 2020  1.2.1dev
1.2         Debora Pelliccia   26 Jan 2021 1.3.1dev
=========   ================   =========== ===========

----

Basics
------

The procedure used to assign DEIMOS slitmask ID to each slit 
is part of the more general slit tracing procedure described in :ref:`slit_tracing`.

Procedure
---------

Slitmask ID assignment and missing slits
++++++++++++++++++++++++++++++++++++++++

DEIMOS slitmask ID assignment is primarily performed by
:func:`pypeit.edgetrace.EdgeTraceSet.maskdesign_matching`. This function matches the slit 
edges traced by ``PypeIt`` to the slit edges predicted by the DEIMOS optical model. These 
predictions are generated, using the slitmask design information stored in the DEIMOS data, 
by the function :func:`pypeit.spectrographs.keck_deimos.KeckDEIMOSSpectrograph.mask_to_pixel_coordinates`,
which relies on the existence of pre-generated detectors maps pre- and post-grating,
called `amap` and `bmap`, respectively.

The function :func:`~pypeit.edgetrace.EdgeTraceSet.maskdesign_matching` assigns to each slit 
a ``maskdef_id``, which corresponds to `dSlitId` in the DEIMOS slitmask design. Moreover,
:func:`~pypeit.edgetrace.EdgeTraceSet.maskdesign_matching` uses the prediction from the optical
model to add slit traces that have not been found in the image.

Overlapping alignment slits
+++++++++++++++++++++++++++

Because ``PypeIt`` uses the prediction from the optical model to add missing slit traces,
it is able to trace overlapping alignment slits that otherwise would be identify as one large
alignment slit from the image, i.e., with width larger than 4". To avoid that the edges of overlapping
slits cross, those edges are placed adjacent to one other.

Application
-----------

To perform the slitmask ID assignment, the **use_maskdesign** flag in :ref:`pypeit_par:EdgeTracePar Keywords`
must be **True**.  This is the default for DEIMOS, except when the *LongMirr* or the *LVM* mask is used.

Three other :ref:`pypeit_par:EdgeTracePar Keywords` control the slitmask ID assignment;
these are: **maskdesign_maxsep**, **maskdesign_sigrej**, **maskdesign_step**.


Access
------

The ``maskdef_id`` is recorded for each slits in the :class:`~pypeit.slittrace.SlitTraceSet` datamodel,
which is written to disk as a multi-extension FITS file prefixed by MasterSlits.
In addition, for DEIMOS a second `astropy.io.fits.BinTableHDU`_ is written to disk and contains
more DEIMOS slitmask design information. See :ref:`master_slits` for
a description of the provided information and for a way to visualize them.

Moreover, the ``maskdef_id`` assigned to each slit can be found, after a full reduction with ``PypeIt``
(see :ref:`step_by_step`), by running ``pypeit_chk_2dslits Science/spec2d_XXX.fits``, which lists all
the slits with their associated ``maskdef_id``.


Testing
-------

Requirement PD-9 states: "Ingest slitmask information into a data structure."

Requirement PD-10 states: "Use mask information to determine initial guess for slits positions."

Requirement PD-42 states: "Deal with overlapping alignment boxes that exceed 4”. "

``PypeIt`` meets these requirements as demonstrated by the three tests at
``pypeit/tests/test_maskdesign_matching.py``.  To run the tests:

.. code-block:: bash

    cd pypeit/tests
    pytest test_maskdesign_matching.py::test_maskdef_id -W ignore
    pytest test_maskdesign_matching.py::test_add_missing_slits -W ignore
    pytest test_maskdesign_matching.py::test_overlapped_slits -W ignore

The tests require that you have downloaded the ``PypeIt`` :ref:`dev-suite` and defined
the ``PYPEIT_DEV`` environmental variable that points to the relevant directory. These tests are
run using only one instrument setup and only one DEIMOS detector.

First test ``pytest test_maskdesign_matching.py::test_maskdef_id -W ignore``.
The algorithm of this test is as follows:

    1. Load the information relative to the specific instrument (DEIMOS).

    2. Load the :ref:`pypeit_par:Instrument-Specific Default Configuration` parameters and select the detector.

    3. Build a trace image using three flat-field images from a specific DEIMOS dataset in the :ref:`dev-suite`.

    4. Update the DEIMOS configuration parameters to include configurations specific for the
       used instrument setup. Among others, this step sets the **use_maskdesign** flag in 
       :ref:`pypeit_par:EdgeTracePar Keywords` to *True*.

    5. Run the slit tracing procedure using :class:`~pypeit.edgetrace.EdgeTraceSet`, during which
       the slitmask ID assignment is performed, and record the ``maskdef_id`` associated to each slit
       in the :class:`~pypeit.slittrace.SlitTraceSet` datamodel.

    6. Read the ``maskdef_id`` for the first and last slits of the selected detector and check if
       those correspond to the expected values. The expected values are determined by previously 
       reducing the same dataset with the already existing *DEEP2 IDL-based pipeline*, which is 
       optimized to match the slits found on the DEIMOS detectors to the slitmask information.


Second test ``pytest test_maskdesign_matching.py::test_add_missing_slits -W ignore``.
The algorithm of this test is as follows:

    1. Load the information relative to the specific instrument (DEIMOS).

    2. Load the :ref:`pypeit_par:Instrument-Specific Default Configuration` parameters and select the detector.

    3. Build a trace image using three flat-field images from a specific DEIMOS dataset in the :ref:`dev-suite`.

    4. Update the DEIMOS configuration parameters to include configurations specific for the
       used instrument setup. Among others, this step sets the **use_maskdesign** flag in
       :ref:`pypeit_par:EdgeTracePar Keywords` to **True**.

    5. Run step-by-step (lines 72-100) the slit tracing procedure performed by :class:`~pypeit.edgetrace.EdgeTraceSet`.
       This enable to add an extra step, where we remove 4 edges (lines 112-118) that were found
       by the slit tracing procedure.

    6. Continue with the remaining steps of :class:`~pypeit.edgetrace.EdgeTraceSet`, including
       :func:`~pypeit.edgetrace.EdgeTraceSet.maskdesign_matching`, which adds the missing slits and
       performs the the slitmask ID assignment.

    7. Check that the pixel position of the traces that were previously removed are now recovered.


Third test ``pytest test_maskdesign_matching.py::test_overlapped_slits -W ignore``.
The algorithm of this test is identical to the first test, but using a different dataset that shows
overlapping alignment slits. The algorithm, then, checks that the total number of slits and the number alignment
slits are as expected. The expected numbers are determined by visual inspection of the observations.

Because these tests are now included in the ``PypeIt`` :ref:`unit-tests`, these checks are performed by the
developers for every new version of the code.

The ``PypeIt`` team have carefully tested ~10 datasets, but more tests are welcome.