"""
Fit telluric absorption to observed spectra

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

import os
from pkg_resources import resource_filename

from astropy.io import fits

from pypeit import par, msgs
from pypeit.par import pypeitpar
from pypeit.spectrographs.util import load_spectrograph
from pypeit.core import telluric
from pypeit.scripts import scriptbase


def read_tellfile(ifile):
    """
    Read a PypeIt telluric file, akin to a standard PypeIt file

    The top is a config block that sets ParSet parameters
      The spectrograph is not required

    Parameters
    ----------
    ifile: str
        Name of the flux file

    Returns
    -------
    cfg_lines: list
        Config lines to modify ParSet values
    """

    # Read in the pypeit reduction file
    msgs.info('Loading the telluric file')
    lines = par.util._read_pypeit_file_lines(ifile)

    # Construct config to get spectrograph
    cfg_lines = list(lines)

    # Return
    return cfg_lines


class TellFit(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        parser = super().get_parser(description='Telluric correct a spectrum',
                                    width=width, formatter=scriptbase.SmartFormatter)
        parser.add_argument("spec1dfile", type=str,
                            help="spec1d file that will be used for telluric correction.")
        parser.add_argument("--objmodel", type=str, default=None, choices=['qso', 'star', 'poly'],
                            help="R|science object model used in the fitting.\n"
                            "The options are:\n"
                            "\n"
                            "    qso  = For quasars. You might need to set redshift, bal_wv_min_max in the tell file.\n"
                            "\n"
                            "    star  = For stars. You need to set star_type, star_ra, star_dec, and star_mag in the tell_file.\n"
                            "\n"
                            "    poly = For other type object, You might need to set fit_wv_min_max, \n"
                            "           and norder in the tell_file."
                            )
        parser.add_argument("-r", "--redshift", type=float, default=None,
                            help="Specify redshift. Used with the --objmodel qso option above.")
        parser.add_argument("-g", "--tell_grid", type=str,
                            help='Telluric grid. You should download the giant grid file to the '
                                 'pypeit/data/telluric folder. It should only be passed if you '
                                 'want to overwrite the default tell_grid that is set via each '
                                 'spectrograph file.')
        parser.add_argument("-p", "--pca_file", type=str,
                            help='Quasar PCA fits file with full path. The default file '
                                 '(qso_pca_1200_3100.fits) is stored in the pypeit/data/telluric '
                                 'folder. If you change the fits file, make sure to set the '
                                 'pca_lower and pca_upper in the tell_file to specify the '
                                 'wavelength coverage of your model. The defaults are '
                                 'pca_lower=1220. and pca_upper=3100.')
        parser.add_argument("-t", "--tell_file", type=str,
                            help="R|Configuration file to change default telluric parameters.\n"
                            "Note that the parameters in this file will be overwritten if you set argument in your terminal. \n"
                            "The --tell_file option requires a .tell file with the following format:\n"
                            "\n"
                            "    [tellfit]\n"
                            "         objmodel = qso\n"
                            "         redshift = 7.6\n"
                            "         bal_wv_min_max = 10825,12060\n"
                            "OR\n"
                            "    [tellfit]\n"
                            "         objmodel = star\n"
                            "         star_type = A0\n"
                            "         star_mag = 8.\n"
                            "OR\n"
                            "    [tellfit]\n"
                            "         objmodel = poly\n"
                            "         polyorder = 3\n"
                            "         fit_wv_min_max = 9000.,9500.\n"
                            "\n"
                            )
        parser.add_argument("--debug", default=False, action="store_true",
                            help="show debug plots?")
        parser.add_argument("--plot", default=False, action="store_true",
                            help="Show the telluric corrected spectrum")
        parser.add_argument("--par_outfile", default='telluric.par',
                            help="Name of outut file to save the parameters used by the fit")
        return parser

    @staticmethod
    def main(args):
        """
        Executes telluric correction.
        """

        # Determine the spectrograph
        header = fits.getheader(args.spec1dfile)
        spectrograph = load_spectrograph(header['PYP_SPEC'])
        spectrograph_def_par = spectrograph.default_pypeit_par()

        # If the .tell file was passed in read it and overwrite default parameters
        if args.tell_file is not None:
            cfg_lines = read_tellfile(args.tell_file)
            par = pypeitpar.PypeItPar.from_cfg_lines(cfg_lines=spectrograph_def_par.to_config(),
                                                     merge_with=cfg_lines)
        else:
            par = spectrograph_def_par

        # If args was provided override defaults. Note this does undo .tell file
        if args.objmodel is not None:
            par['telluric']['objmodel'] = args.objmodel
        if args.pca_file is not None:
            par['telluric']['pca_file'] = args.pca_file
        if args.redshift is not None:
            par['telluric']['redshift'] = args.redshift
        if args.tell_grid is not None:
            par['telluric']['telgridfile'] = args.tell_grid

        if par['telluric']['telgridfile'] is None:
            if par['sensfunc']['IR']['telgridfile'] is not None:
                par['telluric']['telgridfile'] = par['sensfunc']['IR']['telgridfile']
            else:
                par['telluric']['telgridfile'] = resource_filename('pypeit',
                                 '/data/telluric/atm_grids/TelFit_MaunaKea_3100_26100_R20000.fits')
                msgs.warn(f"No telluric grid file given. Using {par['telluric']['telgridfile']}.")

        # Write the par to disk
        print("Writing the parameters to {}".format(args.par_outfile))
        par['telluric'].to_config('telluric.par', section_name='telluric', include_descr=False)

        # Parse the output filename
        outfile = (os.path.basename(args.spec1dfile)).replace('.fits','_tellcorr.fits')
        modelfile = (os.path.basename(args.spec1dfile)).replace('.fits','_tellmodel.fits')

        # Run the telluric fitting procedure.
        if par['telluric']['objmodel']=='qso':
            # run telluric.qso_telluric to get the final results
            TelQSO = telluric.qso_telluric(args.spec1dfile, par['telluric']['telgridfile'],
                                           par['telluric']['pca_file'],
                                           par['telluric']['redshift'], modelfile, outfile,
                                           npca=par['telluric']['npca'],
                                           pca_lower=par['telluric']['pca_lower'],
                                           pca_upper=par['telluric']['pca_upper'],
                                           bounds_norm=par['telluric']['bounds_norm'],
                                           tell_norm_thresh=par['telluric']['tell_norm_thresh'],
                                           only_orders=par['telluric']['only_orders'],
                                           bal_wv_min_max=par['telluric']['bal_wv_min_max'],
                                           maxiter=par['telluric']['maxiter'],
                                           debug_init=args.debug, disp=args.debug,
                                           debug=args.debug, show=args.plot)
        elif par['telluric']['objmodel']=='star':
            TelStar = telluric.star_telluric(args.spec1dfile, par['telluric']['telgridfile'],
                                             modelfile, outfile,
                                             star_type=par['telluric']['star_type'],
                                             star_mag=par['telluric']['star_mag'],
                                             star_ra=par['telluric']['star_ra'],
                                             star_dec=par['telluric']['star_dec'],
                                             func=par['telluric']['func'],
                                             model=par['telluric']['model'],
                                             polyorder=par['telluric']['polyorder'],
                                             only_orders=par['telluric']['only_orders'],
                                             mask_abs_lines=par['telluric']['mask_abs_lines'],
                                             delta_coeff_bounds=par['telluric']['delta_coeff_bounds'],
                                             minmax_coeff_bounds=par['telluric']['minmax_coeff_bounds'],
                                             maxiter=par['telluric']['maxiter'],
                                             debug_init=args.debug, disp=args.debug,
                                             debug=args.debug, show=args.plot)
        elif par['telluric']['objmodel']=='poly':
            TelPoly = telluric.poly_telluric(args.spec1dfile, par['telluric']['telgridfile'],
                                             modelfile, outfile,
                                             z_obj=par['telluric']['redshift'],
                                             func=par['telluric']['func'],
                                             model=par['telluric']['model'],
                                             polyorder=par['telluric']['polyorder'],
                                             fit_wv_min_max=par['telluric']['fit_wv_min_max'],
                                             mask_lyman_a=par['telluric']['mask_lyman_a'],
                                             delta_coeff_bounds=par['telluric']['delta_coeff_bounds'],
                                             minmax_coeff_bounds=par['telluric']['minmax_coeff_bounds'],
                                             only_orders=par['telluric']['only_orders'],
                                             maxiter=par['telluric']['maxiter'],
                                             debug_init=args.debug, disp=args.debug,
                                             debug=args.debug, show=args.plot)
        else:
            msgs.error("Object model is not supported yet. Please choose one of 'qso', 'star', 'poly'.")


