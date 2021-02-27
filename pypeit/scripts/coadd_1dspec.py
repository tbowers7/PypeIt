#!/usr/bin/env python
"""
Script for coadding PypeIt 1d spectra
"""
import os

from configobj import ConfigObj
import numpy as np
from pypeit import par, msgs
import argparse
from pypeit import coadd1d
from pypeit.core import coadd
from pypeit.par import pypeitpar
from pypeit.spectrographs.util import load_spectrograph
from astropy.io import fits

from IPython import embed

# TODO JFH: Put this SmartFormatter in a common place, like pypeit.pypmsgs

# A trick from stackoverflow to allow multi-line output in the help:
#https://stackoverflow.com/questions/3853722/python-argparse-how-to-insert-newline-in-the-help-text
class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            # TODO: We shouldn't be ignoring the terminal width here,
            # but doing anything fancier than splitlines() gets complicated quickly. I
            # think we should be careful with using this formatter to
            # make lines no longer than about 60 characters.
            # import textwrap
            # lines = np.concatenate([textwrap.wrap(t, width) if len(t) > 0 else [' ']
            #                            for t in text[2:].split('\n')]).tolist()
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


# TODO This is basically the exact same code as read_fluxfile in the fluxing script. Consolidate them? Make this
# a standard method in parse or io.
def read_coaddfile(ifile):
    """
    Read a PypeIt .coadd1d file, akin to a standard PypeIt file

    The top is a config block that sets ParSet parameters
      The spectrograph is required

    Args:
        ifile (str):
          Name of the flux file

    Returns:
        cfg_lines (list):
          Config lines to modify ParSet values
        spec1dfiles (list):
          Contains spec1dfiles to be coadded
        objids (list):
          Object ids aligned with each of the spec1dfiles


    """
    # Read in the pypeit reduction file
    msgs.info('Loading the coadd1d file')
    lines = par.util._read_pypeit_file_lines(ifile)
    is_config = np.ones(len(lines), dtype=bool)


    # Parse the fluxing block
    spec1dfiles = []
    objids_in = []
    s, e = par.util._find_pypeit_block(lines, 'coadd1d')
    if s >= 0 and e < 0:
        msgs.error("Missing 'coadd1d end' in {0}".format(ifile))
    elif (s < 0) or (s==e):
        msgs.error("Missing coadd1d read or [coadd1d] block in in {0}. Check the input format for the .coadd1d file".format(ifile))
    else:
        for ctr, line in enumerate(lines[s:e]):
            prs = line.split(' ')
            spec1dfiles.append(prs[0])
            if ctr == 0 and len(prs) != 2:
                msgs.error('Invalid format for .coadd1d file.' + msgs.newline() +
                           'You must have specify a spec1dfile and objid on the first line of the coadd1d block')
            if len(prs) > 1:
                objids_in.append(prs[1])
        is_config[s-1:e+1] = False

    # Chck the sizes of the inputs
    nspec = len(spec1dfiles)
    if len(objids_in) == 1:
        objids = nspec*objids_in
    elif len(objids_in) == nspec:
        objids = objids_in
    else:
        msgs.error('Invalid format for .flux file.' + msgs.newline() +
                   'You must specify a single objid on the first line of the coadd1d block,' + msgs.newline() +
                   'or specify am objid for every spec1dfile in the coadd1d block.' + msgs.newline() +
                   'Run pypeit_coadd_1dspec --help for information on the format')
    # Construct config to get spectrograph
    cfg_lines = list(lines[is_config])

    # Return
    return cfg_lines, spec1dfiles, objids


def coadd1d_filelist(files, outroot, det, debug=False, show=False):
    """

    Args:
        files:
        outroot:
        det:
        debug:
        show:

    Returns:
        outfiles (list): list of files written to

    """
    # Build sync_dict
    sync_dict = None
    for ifile in files[1:]:
        sync_dict = coadd.sync_pair(files[0], ifile, det, sync_dict=sync_dict)
    #
    header = fits.getheader(files[0])
    spectrograph = load_spectrograph(header['PYP_SPEC'])
    par = spectrograph.default_pypeit_par()

    par['coadd1d']['flux_value'] = False

    sensfile = None
    outfiles = []
    # Loop on entries
    for key in sync_dict:

        coaddfile = outroot+'-SPAT{:04d}-DET{:02d}'.format(key, det)+'.fits'

        coAdd1d = coadd1d.CoAdd1D.get_instance(sync_dict[key]['files'],
                                               sync_dict[key]['names'],
                                               spectrograph=spectrograph, par=par['coadd1d'],
                                               sensfile=sensfile, debug=debug, show=show)
        # Run
        coAdd1d.run()
        # Save to file
        coAdd1d.save(coaddfile)
        outfiles.append(coaddfile)

    return outfiles


def parse_args(options=None, return_parser=False):

    parser = argparse.ArgumentParser(description='Parse', formatter_class=SmartFormatter)
    parser.add_argument("coadd1d_file", type=str,
                        help="R|File to guide coadding process. This file must have the following format: \n"
                             "\n"
                             "[coadd1d]\n"
                             "   coaddfile='output_filename.fits'\n"
                             "   sensfuncfile = 'sensfunc.fits' # Required only for Echelle\n"
                             "\n"
                             "   coadd1d read\n"
                             "     spec1dfile1 objid1\n"
                             "     spec1dfile2 objid2\n"
                             "     spec1dfile3 objid3\n"
                             "        ...    \n"
                             "   coadd1d end\n"
                             "\n"
                             "         OR the coadd1d read/end block can look like \n"
                             "\n"
                             "  coadd1d read\n"
                             "     spec1dfile1 objid \n"
                             "     spec1dfile2 \n"
                             "     spec1dfile3 \n"
                             "     ...    \n"
                             "  coadd1d end\n"
                             "\n"
                             "That is the coadd1d block must either be a two column list of spec1dfiles and objids,\n"
                             "or you can specify a single objid for all spec1dfiles on the first line\n"
                             "\n"
                             "Where: \n"
                             "\n"
                             "   spec1dfile -- full path to a PypeIt spec1dfile\n"
                             "   objid      -- is the object identifier. To determine the objids inspect the \n"
                             "                 spec1d_*.txt files or run pypeit_show_1dspec spec1dfile --list\n"
                             "\n")
    parser.add_argument("--debug", default=False, action="store_true", help="show debug plots?")
    parser.add_argument("--show", default=False, action="store_true", help="show QA during coadding process")
    parser.add_argument("--par_outfile", default='coadd1d.par', help="Output to save the parameters")
    parser.add_argument("--test_spec_path", type=str, help="Path for testing")
#    parser.add_argument("--plot", default=False, action="store_true", help="Show the sensitivity function?")

    if return_parser:
        return parser

    return parser.parse_args() if options is None else parser.parse_args(options)


def main(args):
    """ Runs the 1d coadding steps
    """
    # Load the file
    config_lines, spec1dfiles, objids = read_coaddfile(args.coadd1d_file)
    # Append path for testing
    if args.test_spec_path is not None:
        spec1dfiles = [os.path.join(args.test_spec_path, ifile) for ifile in spec1dfiles]
    # Read in spectrograph from spec1dfile header
    header = fits.getheader(spec1dfiles[0])

    # NOTE: This was some test code for Travis. Keep it around for now
    # in case we need to do this again. (KBW)
#    try:
#        header = fits.getheader(spec1dfiles[0])
#    except Exception as e:
#        raise Exception('{0}\n {1}\n {2}\n'.format(spec1dfiles[0], os.getcwd(),
#                        os.getenv('TRAVIS_BUILD_DIR', default='None'))) from e

    spectrograph = load_spectrograph(header['PYP_SPEC'])

    # Parameters
    spectrograph_def_par = spectrograph.default_pypeit_par()
    par = pypeitpar.PypeItPar.from_cfg_lines(cfg_lines=spectrograph_def_par.to_config(), merge_with=config_lines)
    # Write the par to disk
    print("Writing the parameters to {}".format(args.par_outfile))
    par.to_config(args.par_outfile)
    sensfile = par['coadd1d']['sensfuncfile']
    coaddfile = par['coadd1d']['coaddfile']

    # Testing?
    if args.test_spec_path is not None:
        if sensfile is not None:
            sensfile = os.path.join(args.test_spec_path, sensfile)
        coaddfile = os.path.join(args.test_spec_path, coaddfile)

    if spectrograph.pypeline == 'Echelle' and sensfile is None:
        msgs.error('You must specify set the sensfuncfile in the .coadd1d file for Echelle coadds')

    # Instantiate
    coAdd1d = coadd1d.CoAdd1D.get_instance(spec1dfiles, objids,
                                           spectrograph=spectrograph, par=par['coadd1d'],
                                           sensfile=sensfile,
                                           debug=args.debug, show=args.show)
    # Run
    coAdd1d.run()
    # Save to file
    coAdd1d.save(coaddfile)
    msgs.info('Coadding complete')


def entry_point():
    main(parse_args())


if __name__ == '__main__':
    entry_point()
