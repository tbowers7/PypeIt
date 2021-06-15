"""
This script displays the wavelength calibration diagnostics.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

from pypeit.scripts import scriptbase


class ChkWaveCalib(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        parser = super().get_parser(description='Print QA on Wavelength Calib to the screen',
                                    width=width)

        parser.add_argument('master_file', type=str,
                            help='PypeIt MasterWaveCalib file [e.g. MasterWaveCalib_A_1_01.fits]')
        #parser.add_argument('--try_old', default=False, action='store_true',
        #                    help='Attempt to load old datamodel versions.  A crash may ensue..')
        return parser

    @staticmethod
    def main(args):

        from pypeit import wavecalib

        # Load
        waveCalib = wavecalib.WaveCalib.from_file(args.master_file)
                                                  #, chk_version=(not args.try_old))
        # Do it
        waveCalib.print_diagnostics()


