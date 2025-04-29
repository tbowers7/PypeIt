"""
This script displays the flat images in an RC Ginga window.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

from pypeit.scripts import scriptbase


class ChkFlats(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        parser = super().get_parser(description='Display flat images in Ginga viewer',
                                    width=width)
        parser.add_argument('file', type=str,
                            help='PypeIt Flat file [e.g. Flat_A_1_DET01.fits]')
        parser.add_argument('--slits_file', type=str, default=None,
                            help='PypeIt Slits file [e.g. Slits_A_1_01.fits]. If this file does '
                                 'not exist or is not provided, PypeIt will attempt to read the '
                                 'default file name (in the Calibrations directory).')
        parser.add_argument("--type", type=str, default='all',
                            help="Which flats to display. Must be one of: pixel, illum, all")
        parser.add_argument('--try_old', default=False, action='store_true',
                            help='Attempt to load old datamodel versions.  A crash may ensue..')
        return parser

    @staticmethod
    def main(args):

        from pypeit import flatfield
        from pypeit import slittrace

        chk_version = not args.try_old

        # Load
        flatImages = flatfield.FlatImages.from_file(args.file, chk_version=chk_version)
        slits = slittrace.SlitTraceSet.from_file(args.slits_file, chk_version=chk_version) if args.slits_file is not None else None
        # Show
        flatImages.show(args.type, slits=slits, chk_version=chk_version)
