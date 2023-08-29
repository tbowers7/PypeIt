# -*- coding: utf-8 -*-
"""
Configuration file loaders for specific PypeIt functionality

.. note::

    In order to support small-aperture commercial-off-the-shelf equipment, we
    introduce a configuration file into which the user adds all of the relevant
    information about their observatory and equipment.  This is used by the
    :mod:`~pypeit.spectrographs.smallap_cots` module to populate the
    spectrograph- and detector-specific metadata attributes.

----

Implementation Documentation
----------------------------

Included here are three primary tools:

    * YAML loader that reads in the configuration file and does basic
      functional testing to ensure the file was not corrupt.  This should be
      general to all possible PypeIt configuration files.
    * Parser for _this type_ of configuration file to ensure the user has
      included all of the necessary information.
    * Tool for how to install / keep the configuration file?  Not sure this
      will be needed, or just specify that all configuration files be placed
      in the ~/.pypeit directory.

.. include:: ../include/links.rst
"""
import pathlib

import astropy.units as u
import yaml

from pypeit import msgs
from pypeit import data

# Imperial units, what-what?
u.add_enabled_units(u.imperial)

__all__ = ["get_cots_config"]


# Top-Level Loader Functions =================================================#
def get_cots_config(filename: str) -> dict:
    """Top-level configuration file loader for COTS conffiles

    _extended_summary_

    Parameters
    ----------
    filename : :obj:`str`
        The filename (no path) of the configuration file to be loaded

    Returns
    -------
    :obj:`dict`
        The checked and parsed dictionary containing the configuration info
    """
    # if filename is None:
    #     msgs.error(f'For smallap_cots, you must also specify the location '
    #                f'of the associated configutation file.  {msgs.newline()}'
    #                 'Consult the documentation '
    #                 'on how to create this file.')

    # Define where the file location, and check existance and readability
    if filename is None:
        conf_fn = data.Paths.data / "template_cots.yaml"
    else:
        conf_fn = pathlib.Path.home().joinpath(".pypeit", filename).resolve()
    if not (conf_fn.exists() and conf_fn.is_file()):
        msgs.error(
            f"COTS configuration file {filename} is not found in the "
            f"pypeit cache ({pathlib.Path.home().joinpath('.pypeit').resolve()})"
        )
    # Load, parse, and return
    return cots_config_parser(yaml_config_loader(conf_fn))


# Internal Helper Functions ==================================================#
def yaml_config_loader(conf_fn: pathlib.Path, print_keys=False) -> dict:
    """Configuration file loader (YAML format)

    This is a base YAML configuration file loader that can be used with any of
    the specific configuration types above.  The file is checked for existance
    and readability, loaded, and checked that the result is a dictionary.

    Parameters
    ----------
    conf_fn : :obj:`~pathlib.Path`
        The path to the configuration file to be read in
    print_keys : :obj:`bool`, optional
        Print the dictionary keys as a debugging step? (Default: False)

    Returns
    -------
    :obj:`dict`
        The dictionary form of the YAML configuration file
    """
    # Check for existance and readability
    if not (conf_fn.exists() and conf_fn.is_file()):
        msgs.error(f"Configuration file {conf_fn} cannot be read.")

    # Open and read in the YAML file
    with open(conf_fn, "r", encoding="utf-8") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as err:
            msgs.error(f"Problem reading configuration file: {err}")

    # Error checking
    if not isinstance(config, dict):
        msgs.error(f"Configuration file {conf_fn.name} not read in as a dictionary")

    if print_keys:
        print(list(config.keys()))

    return config


def cots_config_parser(input_dict: dict) -> dict:
    """Parse the COTS configuration file

    To ensure the configuration file matches what is expected, this function:

        #. Checks that the YAML document has the correct main sections.
        #. Converts values with units into AstroPy Quantities

    Parameters
    ----------
    input_dict : :obj:`dict`
        The input dictionary read from the YAML configuration file

    Returns
    -------
    :obj:`dict`
        The parsed and vetted configuration dictionary
    """
    # The set of top-level keys must be restricted to:
    if (top_keys := sorted(input_dict.keys())) != [
        "camera",
        "spectrograph",
        "telescope",
    ]:
        msgs.error("Silly customer, you cannot hurt a twinkie!")

    # Convert any entries with units into Quantity objects
    for top_key in top_keys:
        # This config file is a dictionary of dictionaries of parameters
        for key, val in input_dict[top_key].items():
            # Values with units will appear as lists in the YAML
            if isinstance(val, list):
                try:
                    # Turn the list into a quantity
                    quantity = u.Quantity(f"{val[0]} {val[1]}")
                    # Replace the list with the quantity
                    input_dict[top_key][key] = quantity
                except TypeError:
                    # If the list doesn't convert to a Quantity, move along
                    pass

    return input_dict
