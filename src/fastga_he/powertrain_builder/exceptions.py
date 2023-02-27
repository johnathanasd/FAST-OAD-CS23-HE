# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from fastoad.exceptions import FastError


class FASTGAHEUnknownComponentID(FastError):
    """
    Class for managing errors that result from trying to add a component to the power train with
    an ID that is not recognized.
    """


class FASTGAHEUnknownOption(FastError):
    """
    Class for managing errors that result from trying to add a component to the power train with
    options that are not recognized or not all options needed.
    """


class FASTGAHEComponentsNotIdentified(FastError):
    """
    Class for managing errors that result from trying to run the _get_connection method before
    having identified the components in the power train with the _get_components method.
    """


class FASTGAHESingleSSPCAtEndOfLine(FastError):
    """
    Class for managing errors that result from connecting a dc line to an SSPC but only a single
    one. Because of the way equations were coded, if one end of a harness is connected to an
    SSPC, the other shall be as well to allow for a possible opening of the 2 SSPCs.
    """
