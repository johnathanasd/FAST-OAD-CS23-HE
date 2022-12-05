# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from fastoad.exceptions import FastError


class FASTGAHEUnknownComponentID(FastError):
    """
    Class for managing errors that result from trying to add a component to the power train with
    an ID that is not recognized.
    """
