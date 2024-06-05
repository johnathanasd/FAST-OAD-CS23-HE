# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_PEMFC_NL_POWER


class ConstraintsPEMFC(om.Group):
    """
    Class that gather the different constraints for the pemfc, be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the pemfc stack",
            allow_none=False,
        )

    def setup(self):

        option_pemfc_stack_id = {"pemfc_stack_id": self.options["pemfc_stack_id"]}

        self.add_subsystem(
            name="constraints_pemfc_nominal_power",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_PEMFC_NL_POWER, options=option_pemfc_stack_id
            ),
            promotes=["*"],
        )
