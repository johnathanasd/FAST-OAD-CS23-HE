# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_BATTERY_SOC


class ConstraintsPEMFC(om.Group):
    """
    Class that gather the different constraints for the battery, be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):

        option_battery_pack_id = {"battery_pack_id": self.options["battery_pack_id"]}

        self.add_subsystem(
            name="constraints_soc_battery",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_BATTERY_SOC, options=option_battery_pack_id
            ),
            promotes=["*"],
        )
