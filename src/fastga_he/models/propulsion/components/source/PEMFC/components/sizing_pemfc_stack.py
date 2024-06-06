# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om


from .sizing_pemfc_weight import SizingPEMFCWeight
from .sizing_number_stacks import SizingBatteryNumberLayers
from .sizing_pemfc_volume import SizingPEMFCVolume
from .sizing_pemfc_dimensions import SizingBatteryDimensions
from .sizing_pemfc_cg_x import SizingBatteryCGX
from .sizing_pemfc_cg_y import SizingBatteryCGY
from .sizing_pemfc_drag import SizingBatteryDrag
from .sizing_pemfc_prep_for_loads import SizingBatteryPreparationForLoads


from .cstr_pemfc_stack import ConstraintsPEMFC

from ..constants import POSSIBLE_POSITION


class SizingBatteryPack(om.Group):
    """Class that regroups all of the sub components for the sizing of the battery pack."""

    def initialize(self):

        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the battery, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        battery_pack_id = self.options["battery_pack_id"]
        position = self.options["position"]

        # It was decided to add the constraints computation at the beginning of the sizing to
        # ensure that both are ran along and to avoid having an additional id to add in the
        # configuration file.
        self.add_subsystem(
            name="constraints_battery",
            subsys=ConstraintsBattery(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="number_of_layers",
            subsys=SizingBatteryNumberLayers(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="pemfc_weight",
            subsys=SizingPEMFCWeight(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="pemfc_volume",
            subsys=SizingPEMFCVolume(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="battery_dimensions",
            subsys=SizingBatteryDimensions(battery_pack_id=battery_pack_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="battery_CG_x",
            subsys=SizingBatteryCGX(battery_pack_id=battery_pack_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="battery_CG_y",
            subsys=SizingBatteryCGY(battery_pack_id=battery_pack_id, position=position),
            promotes=["*"],
        )
        for low_speed_aero in [True, False]:
            system_name = "battery_drag_ls" if low_speed_aero else "battery_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingBatteryDrag(
                    battery_pack_id=battery_pack_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )

        if position == "inside_the_wing":

            self.add_subsystem(
                name="preparation_for_loads",
                subsys=SizingBatteryPreparationForLoads(
                    battery_pack_id=battery_pack_id,
                    position=position,
                ),
                promotes=["*"],
            )
