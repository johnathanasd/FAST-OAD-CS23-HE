# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .sizing_tank_unusable_hydrogen import SizingHydrogenGasTankUnusableHydrogen
from .sizing_tank_total_hydrogen_mission import SizingHydrogenGasTankTotalHydrogenMission
from .sizing_tank_volume import SizingFuelTankVolume
from .sizing_tank_cg_x import SizingHydrogenGasTankCGX
from .sizing_tank_length import SizingFuelTankLength
from .sizing_tank_height import SizingFuelTankHeight
from .sizing_tank_width import SizingFuelTankWidth
from .sizing_tank_weight import SizingFuelTankWeight
from .sizing_tank_drag import SizingFuelTankDrag
from .sizing_tank_prep_for_loads import SizingFuelTankPreparationForLoads

from .cstr_hydrogen_gas_tank import ConstraintsHydrogenGasTank

from ..constants import POSSIBLE_POSITION


class SizingHydrogenGasTank(om.Group):
    """
    Class that regroups all of the sub components for the sizing of the hydrogen gas tank.
    """

    def initialize(self):
        self.options.declare(
            name="hydrogen_gas_tank_id",
            default=None,
            desc="Identifier of the hydrogen gas tank",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="in_the_fuselage",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the hydrogen gas tank, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="unusable_fuel",
            subsys=SizingFuelTankUnusableFuel(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="total_fuel",
            subsys=SizingFuelTankTotalFuelMission(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="constraints_tank",
            subsys=ConstraintsFuelTank(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_cg_x",
            subsys=SizingHydrogenGasTankCGX(
                hydrogen_gas_tank_id=hydrogen_gas_tank_id, position=position
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_volume",
            subsys=SizingFuelTankVolume(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tank_length",
            subsys=SizingFuelTankLength(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tank_height",
            subsys=SizingFuelTankHeight(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tank_width",
            subsys=SizingFuelTankWidth(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_weight",
            subsys=SizingFuelTankWeight(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )

        for low_speed_aero in [True, False]:
            system_name = "tank_drag_ls" if low_speed_aero else "tank_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingFuelTankDrag(
                    hydrogen_gas_tank_id=hydrogen_gas_tank_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )

        if position == "inside_the_wing":

            self.add_subsystem(
                name="preparation_for_loads",
                subsys=SizingFuelTankPreparationForLoads(
                    hydrogen_gas_tank_id=hydrogen_gas_tank_id,
                    position=position,
                ),
                promotes=["*"],
            )
