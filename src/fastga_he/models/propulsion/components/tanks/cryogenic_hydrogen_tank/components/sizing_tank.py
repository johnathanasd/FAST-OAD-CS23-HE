# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .sizing_tank_unusable_hydrogen import SizingCryogenicHydrogenTankUnusableHydrogen
from .sizing_tank_total_hydrogen_mission import SizingCrogenicHydrogenTankTotalHydrogenMission
from .sizing_tank_wall_thickness import SizingHydrogenGasTankWallThickness
from .sizing_tank_cg_x import SizingHydrogenGasTankCGX
from .sizing_tank_cg_y import SizingHydrogenGasTankCGY
from .sizing_tank_length import SizingHydrogenGasTankLength
from .sizing_tank_inner_volume import SizingHydrogenGasTankInnerVolume
from .sizing_tank_inner_diameter import SizingHydrogenGasTankInnerDiameter
from .sizing_tank_weight import SizingHydrogenGasTankWeight
from .sizing_specific_weight import SizingHydrogenGasTankSpecificWeight
from .sizing_tank_drag import SizingHydrogenGasTankDrag
from .sizing_tank_outer_diameter import SizingHydrogenGasTankOuterDiameter
from .sizing_tank_diameter_update import SizingHydrogenGasTankDiameterUpdate
from .sizing_tank_overall_length import SizingHydrogenGasTankOverallLength
from .sizing_tank_overall_length_fuselage_check import (
    SizingHydrogenGasTankOverallLengthFuselageCheck,
)

from .cstr_cryogenic_hydrogen_tank import ConstraintsCryogenicHydrogenTank

from ..constants import POSSIBLE_POSITION


class SizingHydrogenGasTank(om.Group):
    """
    Class that regroups all the subcomponents for the sizing of the hydrogen gas tank.
    """

    def initialize(self):
        self.options.declare(
            name="cryogenic_hydrogen_tank_id",
            default=None,
            desc="Identifier of the cryogenic hydrogen tank",
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

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        position = self.options["position"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            name="tank_outer_diameter",
            subsys=SizingHydrogenGasTankOuterDiameter(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id, position=position
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_diameter_update",
            subsys=SizingHydrogenGasTankDiameterUpdate(cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="unusable_hydrogen_gas",
            subsys=SizingCryogenicHydrogenTankUnusableHydrogen(number_of_points=number_of_points , cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="total_hydrogen_gas",
            subsys=SizingCrogenicHydrogenTankTotalHydrogenMission(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_constraints",
            subsys=ConstraintsCryogenicHydrogenTank(cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_inner_diameter",
            subsys=SizingHydrogenGasTankInnerDiameter(cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_wall_thickness",
            subsys=SizingHydrogenGasTankWallThickness(cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_inner_volume",
            subsys=SizingHydrogenGasTankInnerVolume(cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_length",
            subsys=SizingHydrogenGasTankLength(cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_overall_length",
            subsys=SizingHydrogenGasTankOverallLength(cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_cg_x",
            subsys=SizingHydrogenGasTankCGX(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id, position=position
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_cg_y",
            subsys=SizingHydrogenGasTankCGY(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id, position=position
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_overall_length_length_fuselage_check",
            subsys=SizingHydrogenGasTankOverallLengthFuselageCheck(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
                position=position,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_weight",
            subsys=SizingHydrogenGasTankWeight(cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_specific_weight",
            subsys=SizingHydrogenGasTankSpecificWeight(cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id),
            promotes=["*"],
        )

        for low_speed_aero in [True, False]:
            system_name = "tank_drag_ls" if low_speed_aero else "tank_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingHydrogenGasTankDrag(
                    cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
