# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.perf_fuel_mission_consumed import PerformancesLiquidHydrogenConsumedMission
from ..components.perf_fuel_remaining import PerformancesLiquidHydrogenRemainingMission
from ..components.perf_fuel_boil_off import PerformancesHydrogenBoilOffMission


class PerformancesCryogenicHydrogenTank(om.Group):
    """
    Regrouping all the components for the performances of the tank. Note that to limit the work
    to be done for the implementation of fuel tanks, fuel tanks don't output the fuel consumed
    used to iterate on the mass during the mission, but it uses it. Just like for the CG where we
    will output the "varying" part of the CG straight from the mission; we could do the same for
    mass (which may or may not improve the computation time).
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="cryogenic_hydrogen_tank_id",
            default=None,
            desc="Identifier of the cryogenic hydrogen tank",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        self.add_subsystem(
            "liquid_hydrogen_consumed_mission",
            PerformancesLiquidHydrogenConsumedMission(
                number_of_points=number_of_points,
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "liquid_hydrogen_remaining_mission",
            PerformancesLiquidHydrogenRemainingMission(
                number_of_points=number_of_points,
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "hydrogen_boil_off_mission",
            PerformancesHydrogenBoilOffMission(
                number_of_points=number_of_points,
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
            ),
            promotes=["*"],
        )
