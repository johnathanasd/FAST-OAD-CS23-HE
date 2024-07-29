# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from utils.filter_residuals import filter_residuals


class SizingCryogenicHydrogenTankUnusableHydrogen(om.ExplicitComponent):
    """
    Computation of the amount of boil-off hydrogen in tank.
    """

    def initialize(self):

        self.options.declare(
            name="cryogenic_hydrogen_tank_id",
            default=None,
            desc="Identifier of the hydrogen gas tank",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        # To modify based on the minimum pressure for the output hydrogen mass flow
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "hydrogen_boil_off_t",
            units="kg",
            val=np.full(number_of_points, np.nan),
            desc="Amount of hydrogen from that tank which will be consumed during mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":unusable_fuel_mission",
            units="kg",
            val=3.,
            desc="Amount of trapped hydrogen in the tank",
        )

        self.declare_partials(of="*", wrt="*", val=np.ones(number_of_points))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # To modify based on the minimum pressure for the output hydrogen mass flow
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        outputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":unusable_fuel_mission"
        ] = np.sum(inputs["hydrogen_boil_off_t"])
