# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingHydrogenGasTankTotalHydrogenMission(om.ExplicitComponent):
    """
    Computation of the amount of the total amount of hydrogen loaded for the mission. Is the sum of
    the consumed hydrogen and unusable hydrogen.
    """

    def initialize(self):

        self.options.declare(
            name="hydrogen_gas_tank_id",
            default=None,
            desc="Identifier of the hydrogen gas tank",
            allow_none=False,
        )

    def setup(self):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":hydrogen_consumed_mission",
            units="kg",
            val=np.nan,
            desc="Amount of hydrogen from that tank which will be consumed during mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":unusable_hydrogen_mission",
            units="kg",
            val=np.nan,
            desc="Amount of trapped hydrogen in the tank",
        )

        self.add_output(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":hydrogen_total_mission",
            units="kg",
            val=50.5,
            desc="Total amount of hydrogen loaded in the tank for the mission",
        )

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

        outputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":hydrogen_total_mission"
        ] = (
            inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":hydrogen_consumed_mission"
            ]
            + inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":unusable_hydrogen_mission"
            ]
        )
