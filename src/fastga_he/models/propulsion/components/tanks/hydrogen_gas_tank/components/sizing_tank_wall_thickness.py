# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

# To modify
class SizingFuelTankWallThickness(om.ExplicitComponent):
    """
    Computation of the weight of the tank. The very simplistic approach we will use is to say
    that weight of tank is the weight of unused fuel and the weight of the tank itself.
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
            "data:propulsion:he_power_train:hydrogen_gas_tank:" + hydrogen_gas_tank_id + ":radius",
            units="m",
            val=np.nan,
            desc="Outer Radius of the hydrogen gas tank",
        )

        self.add_output(
            name="data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":mass",
            units="kg",
            val=1.0,
            desc="Weight of the hydrogen gas tanks",
        )

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

        outputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:" + hydrogen_gas_tank_id + ":mass"
        ] = inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":unusable_fuel_mission"
        ]
