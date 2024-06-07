# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

# To modify
class SizingHydrogenGasTankInnerDiameter(om.ExplicitComponent):
    """
    Computation of the inner diameter of the tank. Using the relation of the tank pressure and the yield strength of
    the wall material
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
            + ":dimension:outer_diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the hydrogen gas tank",
        )

        self.add_input(
            "tank_pressure",
            val=np.nan,
            units="Pa",
            desc="Hydrogen gas tank static pressure",
        )

        self.add_input(
            "Safety_factor",
            val=1.0,
            desc="Hydrogen gas tank design safety factor",
        )

        self.add_input(
            "material_yield_strength",
            val=np.nan,
            units="Pa",
            desc="Hydrogen gas tank material yield stress",
        )

        self.add_output(
            name="data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:inner_diameter",
            units="m",
            val=1.0,
            desc="Inner diameter of the hydrogen gas tanks",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

        outputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:inner_diameter"
        ] = inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:outer_diameter"
        ] / (
            1
            + 0.5
            * inputs["tank_pressure"]
            * inputs["safety_factor"]
            / inputs["material_yield_strength"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

        pt = inputs["tank_pressure"]
        sf = inputs["safety_factor"]
        sigma = inputs["material_yield_strength"]
        d_outer = inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:outer_diameter"
        ]
        partials[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:inner_diameter",
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:outer_diameter",
        ] = 1 // (1 + 0.5 * pt * sf / sigma)
        partials[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:inner_diameter",
            "tank_pressure",
        ] = (
            -2 * d_outer * sf * sigma / (sf * pt + 2 * sigma) ** 2
        )
        partials[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:inner_diameter",
            "safety_factor",
        ] = (
            -2 * d_outer * pt * sigma / (pt * sf + 2 * sigma) ** 2
        )
        partials[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:inner_diameter",
            "material_yield_strength",
        ] = (
            2 * d_outer * sf * pt / (2 * sigma + sf * pt) ** 2
        )
