# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

# To modify
class SizingHydrogenGasTankWallThickness(om.ExplicitComponent):
    """
    Computation of the wall thickness of the tank. Using tank pressure and yield stress of material
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
            name="data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:inner_diameter",
            units="m",
            val=np.nan,
            desc="Inner diameter of the hydrogen gas tanks",
        )

        self.add_input(
            "material_yield_strength",
            val=np.nan,
            units="Pa",
            desc="Hydrogen gas tank material yield stress",
        )

        self.add_input(
            "Safety_factor",
            val=1.0,
            desc="Hydrogen gas tank design safety factor",
        )

        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":tank_pressure",
            val=np.nan,
            units="Pa",
            desc="Hydrogen gas tank static pressure",
        )

        self.add_output(
            name="data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:wall_thickness",
            units="m",
            val=0.01,
            desc="Inner diameter of the hydrogen gas tanks",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

        outputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:wall_thickness"
        ] = (
            0.25
            * inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":tank_pressure"
            ]
            * inputs["Safety_factor"]
            * inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:inner_diameter"
            ]
            / inputs["material_yield_strength"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

        partials[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:wall_thickness",
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":tank_pressure",
        ] = (
            0.25
            * inputs["Safety_factor"]
            * inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:inner_diameter"
            ]
            / inputs["material_yield_strength"]
        )
        partials[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:wall_thickness",
            "safety_factor",
        ] = (
            0.25
            * inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":tank_pressure"
            ]
            * inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:inner_diameter"
            ]
            / inputs["material_yield_strength"]
        )
        partials[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:wall_thickness",
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:inner_diameter",
        ] = (
            0.25
            * inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":tank_pressure"
            ]
            * inputs["Safety_factor"]
            / inputs["material_yield_strength"]
        )

        partials[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:wall_thickness",
            "material_yield_strength",
        ] = (
            -0.25
            * inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":tank_pressure"
            ]
            * inputs["Safety_factor"]
            * inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:inner_diameter"
            ]
            / inputs["material_yield_strength"] ** 2
        )
