# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

# To modify
class SizingCryogenicHydrogenTankInnerDiameter(om.ExplicitComponent):
    """
    Computation of the inner diameter of the tank. Using the relation of the tank pressure and the yield strength of
    the wall material
    """

    def initialize(self):

        self.options.declare(
            name="cryogenic_hydrogen_tank_id",
            default=None,
            desc="Identifier of the cryogenic hydrogen tank",
            allow_none=False,
        )

    def setup(self):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:outer_diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the cryogenic hydrogen tank",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:insulation_thickness",
            units="m",
            val=np.nan,
            desc="Insulation layer thickness of the cryogenic hydrogen tank",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":tank_pressure",
            val=np.nan,
            units="Pa",
            desc="Cryogenic hydrogen tank static pressure",
        )

        self.add_input(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":Safety_factor",
            val=1.0,
            desc="Cryogenic hydrogen tank design safety factor",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":material:yield_strength",
            val=np.nan,
            units="Pa",
            desc="Cryogenic hydrogen tank material yield stress",
        )

        self.add_output(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:inner_diameter",
            units="m",
            val=1.0,
            desc="Inner diameter of the cryogenic hydrogen tank",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        outputs[input_prefix + ":dimension:inner_diameter"] = (
            inputs[input_prefix + ":dimension:outer_diameter"]
            - 2 * inputs[input_prefix + ":dimension:insulation_thickness"]
        ) / (
            1
            + 0.5
            * inputs[input_prefix + ":tank_pressure"]
            * inputs[input_prefix + ":Safety_factor"]
            / inputs[input_prefix + ":material:yield_strength"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        tank_pressure = inputs[input_prefix + ":tank_pressure"]

        sf = inputs[input_prefix + ":Safety_factor"]

        sigma = inputs[input_prefix + ":material:yield_strength"]

        d_outer = inputs[input_prefix + ":dimension:outer_diameter"]

        t_insulation = inputs[input_prefix + ":dimension:insulation_thickness"]

        partials[
            input_prefix + ":dimension:inner_diameter",
            input_prefix + ":dimension:outer_diameter",
        ] = 1 / (1 + 0.5 * tank_pressure * sf / sigma)

        partials[
            input_prefix + ":dimension:inner_diameter",
            input_prefix + ":dimension:insulation_thickness",
        ] = -2 / (1 + 0.5 * tank_pressure * sf / sigma)

        partials[input_prefix + ":dimension:inner_diameter", input_prefix + ":tank_pressure",] = (
            -2 * (d_outer - 2 * t_insulation) * sf * sigma / (sf * tank_pressure + 2 * sigma) ** 2
        )

        partials[input_prefix + ":dimension:inner_diameter", input_prefix + ":Safety_factor",] = (
            -2
            * (d_outer - 2 * t_insulation)
            * tank_pressure
            * sigma
            / (tank_pressure * sf + 2 * sigma) ** 2
        )

        partials[
            input_prefix + ":dimension:inner_diameter",
            input_prefix + ":material:yield_strength",
        ] = (
            2*tank_pressure*sf*(d_outer-2*t_insulation)/(2*sigma+tank_pressure*sf)**2
        )
