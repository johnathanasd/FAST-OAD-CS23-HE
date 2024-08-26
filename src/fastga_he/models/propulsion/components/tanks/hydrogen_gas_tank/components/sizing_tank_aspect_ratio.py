# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingHydrogenGasTankAspectRatio(om.ExplicitComponent):
    """
    Computation of the ratio between the overall length and the outer diameter
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
            desc="Outer diameter of the hydrogen tank",
        )

        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:overall_length",
            val=np.nan,
            units="m",
            desc="Overall length of the tank" "on the location of the tank",
        )

        self.add_output(
            name="data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:aspect_ratio",
            val=10.0,
            desc="Tank aspect between the overall length and outer diameter, the higher the more cylindrical",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

        unclipped_ar = (
            inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:overall_length"
            ]
            / inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:outer_diameter"
            ]
        )

        outputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:aspect_ratio"
        ] = np.clip(
            unclipped_ar,
            1.0,
            np.inf,
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

        unclipped_ar = (
            inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:overall_length"
            ]
            / inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:outer_diameter"
            ]
        )

        if unclipped_ar >= 1.:

            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:aspect_ratio",
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:overall_length",
            ] = (
                1
                / inputs[
                    "data:propulsion:he_power_train:hydrogen_gas_tank:"
                    + hydrogen_gas_tank_id
                    + ":dimension:outer_diameter"
                ]
            )

            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:aspect_ratio",
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:outer_diameter",
            ] = -(
                inputs[
                    "data:propulsion:he_power_train:hydrogen_gas_tank:"
                    + hydrogen_gas_tank_id
                    + ":dimension:overall_length"
                ]
                / inputs[
                    "data:propulsion:he_power_train:hydrogen_gas_tank:"
                    + hydrogen_gas_tank_id
                    + ":dimension:outer_diameter"
                ]
                ** 2
            )

        else:
            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:aspect_ratio",
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:overall_length",
            ] = 0.0

            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:aspect_ratio",
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:outer_diameter",
            ] = 0.0
