# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import openmdao.api as om

from ..constants import POSSIBLE_POSITION

STEP = 1e-7
TE_CHORD_MARGIN_RATIO = 0.05  # Ratio of the chord left between the flaps and the tank


class SizingHydrogenGasTankLength(om.ExplicitComponent):
    """
    Computation of the cylindrical part length of the tank, which does not include the cap from both end.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.spline = None

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

        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":inner_volume",
            units="m**3",
            val=np.nan,
            desc="Capacity of the tank in terms of volume",
        )

        self.add_input(
            name="data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:inner_diameter",
            units="m",
            val=np.nan,
            desc="Inner diameter of the hydrogen gas tanks",
        )

        self.add_input(
            name="data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:outer_diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the hydrogen gas tanks",
        )

        if position == "underbelly" or position == "in_the_fuselage":
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
            self.add_output(
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:diameter",
                units="m",
                val=1.0,
            )

        self.add_output(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:length",
            val=1.0,
            units="m",
            desc="Value of the cylindrical length of the tank in the x-direction, computed differently based "
            "on the location of the tank",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]
        position = self.options["position"]
        d = inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:inner_diameter"
        ]
        length = (
            inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":inner_volume"
            ]
            - np.pi * d ** 3 / 6
        ) / (np.pi * d ** 2 / 4)

        if (position == "underbelly" or position == "in_the_fuselage") and length > inputs[
            "data:geometry:cabin:length"
        ]:
            outputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:length"
            ] = inputs["data:geometry:cabin:length"] = inputs["data:geometry:cabin:length"]
            sizing_factor = np.sqrt(length / inputs["data:geometry:cabin:length"])
            outputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:diameter"
            ] = (
                sizing_factor
                * inputs[
                    "data:propulsion:he_power_train:hydrogen_gas_tank:"
                    + hydrogen_gas_tank_id
                    + ":dimension:outer_diameter"
                ]
            )

        elif (position == "underbelly" or position == "in_the_fuselage") and length < inputs[
            "data:geometry:cabin:length"
        ]:
            outputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:length"
            ] = length
            outputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:diameter"
            ] = inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:outer_diameter"
            ]

        else:
            outputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:length"
            ] = length

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]
        position = self.options["position"]
        length = (
            inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":inner_volume"
            ]
            - np.pi * d ** 3 / 6
        ) / (np.pi * d ** 2 / 4)
        d = inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:inner_diameter"
        ]
        if (position == "underbelly" or position == "in_the_fuselage") and length > inputs[
            "data:geometry:cabin:length"
        ]:
            sizing_factor = np.sqrt(length / inputs["data:geometry:cabin:length"])
            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:length",
                "data:geometry:cabin:length",
            ] = 1.0

            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:length",
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:outer_diameter",
            ] = sizing_factor

            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:length",
                "data:geometry:cabin:length",
            ] = (
                -0.5
                * sizing_factor
                / inputs["data:geometry:cabin:length"]
                * inputs[
                    "data:propulsion:he_power_train:hydrogen_gas_tank:"
                    + hydrogen_gas_tank_id
                    + ":dimension:outer_diameter"
                ]
            )
            # TODO: Calculate the partial
            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:length",
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:inner_diameter",
            ] = 0.0

            # TODO: Calculate the partial
            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:length",
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":inner_volume",
            ] = 0.0

        elif (position == "underbelly" or position == "in_the_fuselage") and length < inputs[
            "data:geometry:cabin:length"
        ]:
            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:length",
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":inner_volume",
            ] = 1 / (d ** 2 * np.pi / 4)

            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:length",
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:inner_diameter",
            ] = (
                -2 * d / 3
            )

            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:diameter",
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:outer_diameter",
            ] = 1.0

        else:
            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:length",
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":inner_volume",
            ] = 1 / (d ** 2 * np.pi / 4)

            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:length",
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:inner_diameter",
            ] = (
                -2 * d / 3
            )
