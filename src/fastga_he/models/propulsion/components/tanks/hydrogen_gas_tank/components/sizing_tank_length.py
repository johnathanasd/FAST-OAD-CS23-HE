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

    def setup(self):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

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
        d = inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:inner_diameter"
        ]
        outputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:length"
        ] = (
            inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":inner_volume"
            ]
            - np.pi * d ** 3 / 6
        ) / (
            np.pi * d ** 2 / 4
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]
        d = inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:inner_diameter"
        ]
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
        ] = (-2 * d / 3)
