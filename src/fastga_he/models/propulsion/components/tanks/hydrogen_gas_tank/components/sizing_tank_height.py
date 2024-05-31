# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import openmdao.api as om

from ..constants import POSSIBLE_POSITION

FUSELAGE_HEIGHT_RATIO = 0.1  # Ratio between the fuselage height and the height


class SizingHydrogenGasTankHeight(om.ExplicitComponent):
    """
    Computation of the reference height for the computation of the tank width. If the tank is in
    a pod, it will depend on volume and a fineness ratio. If it is in the wing, it will depend on
    the wing chord and thickness ratio. If it is in the fuselage it depend on the fuselage max
    height.
    """

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

        self.add_output(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:height",
            val=0.1,
            units="m",
            desc="Value of the length of the tank in the z-direction, computed differently based "
            "on the location of the tank",
        )

        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:fuselage_height_ratio",
            val=np.nan,
            desc="Ratio between the fuselage height and the tank height",
        )

        self.declare_partials(
            of="*", wrt="data:geometry:fuselage:maximum_height", val=FUSELAGE_HEIGHT_RATIO
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]
        position = self.options["position"]

        fuselage_height_ratio = inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:fuselage_height_ratio"
        ]

        outputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:height"
        ] = (fuselage_height_ratio * inputs["data:geometry:fuselage:maximum_height"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]
        partials[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:height",
            "data:geometry:fuselage:maximum_height",
        ] = inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:fuselage_height_ratio"
        ]
