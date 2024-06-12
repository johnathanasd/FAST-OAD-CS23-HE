# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION
import logging

_LOGGER = logging.getLogger(__name__)


class SizingHydrogenGasTankOuterDiameter(om.ExplicitComponent):
    """
    Diameter check fot the hydrogen gas tank.
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

        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:diameter",
            units="m",
            val=np.nan,
            desc="Initial Outer diameter of the hydrogen gas tank input",
        )

        self.add_output(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:outer_diameter",
            units="m",
            val=5.0,
            desc="Outer diameter of the hydrogen gas tank",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]
        position = self.options["position"]

        if (position == "underbelly" or position == "in_the_fuselage" or position == "in_the_back") and inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:diameter"
        ] > inputs["data:geometry:fuselage:maximum_height"]:
            outputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:outer_diameter"
            ] = (0.9 * inputs["data:geometry:fuselage:maximum_height"])
            _LOGGER.warning(
                msg="Tank dimension greater than fuselage!! Tank diameter adjust to proper size"
            )
        else:
            outputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:outer_diameter"
            ] = inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:diameter"
            ]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]
        position = self.options["position"]

        if (position == "underbelly" or position == "in_the_fuselage" or position == "in_the_back") and inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:diameter"
        ] > inputs["data:geometry:fuselage:maximum_height"]:
            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:outer_diameter",
                "data:geometry:fuselage:maximum_height",
            ] = 0.9
        else:
            partials[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:outer_diameter",
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:diameter",
            ] = 1.0
