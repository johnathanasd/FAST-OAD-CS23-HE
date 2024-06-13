# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
from ..constants import POSSIBLE_POSITION


class SizingHydrogenGasTankOverallLengthFuselageCheck(om.ExplicitComponent):
    """
    Computation of the cylindrical part length of the tank, which does not include the cap from both end.
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

        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:overall_length",
            val=np.nan,
            units="m",
            desc="Value of the length of the tank in the x-direction, computed differently based "
            "on the location of the tank",
        )

        if position == "in_the_fuselage" or position == "underbelly":
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

        if position == "in_the_back":
            self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output(
            "constraints:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:overall_length",
            val=np.nan,
            units="m",
            desc="Constraints on the tank length w.r.t the cabin/rear_fuselage length,  respected if <0",
        )

        if position != "wing_pod":
            self.declare_partials(
                of="*",
                wrt="data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:overall_length",
                val=1.0,
            )
        else:
            self.declare_partials(
                of="*",
                wrt="data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:overall_length",
                val=0.0,
            )

        if position == "in_the_fuselage" or position == "underbelly":
            self.declare_partials(
                of="*",
                wrt="data:geometry:cabin:length",
                val=-1.0,
            )

        elif position == "in_the_back":
            self.declare_partials(
                of="*",
                wrt="data:geometry:fuselage:rear_length",
                val=-0.5,
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]
        position = self.options["position"]
        if position == "in_the_fuselage" or position == "underbelly":
            outputs[
                "constraints:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:overall_length"
            ] = (
                inputs[
                    "data:propulsion:he_power_train:hydrogen_gas_tank:"
                    + hydrogen_gas_tank_id
                    + ":dimension:overall_length"
                ]
                - inputs["data:geometry:cabin:length"]
            )

        elif position == "in_the_back":
            outputs[
                "constraints:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:overall_length"
            ] = (
                inputs[
                    "data:propulsion:he_power_train:hydrogen_gas_tank:"
                    + hydrogen_gas_tank_id
                    + ":dimension:overall_length"
                ]
                - 0.5 * inputs["data:geometry:fuselage:rear_length"]
            )
        else:
            outputs[
                "constraints:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:overall_length"
            ] = 0.0
