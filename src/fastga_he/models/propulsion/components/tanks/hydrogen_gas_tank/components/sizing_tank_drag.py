# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingHydrogenTankDrag(om.ExplicitComponent):
    """
    Class that computes the contribution to profile drag of the fuel tanks according to the
    position given in the options. For now this will be 0.0 regardless of the option except when
    in a pod.
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
        # Not as useful as the ones in aerodynamics, here it will just be run twice in the sizing
        # group
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]
        position = self.options["position"]
        # For refractoring purpose we just match the option to the tag in the variable name and
        # use it
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        # At least one input is needed regardless of the case
        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:width",
            units="m",
            val=np.nan,
            desc="Width of the battery, as in the size of the battery along the Y-axis",
        )

        self.add_output(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":"
            + ls_tag
            + ":CD0",
            val=0.0,
        )

        # Should not work but actually does. I expected the value to be zero everywhere but it
        # seems like this value is overwritten by the compute_partials function
        self.declare_partials(of="*", wrt="*", val=0.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        cd0 = 0.0

        outputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":"
            + ls_tag
            + ":CD0"
        ] = cd0

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        partials[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":"
            + ls_tag
            + ":CD0",
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:width",
        ] = 0.0
