# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingCryogenicHydrogenTankDiameterUpdate(om.ExplicitComponent):
    """
    Update Diameter variable to activate length adjustment if negative length occurs
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
            val=np.nan,
            units="m",
        )

        self.add_output(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:diameter",
            val=10.,
            units="m",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            val=1.0,
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        outputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:diameter"
        ] = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:outer_diameter"
        ]
