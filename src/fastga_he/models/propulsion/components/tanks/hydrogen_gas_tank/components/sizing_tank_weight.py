# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
import logging
#TODO: finish the partial part and think about how define the partial of the material choice
_LOGGER = logging.getLogger(__name__)
# To modify
class SizingHydrogenGasTankWeight(om.ExplicitComponent):
    """
    Computation of the weight of the tank. The very simplistic approach we will use is to say
    that weight of tank is the weight of unused fuel and the weight of the tank itself.
    material density are cite from: Hydrogen Storage for Aircraft Application Overview, NASA 2002
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
            + ":unusable_fuel_mission",
            units="kg",
            val=np.nan,
            desc="Amount of trapped hydrogen gas in the tank",
        )

        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:" + hydrogen_gas_tank_id + ":volume",
            units="m**3",
            val=np.nan,
            desc="Capacity of the tank in terms of volume",
        )

        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:length",
            val=np.nan,
            units="m",
            desc="Value of the length of the tank in the x-direction",
        )
        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the hydrogen gas tank",
        )

        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":tank_material",
            val=np.nan,
            desc="Choice of the tank material 1:Steel(ASTM-A514), 2:Aluminum(2014-T6), 3:Titanium(6%Al,4%V), 4:Carbon Composite",
        )

        self.add_output(
            name="data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":mass",
            units="kg",
            val=1.0,
            desc="Weight of the hydrogen gas tanks",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]
        material = inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":tank_material"
        ]
        if material == 1.0:
            wall_density = 7860.0  # kg/m^3
        elif material == 2.0:
            wall_density = 2800.0  # kg/m^3
        elif material == 3.0:
            wall_density = 4460  # kg/m^3
        elif material == 4.0:
            wall_density = 1530  # kg/m^3
        else:
            wall_density = 7860.0  # kg/m^3
        _LOGGER.warning("Fuel type %f does not exist, replaced by type 1!", fuel_type)
        r = (
            inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":dimension:diameter"
            ]
            / 2
        )
        l = inputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":dimension:length"
        ]

        outputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:" + hydrogen_gas_tank_id + ":mass"
        ] = wall_density * (
            0.75 * np.pi * r ** 3
            + np.pi * r ** 2 * l
            - inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":volume"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]
