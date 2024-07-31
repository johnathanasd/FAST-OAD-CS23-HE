# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION
import logging

_LOGGER = logging.getLogger(__name__)


class SizingCryogenicHydrogenTankThermalResistance(om.ExplicitComponent):
    """
    Computation of the weight of the tank. The very simplistic approach we will use is to say
    that weight of tank is the weight of the tank itself.
    Reference material density are cite from: Hydrogen Storage for Aircraft Application Overview, NASA 2002
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
            + ":dimension:length",
            val=np.nan,
            units="m",
            desc="Value of the length of the tank in the x-direction",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:outer_diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the hydrogen gas tank",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":material:thermal_conductivity",
            units="W/m/K",
            val=np.nan,
            desc="Thermal conductivity of the tank wall material",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:thermal_conductivity",
            units="W/m/K",
            val=np.nan,
            desc="Thermal conductivity of the insulation material",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:insulation_thickness",
            units="m",
            val=np.nan,
            desc="Insulation layer thickness of the cryogenic hydrogen tank",
        )

        self.add_output(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:thermal_resistance",
            units="K/W",
            val=50.0,
            desc="Thermal resistance of the tank wall including the insulation layer",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        wall_conductivity = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":material:thermal_conductivity"
        ]

        insulation_conductivity = inputs[input_prefix + ":insulation:thermal_conductivity"]

        d = inputs[input_prefix + ":dimension:outer_diameter"]

        t = inputs[input_prefix + ":dimension:insulation_thickness"]

        l = inputs[input_prefix + ":dimension:length"]

        resistance_cylindrical = np.log(d / (d - 2 * t)) / (
            2 * np.pi * l * wall_conductivity
        ) + np.log((d + 2 * t) / d) / (2 * np.pi * l * insulation_conductivity)

        resistance_spherical = (1 / d + 1 / (d - 2 * t)) / (2 * np.pi * wall_conductivity) + (
            1 / (d + 2 * t) + 1 / d
        ) / (2 * np.pi * insulation_conductivity)

        outputs[input_prefix + ":insulation:thermal_resistance"] = (
            1 / resistance_cylindrical + 1 / resistance_spherical
        ) ** -1

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        wall_conductivity = inputs[input_prefix + ":material:thermal_conductivity"]

        insulation_conductivity = inputs[input_prefix + ":insulation:thermal_conductivity"]

        d = inputs[input_prefix + ":dimension:outer_diameter"]

        t = inputs[input_prefix + ":dimension:insulation_thickness"]

        l = inputs[input_prefix + ":dimension:length"]

        rc_n1 = np.log(d / (d - 2 * t))

        rc_d1 = 2 * np.pi * l * wall_conductivity

        rc_n2 = np.log((d + 2 * t) / d)

        rc_d2 = 2 * np.pi * l * insulation_conductivity

        resistance_cylindrical = rc_n1 / rc_d1 + rc_n2 / rc_d2

        rs_n1 = 1 / d + 1 / (d - 2 * t)

        rs_d1 = 2 * np.pi * wall_conductivity

        rs_n2 = 1 / (d + 2 * t) + 1 / d

        rs_d2 = 2 * np.pi * insulation_conductivity

        resistance_spherical = rs_n1 / rs_d1 + rs_n2 / rs_d2

        partials[
            input_prefix + ":insulation:thermal_resistance", input_prefix + ":dimension:length"
        ] = -(
            (rc_d1 * rc_d2 / l ** 2)
            * (rc_n1 * rc_d2 / l + rc_n2 * rc_d1 / l)
            * resistance_spherical ** 2
            / (rc_d1 * rc_d2 / l ** 2 + rc_n1 * rc_d2 / l + rc_n2 * rc_d1 / l) ** 2
        )

        partials[
            input_prefix + ":insulation:thermal_resistance",
            input_prefix + ":material:thermal_conductivity",
        ] = (
            -(
                (rs_n1 / (2 * np.pi * resistance_spherical ** 2 * wall_conductivity ** 2))
                + rc_n1 / (2 * np.pi * l * resistance_cylindrical ** 2 * wall_conductivity ** 2)
            )
            / (1 / resistance_cylindrical + 1 / resistance_spherical) ** 2
        )

        partials[
            input_prefix + ":insulation:thermal_resistance",
            input_prefix + ":insulation:thermal_conductivity",
        ] = (
            -(
                (rs_n2 / (2 * np.pi * resistance_spherical ** 2 * insulation_conductivity ** 2))
                + rc_n2
                / (2 * np.pi * l * resistance_cylindrical ** 2 * insulation_conductivity ** 2)
            )
            / (1 / resistance_cylindrical + 1 / resistance_spherical) ** 2
        )

        partials[
            input_prefix + ":insulation:thermal_resistance",
            input_prefix + ":dimension:insulation_thickness",
        ] = (
            ((2 / (d + 2 * t) / rc_d2 + 2 / (d - 2 * t) / rc_d1) / resistance_cylindrical ** 2)
            + (
                (2 / rs_d1 / (d - 2 * t) ** 2 - 2 / rs_d2 / (d + 2 * t) ** 2)
                / resistance_spherical ** 2
            )
        ) / (
            1 / resistance_spherical + 1 / resistance_cylindrical
        ) ** 2

        partials[
            input_prefix + ":insulation:thermal_resistance",
            input_prefix + ":dimension:outer_diameter",
        ] = (
            (
                ((1 - d / (d - 2 * t)) / rc_d1 / d + (1 - (d + 2 * t) / d) / rc_d2 / (d + 2 * t))
                / resistance_cylindrical ** 2
            )
            + (
                (-((d + 2 * t) ** -2 + d ** -2) / rs_d2 - ((d - 2 * t) ** -2 + d ** -2) / rs_d1)
                / resistance_spherical ** 2
            )
        ) / (
            1 / resistance_spherical + 1 / resistance_cylindrical
        ) ** 2
