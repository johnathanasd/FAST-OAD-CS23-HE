# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION


PRANDTL_NUMBER = 0.71
GRAVITY_ACCELERATION = 9.81  # m/s**2


class PerformancesCryogenicHydrogenTankNusseltNumber(om.ExplicitComponent):
    """
    Computation of the amount of the amount of hydrogen boil-off during the mission.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

        self.options.declare(
            name="cryogenic_hydrogen_tank_id",
            default=None,
            desc="Identifier of the cryogenic hydrogen tank",
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

        number_of_points = self.options["number_of_points"]
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        position = self.options["position"]

        self.add_input(
            name="air_kinematic_viscosity",
            units="m**2/s",
            val=np.full(number_of_points, np.nan),
        )

        self.add_output(
            "tank_nusselt_number",
            val=np.linspace(1.15, 0.15, number_of_points),
            desc="Tank Nusselt number at each time step",
        )

        if position == "wing_pod" or position == "underbelly":
            self.add_input(
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:overall_length",
                val=np.nan,
                units="m",
                desc="Value of the length of the tank in the x-direction, computed differently based "
                "on the location of the tank",
            )
            self.add_input(
                name="true_airspeed",
                units="m/s",
                val=np.full(number_of_points, np.nan),
            )
            self.declare_partials(
                of="tank_nusselt_number",
                wrt=["true_airspeed", "air_kinematic_viscosity"],
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
            )
            self.declare_partials(
                of="tank_nusselt_number",
                wrt="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:overall_length",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
            )

        else:
            self.add_input(
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:outer_diameter",
                units="m",
                val=np.nan,
                desc="Outer diameter of the hydrogen gas tank",
            )

            self.add_input(
                name="exterior_temperature",
                units="K",
                val=np.full(number_of_points, np.nan),
            )

            self.add_input(
                name="skin_temperature",
                units="K",
                val=np.full(number_of_points, np.nan),
            )
            self.declare_partials(
                of="tank_nusselt_number",
                wrt=["air_kinematic_viscosity", "exterior_temperature", "skin_temperature"],
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
            )
            self.declare_partials(
                of="tank_nusselt_number",
                wrt="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
                + cryogenic_hydrogen_tank_id
                + ":dimension:outer_diameter",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        position = self.options["position"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        if position == "wing_pod" or position == "underbelly":
            reynolds_number = (
                inputs["true_airspeed"]
                * inputs[input_prefix + ":dimension:overall_length"]
                / inputs["air_kinematic_viscosity"]
            )
            outputs["tank_nusselt_number"] = (
                0.03625 * PRANDTL_NUMBER ** 0.43 * reynolds_number ** 0.8
            )
        else:
            rayleigh_number = (
                GRAVITY_ACCELERATION
                * (1 - inputs["skin_temperature"] / inputs["exterior_temperature"])
                * inputs[input_prefix + ":dimension:outer_diameter"] ** 3
                * PRANDTL_NUMBER
                / inputs["air_kinematic_viscosity"]
            )
            outputs["tank_nusselt_number"] = 0.555 * rayleigh_number ** 0.25 + 0.447

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        position = self.options["position"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        if position == "wing_pod" or position == "underbelly":

            partials["tank_nusselt_number", "true_airspeed"] = (
                0.03625
                * PRANDTL_NUMBER ** 0.43
                * 0.8
                * (
                    inputs[input_prefix + ":dimension:overall_length"]
                    / inputs["air_kinematic_viscosity"]
                )
                ** 0.8
                / inputs["true_airspeed"] ** 0.2
            )

            partials["tank_nusselt_number", input_prefix + ":dimension:overall_length"] = (
                0.03625
                * PRANDTL_NUMBER ** 0.43
                * 0.8
                * (inputs["true_airspeed"] / inputs["air_kinematic_viscosity"]) ** 0.8
                / inputs[input_prefix + ":dimension:overall_length"] ** 0.2
            )

            partials["tank_nusselt_number", "air_kinematic_viscosity"] = (
                -0.03625
                * PRANDTL_NUMBER ** 0.43
                * 0.8
                * (inputs["true_airspeed"] * inputs[input_prefix + ":dimension:overall_length"])
                ** 0.8
                / inputs["air_kinematic_viscosity"] ** 1.8
            )

        else:
            rayleigh_number = (
                GRAVITY_ACCELERATION
                * (1 - inputs["skin_temperature"] / inputs["exterior_temperature"])
                * inputs[input_prefix + ":dimension:outer_diameter"] ** 3
                * PRANDTL_NUMBER
                / inputs["air_kinematic_viscosity"]
            )
            partials["tank_nusselt_number", "skin_temperature"] = (
                -0.25
                * 0.555
                * inputs[input_prefix + ":dimension:outer_diameter"] ** 3
                * GRAVITY_ACCELERATION
                * PRANDTL_NUMBER
                / (
                    inputs["exterior_temperature"]
                    * inputs["air_kinematic_viscosity"]
                    * rayleigh_number ** 0.75
                )
            )
            partials["tank_nusselt_number", "exterior_temperature"] = (
                0.555
                * 0.25
                * inputs[input_prefix + ":dimension:outer_diameter"] ** 3
                * GRAVITY_ACCELERATION
                * PRANDTL_NUMBER
                * inputs["skin_temperature"]
                / (
                    inputs["exterior_temperature"] ** 2
                    * inputs["air_kinematic_viscosity"]
                    * rayleigh_number ** 0.75
                )
            )
            partials["tank_nusselt_number", "air_kinematic_viscosity"] = (
                -0.555
                * 0.25
                * inputs[input_prefix + ":dimension:outer_diameter"] ** 3
                * GRAVITY_ACCELERATION
                * PRANDTL_NUMBER
                * (1 - inputs["skin_temperature"] / inputs["exterior_temperature"])
                / (rayleigh_number ** 0.75 * inputs["air_kinematic_viscosity"] ** 2)
            )
            partials["tank_nusselt_number", input_prefix + ":dimension:outer_diameter"] = (
                0.555
                * 0.75
                * GRAVITY_ACCELERATION
                * (1 - inputs["skin_temperature"] / inputs["exterior_temperature"])
                * inputs[input_prefix + ":dimension:outer_diameter"] ** 2
                * PRANDTL_NUMBER
                / inputs["air_kinematic_viscosity"]
                / (
                    GRAVITY_ACCELERATION
                    * (1 - inputs["skin_temperature"] / inputs["exterior_temperature"])
                    * inputs[input_prefix + ":dimension:outer_diameter"] ** 3
                    * PRANDTL_NUMBER
                    / inputs["air_kinematic_viscosity"]
                )
                ** 0.75
            )
