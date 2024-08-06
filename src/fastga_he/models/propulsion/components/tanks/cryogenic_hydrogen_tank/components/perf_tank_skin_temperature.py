# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

SKIN_TEMPERATURE_MAX = 280.0
SKIN_TEMPERATURE_MIN = 30.0


class PerformancesLiquidHydrogenTankSkinTemperature(om.ExplicitComponent):
    """
    Computation of the amount of the amount of hydrogen remaining inside the tank.
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

    def setup(self):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        number_of_points = self.options["number_of_points"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        self.add_input(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":insulation:thermal_resistance",
            units="K/W",
            val=14.5,
            desc="Thermal resistance of the tank wall including the insulation layer",
        )

        self.add_input(
            name="data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":liquid_hydrogen_temperature",
            units="K",
            val=np.nan,
            desc="Liquid hydrogen temperature in the tank",
        )

        self.add_input(
            "heat_conduction",
            units="W",
            val=np.full(number_of_points, 17.23),
            desc="Tank wall heat conduction at each time step",
        )

        self.add_output(
            "skin_temperature",
            units="K",
            val=np.full(number_of_points, 273.15),
            desc="skin temperature of the tank exterior",
        )

        self.declare_partials(
            of="*",
            wrt="heat_conduction",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="*",
            wrt=[
                input_prefix + ":insulation:thermal_resistance",
                input_prefix + ":liquid_hydrogen_temperature",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        number_of_points = self.options["number_of_points"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        unclipped_t_skin = inputs[input_prefix + ":insulation:thermal_resistance"] * inputs[
            "heat_conduction"
        ] + inputs[input_prefix + ":liquid_hydrogen_temperature"] * np.ones(number_of_points)

        outputs["skin_temperature"] = np.clip(
            unclipped_t_skin, SKIN_TEMPERATURE_MIN, SKIN_TEMPERATURE_MAX
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        number_of_points = self.options["number_of_points"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        unclipped_t_skin = inputs[input_prefix + ":insulation:thermal_resistance"] * inputs[
            "heat_conduction"
        ] + inputs[input_prefix + ":liquid_hydrogen_temperature"] * np.ones(number_of_points)

        partials["skin_temperature", input_prefix + ":insulation:thermal_resistance",] = np.where(
            (unclipped_t_skin <= SKIN_TEMPERATURE_MAX) & (unclipped_t_skin >= SKIN_TEMPERATURE_MIN),
            inputs["heat_conduction"],
            np.full_like(inputs["heat_conduction"], 1e-6),
        )

        partials["skin_temperature", "heat_conduction",] = np.where(
            (unclipped_t_skin <= SKIN_TEMPERATURE_MAX) & (unclipped_t_skin >= SKIN_TEMPERATURE_MIN),
            inputs[input_prefix + ":insulation:thermal_resistance"] * np.ones(number_of_points),
            np.full_like(inputs["heat_conduction"], 1e-6),
        )

        partials["skin_temperature", input_prefix + ":liquid_hydrogen_temperature",] = np.where(
            (unclipped_t_skin <= SKIN_TEMPERATURE_MAX) & (unclipped_t_skin >= SKIN_TEMPERATURE_MIN),
            np.ones(number_of_points),
            np.full_like(inputs["heat_conduction"], 1e-6),
        )
