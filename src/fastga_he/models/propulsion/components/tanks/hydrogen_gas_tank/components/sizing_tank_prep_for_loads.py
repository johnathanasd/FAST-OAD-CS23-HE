# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingFuelTankPreparationForLoads(om.ExplicitComponent):
    """
    This components computes the values required for the computation of the influence of the fuel
    tanks on the fuselage and wing loads. For the wing: Will compute the y ratios for the start
    and end, slope is gonna be assumed to be one (as is done in the computation of dimensions)
    and the chord will be assumed at the mean chord (as is done in the computation of
    dimensions). For the fuselage: nothing for now.
    In any case, this component will only be added if the fuel tanks is inside the wing or in the
    fuselage. If the fuel tank is in a pod, it will be treated as a punctual mass !
    """

    def initialize(self):
        self.options.declare(
            name="fuel_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the fuel tank, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        fuel_tank_id = self.options["fuel_tank_id"]
        position = self.options["position"]

        if position == "inside_the_wing":

            self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
            self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
            self.add_input("data:geometry:wing:span", val=np.nan, units="m")

            self.add_input(
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width",
                units="m",
                val=np.nan,
                desc="Value of the length of the tank in the y-direction, computed differently based "
                "on the location of the tank",
            )
            self.add_input(
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":CG:y_ratio",
                val=np.nan,
                desc="Y position of the fuel tank center of gravity as a ratio of the wing half-span",
            )

            self.add_output(
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":distributed_tanks:y_ratio_start",
                val=0.15,
                desc="When tank is considered as a distributed mass, the position of the start as a ratio of the half-span",
            )
            self.add_output(
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":distributed_tanks:y_ratio_end",
                val=0.9,
                desc="When tank is considered as a distributed mass, the position of the end as a ratio of the half-span",
            )
            self.add_output(
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":distributed_tanks:start_chord",
                units="m",
                val=1.0,
                desc="When tank is considered as a distributed mass, the chord at the start",
            )
            self.add_output(
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":distributed_tanks:chord_slope",
                val=0.0,
                desc="When tank is considered as a distributed mass, the rate at which the chord varies",
            )

            self.declare_partials(
                of=[
                    "data:propulsion:he_power_train:fuel_tank:"
                    + fuel_tank_id
                    + ":distributed_tanks:y_ratio_start",
                    "data:propulsion:he_power_train:fuel_tank:"
                    + fuel_tank_id
                    + ":distributed_tanks:y_ratio_end",
                ],
                wrt=[
                    "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":CG:y_ratio",
                    "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width",
                    "data:geometry:wing:span",
                ],
                method="exact",
            )
            self.declare_partials(
                of="data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":distributed_tanks:start_chord",
                wrt=["data:geometry:wing:tip:chord", "data:geometry:wing:root:chord"],
                val=0.5,
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fuel_tank_id = self.options["fuel_tank_id"]
        position = self.options["position"]

        if position == "inside_the_wing":

            root_chord = inputs["data:geometry:wing:root:chord"]
            tip_chord = inputs["data:geometry:wing:tip:chord"]
            half_span = inputs["data:geometry:wing:span"] / 2.0

            cg_y_ratio = inputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":CG:y_ratio"
            ]
            tank_width = inputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width"
            ]

            y_ratio_start = cg_y_ratio - 0.5 * tank_width / half_span
            y_ratio_end = cg_y_ratio + 0.5 * tank_width / half_span

            chord_start = (root_chord + tip_chord) / 2.0

            outputs[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":distributed_tanks:y_ratio_start"
            ] = y_ratio_start
            outputs[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":distributed_tanks:y_ratio_end"
            ] = y_ratio_end
            outputs[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":distributed_tanks:start_chord"
            ] = chord_start

            outputs[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":distributed_tanks:chord_slope"
            ] = 0.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        fuel_tank_id = self.options["fuel_tank_id"]
        position = self.options["position"]

        if position == "inside_the_wing":
            half_span = inputs["data:geometry:wing:span"] / 2.0

            tank_width = inputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width"
            ]

            partials[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":distributed_tanks:y_ratio_start",
                "data:geometry:wing:span",
            ] = (
                0.25 * tank_width / half_span ** 2.0
            )
            partials[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":distributed_tanks:y_ratio_start",
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":CG:y_ratio",
            ] = 1.0
            partials[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":distributed_tanks:y_ratio_start",
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width",
            ] = (
                -0.5 / half_span
            )

            partials[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":distributed_tanks:y_ratio_end",
                "data:geometry:wing:span",
            ] = (
                -0.25 * tank_width / half_span ** 2.0
            )
            partials[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":distributed_tanks:y_ratio_end",
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":CG:y_ratio",
            ] = 1.0
            partials[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":distributed_tanks:y_ratio_end",
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width",
            ] = (
                0.5 / half_span
            )
