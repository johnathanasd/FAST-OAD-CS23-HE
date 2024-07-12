# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingDCAuxLoadCGX(om.ExplicitComponent):
    """
    Computation of the position, along the x-axis of the CG of the auxiliary load. If the
    auxiliary load is inside the wing we will consider it at the quarter chord.
    """

    def initialize(self):

        self.options.declare(
            name="aux_load_id",
            default=None,
            desc="Identifier of the auxiliary load",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="in_the_front",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the auxiliary load, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        aux_load_id = self.options["aux_load_id"]
        position = self.options["position"]

        self.add_output(
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":CG:x",
            units="m",
            val=2.5,
            desc="X position of the auxiliary load center of gravity",
        )

        if position == "inside_the_wing":

            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        elif position == "in_the_front":

            self.add_input(
                name="data:propulsion:he_power_train:aux_load:"
                + aux_load_id
                + ":front_length_ratio",
                val=0.9,
                desc="Location of the auxiliary load as a ratio of the aircraft front length",
            )
            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")

        else:

            self.add_input(
                name="data:propulsion:he_power_train:aux_load:"
                + aux_load_id
                + ":rear_length_ratio",
                val=0.1,
                desc="Location of the auxiliary load CG as a ratio of the aircraft rear length",
            )
            self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        aux_load_id = self.options["aux_load_id"]
        position = self.options["position"]

        if position == "inside_the_wing":

            outputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":CG:x"] = inputs[
                "data:geometry:wing:MAC:at25percent:x"
            ]

        elif position == "in_the_front":

            lav = inputs["data:geometry:fuselage:front_length"]

            outputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":CG:x"] = (
                lav
                * inputs[
                    "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":front_length_ratio"
                ]
            )

        else:

            cabin_length = inputs["data:geometry:cabin:length"]
            lav = inputs["data:geometry:fuselage:front_length"]
            lar = inputs["data:geometry:fuselage:rear_length"]

            outputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":CG:x"] = (
                lav
                + cabin_length
                + lar
                * inputs[
                    "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":rear_length_ratio"
                ]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        aux_load_id = self.options["aux_load_id"]
        position = self.options["position"]

        if position == "inside_the_wing":

            partials[
                "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":CG:x",
                "data:geometry:wing:MAC:at25percent:x",
            ] = 1.0

        elif position == "in_the_front":

            partials[
                "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":CG:x",
                "data:geometry:fuselage:front_length",
            ] = inputs[
                "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":front_length_ratio"
            ]
            partials[
                "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":CG:x",
                "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":front_length_ratio",
            ] = inputs["data:geometry:fuselage:front_length"]

        else:

            partials[
                "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":CG:x",
                "data:geometry:fuselage:front_length",
            ] = 1.0
            partials[
                "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":CG:x",
                "data:geometry:cabin:length",
            ] = 1.0
            partials[
                "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":CG:x",
                "data:geometry:fuselage:rear_length",
            ] = inputs[
                "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":rear_length_ratio"
            ]
            partials[
                "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":CG:x",
                "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":rear_length_ratio",
            ] = inputs["data:geometry:fuselage:rear_length"]
