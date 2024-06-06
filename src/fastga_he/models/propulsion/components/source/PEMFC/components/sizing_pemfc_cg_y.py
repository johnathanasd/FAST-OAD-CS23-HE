# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingPEMFCCGY(om.ExplicitComponent):
    """
    Class that computes the y - CG of the pemfc according to the position given in the options.
    """

    def initialize(self):

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the pemfc pack",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="underbelly",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the pemfc, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]

        # At least one input is needed regardless of the case
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":CG:y",
            units="m",
            val=0.0,
            desc="Y position of the pemfc center of gravity",
        )

        if position == "wing_pod":

            self.add_input(
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":CG:y_ratio",
                val=np.nan,
                desc="X position of the pemfc center of gravity as a ratio of the wing half-span",
            )

            self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]

        if position == "wing_pod":

            outputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":CG:y"] = (
                inputs["data:geometry:wing:span"]
                * inputs[
                    "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":CG:y_ratio"
                ]
                / 2.0
            )

        else:

            outputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":CG:y"
            ] = 0.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]

        if position == "wing_pod":

            partials[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":CG:y",
                "data:geometry:wing:span",
            ] = (
                inputs[
                    "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":CG:y_ratio"
                ]
                / 2.0
            )
            partials[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":CG:y",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":CG:y_ratio",
            ] = (
                inputs["data:geometry:wing:span"] / 2.0
            )
