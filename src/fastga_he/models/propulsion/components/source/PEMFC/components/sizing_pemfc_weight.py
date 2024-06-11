# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

FC_WEIGHT_DENSITY = 8.5e-8  # kg/m^2


class SizingPEMFCWeight(om.ExplicitComponent):
    """
    Computation of the weight the PEMFC based on the layer weight density of the stack.
    """

    # TODO: Adding another way of mass estimation from D.Juschus
    def initialize(self):

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the pemfc pack",
            allow_none=False,
        )

    def setup(self):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers",
            val=np.nan,
            desc="Number of layer in 1 PEMFC stack",
        )

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            units="m**2",
            val=np.nan,
            desc="Effective fuel cell area in the stack",
        )

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass",
            units="kg",
            val=50.0,
            desc="Mass of the pemfc stack",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass"] = (
            FC_WEIGHT_DENSITY
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers"
            ]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers",
        ] = (
            FC_WEIGHT_DENSITY
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
            ]
        )
        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
        ] = (
            FC_WEIGHT_DENSITY
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers"
            ]
        )
