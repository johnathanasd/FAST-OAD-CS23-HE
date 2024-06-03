# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingPEMFCNumberTotalLayers(om.ExplicitComponent):
    """
    Computation of the total number of layers inside the pemfc based on the number of layers in
    each stack and number of stacks.
    """

    def initialize(self):

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":module:number_layers",
            val=np.nan,
            desc="Number of layers in series inside one pemfc stack",
        )

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_stacks",
            val=np.nan,
            desc="Total number of stacks in the pemfc module",
        )

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":total_number_layers",
            val=1000.0,
            desc="Total number of layers in the pemfc stacks",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":total_number_layers"
        ] = (
            inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":module:number_layers"
            ]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_stacks"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":total_number_layers",
            "data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":module:number_layers",
        ] = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_stacks"
        ]

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":total_number_layers",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_stacks",
        ] = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":module:number_layers"
        ]
