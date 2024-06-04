# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingPEMFCWeight(om.ExplicitComponent):
    """
    Computation of the weight the battery based on the weight of the modules.
    """

    # TODO: Adding another way of mass estimation from D.Juschus
    def initialize(self):

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass_per_layer",
            units="kg",
            val=0.014286,
            desc="Mass of one layer of the pemfc stack",
        )
        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":total_number_layers",
            val=np.nan,
            desc="Total number of layers in the pemfc stacks",
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
            inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":total_number_layers"
            ]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass_per_layer"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":total_number_layers",
        ] = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass_per_layer"
        ]
        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass_per_layer",
        ] = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":total_number_layers"
        ]
