# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingPEMFCVolume(om.ExplicitComponent):
    """
    Computation of the volume the battery based on the volume of the modules.
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
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":volume_per_layer",
            units="L",
            val=0.04571,
            desc="Volume of one layer of the pemfc stack",
        )
        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":total_number_layers",
            val=np.nan,
            desc="Total number of layers in the pemfc stacks",
        )

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":volume",
            units="L",
            val=5.0,
            desc="Volume of the pemfc stack",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":volume"] = (
            inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":total_number_layers"
            ]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":volume_per_layer"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":volume",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":total_number_layers",
        ] = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":volume_per_layer"
        ]
        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":volume",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":volume_per_layer",
        ] = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":total_number_layers"
        ]
