# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

MAXIMUM_VOLTAGE_PER_LAYER = 1.2


class SizingBatteryNumberLayers(om.ExplicitComponent):
    """
    Computation of the total number of cells inside the battery based on the number of cells in
    each module and number of modules.
    """

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
            "fc_current_density",
            units="A/cm**2",
            val=np.nan,
        )

        self.add_input(
            "single_layer_pemfc_voltage",
            units="V",
            val=np.nan,
        )
        self.add_input(
            "effective_area", units="cm**2", val=16.8, desc="Effective fuel cell area in the stack"
        )

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            units="W",
            val=np.nan,
            desc="Maximum power needed by the pemfc during the mission",
        )

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":module:number_layers",
            val=np.nan,
            desc="Number of layers in series inside one pemfc stack",
        )

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_stacks",
            val=1.0,
            desc="Total number of stacks in the pemfc module",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]
        power_required = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"
        ]
        power_fc = (
            inputs["single_layer_pemfc_voltage"]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":module:number_layers"
            ]
            * inputs["fc_current_density"]
            * inputs["effective_area"]
        )

        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_stacks"
        ] = (power_required / power_fc)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]
        power_required = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"
        ]
        power_fc = (
            inputs["single_layer_pemfc_voltage"]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":module:number_layers"
            ]
            * inputs["fc_current_density"]
            * inputs["effective_area"]
        )
        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_stacks",
            "data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":module:number_layers",
        ] = (
            -power_required
            / power_fc
            / inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":module:number_layers"
            ]
        )

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_cells",
            "fc_current_density",
        ] = (
            -power_required / power_fc / inputs["fc_current_density"]
        )

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_cells",
            "effective_area",
        ] = (
            -power_required / power_fc / inputs["effective_area"]
        )

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_cells",
            "single_layer_pemfc_voltage",
        ] = (
            -power_required / power_fc / inputs["single_layer_pemfc_voltage"]
        )

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_cells",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
        ] = (
            1 / power_fc
        )
