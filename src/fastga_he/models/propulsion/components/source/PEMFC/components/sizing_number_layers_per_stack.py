# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

MAXIMUM_VOLTAGE_PER_LAYER = 1.2  # voltage


class SizingPEMFCNumberLayers(om.ExplicitComponent):
    """
    Computation of the total number of layers inside one stack based on the maximum voltage required.
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

        self.add_input("ac_voltage_peak_in", units="V", val=np.nan)

        self.add_input(
            "single_layer_pemfc_voltage",
            units="V",
            val=np.nan,
        )

        self.add_output(
            "number_layers_per_stack",
            val=100.0,
            desc="Total number of layers in one stack",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        num_layers = inputs["ac_voltage_peak_in"] / inputs["single_layer_pemfc_voltage"]
        if num_layers > int(num_layers):
            num_layers = int(num_layers) + 1

        outputs["number_layers_per_stack"] = num_layers

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["number_layers_per_stack", "single_layer_pemfc_voltage",] = (
            inputs["ac_voltage_peak_in"] / inputs["single_layer_pemfc_voltage"] ** 2
        )

        partials["number_layers_per_stack", "ac_voltage_peak_in",] = (
            1 / inputs["single_layer_pemfc_voltage"]
        )
