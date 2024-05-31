# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np



# TODO: check the naming of the power required as an input
class SizingPEMFCNumberStacks(om.ExplicitComponent):
    """
    Computation of the total number of stacks based on the maximum voltage required.
    """

    def initialize(self):

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):

        self.add_input("power_required", units="W", val=np.nan)

        self.add_input("ac_voltage_peak_in", units="V", val=np.nan)

        self.add_input(
            "single_stack_pemfc_current",
            units="A",
            val=np.nan,
        )

        self.add_output(
            "number_stacks",
            val=100.0,
            desc="Total number of pemfc stacks",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        I = inputs["power_required"]/inputs["ac_voltage_peak_in"]

        num_stacks = I/inputs["single_stack_pemfc_current"]
        if num_stacks > int(num_stacks):
            num_stacks = int(num_stacks) + 1

        outputs[
            "number_stacks"
        ] = num_stacks

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials[
            "number_stacks",
            "power_required",
        ] = 1/(inputs["ac_voltage_peak_in"]*inputs["single_stack_pemfc_current"])

        partials[
            "number_stacks",
            "ac_voltage_peak_in",
        ] = - inputs["power_required"]/(inputs["ac_voltage_peak_in"]**2*inputs["single_stack_pemfc_current"])

        partials[
            "number_stacks",
            "single_stack_pemfc_current",
        ] = - inputs["power_required"] / (inputs["ac_voltage_peak_in"] * inputs["single_stack_pemfc_current"]**2)
