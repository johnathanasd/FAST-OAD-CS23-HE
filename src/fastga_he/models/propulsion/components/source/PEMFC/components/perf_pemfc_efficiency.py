# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesPEMFCEfficiency(om.ExplicitComponent):
    """
    Computation of efficiency of the battery based on the losses at battery level and the output
    voltage and current.
    """

    def initialize(self):

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the pemfc stack",
            allow_none=False,
        )

    def setup(self):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            name="data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + "operation_pressure",
            units="atm",
            val=1.0,
        )

        self.add_input("nominal_pressure", units="atm", val=1.0)

        self.add_output(
            "single_layer_pemfc_voltage",
            units="V",
            val=np.nan,
        )

        self.add_output("efficiency", val=np.nan, lower=0.0, upper=1.0)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        E0 = 1.23  # ideal potential of the pemfc
        C = 0.06
        Pop = inputs["operation_pressure"]
        Pnom = inputs["nominal_pressure"]
        E = E0 + C * np.log(Pop / Pnom)
        efficiency = inputs["single_layer_pemfc_voltage"] / E
        outputs["efficiency"] = efficiency

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        E0 = 1.23  # ideal potential of the pemfc
        C = 0.06
        Pop = inputs["operation_pressure"]
        Pnom = inputs["nominal_pressure"]
        E = E0 + C * np.log(Pop / Pnom)

        partials["efficiency", "single_layer_pemfc_voltage"] = 1 / E

        partials["efficiency", "operation_pressure"] = -inputs["single_layer_pemfc_voltage"] / (
            C * Pop * np.log(Pop / Pnom) ** 2
        )
        partials["efficiency", "nominal_pressure"] = inputs["single_layer_pemfc_voltage"] / (
            C * Pnom * np.log(Pop / Pnom) ** 2
        )
