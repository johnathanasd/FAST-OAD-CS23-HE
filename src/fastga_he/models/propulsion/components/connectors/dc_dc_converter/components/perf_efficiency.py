# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesEfficiency(om.ExplicitComponent):
    """Computation of the efficiency of the inverter."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "losses_converter",
            units="W",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
        )
        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))
        self.add_input("dc_voltage_out", units="V", val=np.full(number_of_points, np.nan))

        self.add_output("efficiency", val=np.full(number_of_points, 1.0))

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["efficiency"] = (
            inputs["dc_current_out"]
            * inputs["dc_voltage_out"]
            / (inputs["dc_current_out"] * inputs["dc_voltage_out"] + inputs["losses_converter"])
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        losses_converter = inputs["losses_converter"]
        dc_current_out = inputs["dc_current_out"]
        voltage_out = inputs["dc_voltage_out"]

        useful_power = dc_current_out * voltage_out

        partials["efficiency", "dc_voltage_out"] = np.diag(
            dc_current_out * losses_converter / (useful_power + losses_converter) ** 2.0
        )
        partials["efficiency", "dc_current_out"] = np.diag(
            voltage_out * losses_converter / (useful_power + losses_converter) ** 2.0
        )
        partials["efficiency", "losses_converter"] = np.diag(
            -useful_power / (useful_power + losses_converter) ** 2.0
        )
