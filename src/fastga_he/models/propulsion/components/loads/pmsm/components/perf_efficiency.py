# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesEfficiency(om.ExplicitComponent):
    """Computation of the efficiency from shaft power and power losses."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("shaft_power", units="W", val=np.nan, shape=number_of_points)
        self.add_input("power_losses", units="W", val=np.nan, shape=number_of_points)

        self.add_output("efficiency", val=np.full(number_of_points, 0.95), shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["efficiency"] = inputs["shaft_power"] / (
            inputs["shaft_power"] + inputs["power_losses"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["efficiency", "shaft_power"] = np.diag(
            inputs["power_losses"] / (inputs["shaft_power"] + inputs["power_losses"]) ** 2.0
        )
        partials["efficiency", "power_losses"] = -np.diag(
            inputs["shaft_power"] / (inputs["shaft_power"] + inputs["power_losses"]) ** 2.0
        )
