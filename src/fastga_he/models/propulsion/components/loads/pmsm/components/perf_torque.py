# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesTorque(om.ExplicitComponent):
    """Computation of the torque from power and rpm."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("shaft_power_out", units="W", val=np.nan, shape=number_of_points)
        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)

        self.add_output("torque", units="N*m", val=0.0, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        power = inputs["shaft_power_out"]
        rpm = inputs["rpm"]
        omega = rpm * 2.0 * np.pi / 60

        torque = power / omega

        outputs["torque"] = torque

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        power = inputs["shaft_power_out"]
        rpm = inputs["rpm"]

        omega = rpm * 2.0 * np.pi / 60

        partials["torque", "shaft_power_out"] = np.diag(1.0 / omega)
        partials["torque", "rpm"] = -np.diag(power / omega ** 2.0) * 2.0 * np.pi / 60
