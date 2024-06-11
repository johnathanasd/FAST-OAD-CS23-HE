# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from stdatm import Atmosphere

DENSITY_SL = Atmosphere(0.0).density


class PerformancesDensityRatio(om.ExplicitComponent):
    """Computation of the density ratio, often called sigma in literature used in the surrogate."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        # RPM is not used to compute anything but is needed for compatibility reasons
        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input("density", units="kg/m**3", val=np.nan, shape=number_of_points)

        self.add_output("density_ratio", val=1.0, shape=number_of_points, lower=0.0)

        self.declare_partials(of="*", wrt="density", val=np.eye(number_of_points) / DENSITY_SL)
        self.declare_partials(of="*", wrt="rpm", val=np.zeros((number_of_points, number_of_points)))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["density_ratio"] = inputs["density"] / DENSITY_SL
