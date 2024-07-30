# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION


class PerformancesAirThermalConductivity(om.ExplicitComponent):
    """
    Computation of the air thermal conductivity based on free stream temperature
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="free_stream_temperature",
            units="degC",
            val=np.full(number_of_points, np.nan),
        )

        self.add_output(
            "air_thermal_conductivity",
            units="W/m/K",
            val=np.full(number_of_points, 0.024),
            desc="Tank Nusselt number at each time step",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            val=np.full(number_of_points, 0.00007),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["air_thermal_conductivity"] = 0.024 + 0.00007 * inputs["free_stream_temperature"]
