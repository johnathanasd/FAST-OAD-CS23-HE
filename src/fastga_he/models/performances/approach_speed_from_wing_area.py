"""
Computation of approach speed with wing area in low speed
conditions with simple computation.
"""

#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import openmdao.api as om
from fastoad.module_management.constants import ModelDomain

from scipy.constants import g

import fastoad.api as oad


# give to huang
@oad.RegisterOpenMDAOSystem("fastga.performance.approach_speed.retrofit", domain=ModelDomain.OTHER)
class UpdateWingAreaLiftSimple(om.ExplicitComponent):
    """
    Computes needed wing area to have enough lift at required approach speed.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)

    def setup(self):

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)

        self.add_output("data:TLAR:v_approach", val=30.0, units="m/s")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # inputs for approach speed computation
        wing_area_approach = inputs["data:geometry:wing:area"]
        mlw = inputs["data:weight:aircraft:MLW"]
        max_cl = inputs["data:aerodynamics:aircraft:landing:CL_max"]

        # output computation for stalling speed and approach speed
        stall_speed = np.sqrt((2 * mlw * g) / (wing_area_approach * 1.225 * max_cl))
        approach_speed = stall_speed * 1.3

        outputs["data:TLAR:v_approach"] = approach_speed
