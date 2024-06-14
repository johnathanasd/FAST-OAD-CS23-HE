# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from stdatm import AtmosphereWithPartials


class PerformancesNominalPressure(om.ExplicitComponent):
    """
    Computation of the PEMFC nominal pressure based on altitude only applied to the model
    from: `Fuel Cell and Battery Hybrid System Optimization by J. Hoogendoorn:2018`
    """


    def initialize(self):

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("altitude", shape=number_of_points, units="m", val=0.0)

        self.add_output(
            name="data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":nominal_pressure",
            units="Pa",
            val=np.full(number_of_points, 101325.0),
        )


        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs["data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":nominal_pressure"] = AtmosphereWithPartials(
            inputs["altitude"], altitude_in_feet=False
        ).pressure

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials["data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":nominal_pressure", "altitude"] = AtmosphereWithPartials(inputs["altitude"], altitude_in_feet=False
            ).partial_pressure_altitude

