# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesPEMFCFuelConsumed(om.ExplicitComponent):
    """
    Computation of the hydrogen at each flight point for the required power. Simply based on the
    results of the hydrogen consumption and time
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(name="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":fuel_consumption", units="kg/h", val=np.full(number_of_points, np.nan))
        self.add_input("time_step", units="h", val=np.full(number_of_points, np.nan))

        self.add_output(name="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":fuel_consumed_t", val = np.full(number_of_points, 1.0), units="kg")

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        outputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":fuel_consumed_t"] = inputs["time_step"] * inputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":fuel_consumption"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        partials["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":fuel_consumed_t", "time_step"] = inputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":fuel_consumption"]
        partials["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":fuel_consumed_t", "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":fuel_consumption"] = inputs["time_step"]
