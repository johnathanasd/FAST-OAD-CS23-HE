# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum power output from PEMFC.
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

        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("power_out", units="W", val=np.full(number_of_points, np.nan))

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_min",
            units="W",
            val=100,
            desc="Minimum power provided by the pemfc during the mission",
        )
        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            units="W",
            val=1000,
            desc="Maximum power needed by the pemfc during the mission",
        )
        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"
        ] = np.max(inputs["power_out"])
        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_min"
        ] = np.min(inputs["power_out"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            "power_out",
        ] = np.where(inputs["power_out"] == np.max(inputs["power_out"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_min",
            "power_out",
        ] = np.where(inputs["power_out"] == np.min(inputs["power_out"]), 1.0, 0.0)
