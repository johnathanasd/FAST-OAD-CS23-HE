# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesHydrogenConsumption(om.ExplicitComponent):
    """
    Computation of the hydrogen consumption for the required power. Simply based on the
    results of the currrent density and effective area
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "effective_area", units="cm**2", val=16.8, desc="Effective fuel cell area in the stack"
        )

        self.add_input(
            "fc_current_density",
            val=np.nan,
            shape=number_of_points,
            units="A/cm**2",
            desc="Current density of the pemfc stack",
        )

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":total_number_layers",
            val=np.nan,
            desc="Total number of layers in the pemfc stacks",
        )

        self.add_output("hydrogen_consumption", units="kg/h", val=10.0, shape=number_of_points)

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs["hydrogen_consumption"] = (
            inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":total_number_layers"
            ]
            * inputs["fc_current_density"]
            * inputs["effective_area"]
            * 3600
            / (2 * 96500 * 500)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "hydrogen_consumption",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":total_number_layers",
        ] = (
            np.ones(number_of_points)
            * inputs["fc_current_density"]
            * inputs["effective_area"]
            * 3600
            / (2 * 96500 * 500)
        )
        partials["hydrogen_consumption", "fc_current_density"] = (
            inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":total_number_layers"
            ]
            * inputs["effective_area"]
            * 3600
            / (2 * 96500 * 500)
        )
        partials["hydrogen_consumption", "effective_area"] = (
            inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":total_number_layers"
            ]
            * inputs["fc_current_density"]
            * 3600
            / (2 * 96500 * 500)
        )
