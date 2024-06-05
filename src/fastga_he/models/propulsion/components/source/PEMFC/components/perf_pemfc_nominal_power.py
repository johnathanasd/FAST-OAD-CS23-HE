# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesPEMFCNominalPower(om.ExplicitComponent):
    """
    Computation of the power at the output of the battery. As of when I wrote this, it will only
    be used as a post-processing value
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the pemfc stack",
            allow_none=False,
        )

    def setup(self):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "fc_current_density",
            units="A/cm**2",
            val=np.nan,
        )

        self.add_input(
            "effective_area", units="cm**2", val=16.8, desc="Effective fuel cell area in the stack"
        )

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":total_number_layers",
            val=np.nan,
            desc="Total number of layers in the pemfc stacks",
        )

        self.add_output("nominal_power", units="kW", val=30)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        V0 = 0.83
        B = 0.014
        R = 0.24
        m = 5.63 * 10 ** -6
        n = 11.42
        i = inputs["fc_current_density"]
        Vc = V0 - B * np.log(i) - R * i - m * np.exp(n * i)
        outputs["nominal_power"] = (
            Vc
            * i
            * inputs["effective_area"]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":total_number_layers"
            ]
            / 1000.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        V0 = 0.83
        B = 0.014
        R = 0.24
        m = 5.63 * 10 ** -6
        n = 11.42
        i = inputs["fc_current_density"]
        Vc = V0 - B * np.log(i) - R * i - m * np.exp(n * i)
        dVc = -R - m * n * np.exp(n * i) - B / i

        partials["nominal_power", "effective_area"] = (
            Vc
            * i
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":total_number_layers"
            ]
            / 1000.0
        )

        partials[
            "nominal_power",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":total_number_layers",
        ] = (
            Vc * i * inputs["effective_area"] / 1000.0
        )

        partials["nominal_power", "fc_current_density"] = (
            Vc
            * inputs["effective_area"]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":total_number_layers"
            ]
            / 1000.0
            + dVc
            * i
            * inputs["effective_area"]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":total_number_layers"
            ]
            / 1000.0
        )
