# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import logging

_LOGGER = logging.getLogger(__name__)


class PerformancesCurrentDensity(om.ExplicitComponent):
    """
    Computation of the current density, assume each stack provide an equal amount of current.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the pemfc stack",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]
        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))
        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            units="cm**2",
            val=16.8,
            desc="Effective fuel cell area in the stack",
        )

        self.add_output(
            "fc_current_density",
            val=np.full(number_of_points, 0.7),
            units="A/cm**2",
            desc="Current density of the pemfc stack",
        )

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

        self.declare_partials(
            of="*",
            wrt="dc_current_out",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        outputs["fc_current_density"] = (
            inputs["dc_current_out"]
            / inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials["fc_current_density", "dc_current_out"] = (
            np.ones(number_of_points)
            / inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
            ]
        )

        partials[
            "fc_current_density",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
        ] = (
            -inputs["dc_current_out"]
            / inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
            ]
            ** 2
        )
