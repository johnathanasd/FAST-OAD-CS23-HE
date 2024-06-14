# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesPEMFCEfficiency(om.ExplicitComponent):
    """
    Computation of efficiency of the battery based on the losses at battery level and the output
    voltage and current.
    """

    def initialize(self):

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the pemfc stack",
            allow_none=False,
        )

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":operation_pressure",
            units="atm",
            val=1.0,
        )

        self.add_input(
            name="data:propulsion:he_power_train:pemfc_stack:"
                 + pemfc_stack_id
                 + ":nominal_pressure",
            units="atm",
            val=np.full(number_of_points, 1.0),
        )

        self.add_input(
            "single_layer_pemfc_voltage",
            units="V",
            val=np.full(number_of_points, np.nan),
        )

        self.add_output(name="data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":efficiency", val=np.full(number_of_points, 0.4))

        self.declare_partials(
            of="*",
            wrt=["single_layer_pemfc_voltage","data:propulsion:he_power_train:pemfc_stack:"
                 + pemfc_stack_id
                 + ":nominal_pressure"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="*",
            wrt= "data:propulsion:he_power_train:pemfc_stack:"+ pemfc_stack_id
                + ":operation_pressure",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        E0 = 1.23  # ideal potential of the pemfc
        C = 0.06
        pemfc_stack_id = self.options["pemfc_stack_id"]
        Pop = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":operation_pressure"
        ]
        Pnom = inputs["data:propulsion:he_power_train:pemfc_stack:"
                 + pemfc_stack_id
                 + ":nominal_pressure"]
        E = E0 + C * np.log(Pop / Pnom)
        efficiency = inputs["single_layer_pemfc_voltage"] / E
        outputs["data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":efficiency"] = efficiency

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points = self.options["number_of_points"]
        E0 = 1.23  # ideal potential of the pemfc
        C = 0.06
        Pop = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":operation_pressure"
        ]
        Pnom = inputs["data:propulsion:he_power_train:pemfc_stack:"
                 + pemfc_stack_id
                 + ":nominal_pressure"]
        E = E0 + C * np.log(Pop / Pnom)

        partials["data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":efficiency", "single_layer_pemfc_voltage"] = np.ones(number_of_points) / E

        partials[
            "data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":efficiency",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":operation_pressure",
        ] = (-C * inputs["single_layer_pemfc_voltage"] / (Pop * E ** 2))
        partials["data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":efficiency","data:propulsion:he_power_train:pemfc_stack:"
                 + pemfc_stack_id
                 + ":nominal_pressure"] = (
            C * inputs["single_layer_pemfc_voltage"] / (Pnom * E ** 2)
        )
