# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

MAXIMUM_VOLTAGE_PER_LAYER = 1.2  # voltage


class PerformancesSinglePEMFCVoltage(om.ExplicitComponent):
    """
    Computation of the voltage of single layer proton exchange membrane fuel cell inside one stack. Assumes it can be
    estimated with the i-v curve relation. Model based on existing pemfc, Aerostack Ultralight 200, details can be found in:
    cite:`Fuel Cell and Battery Hybrid System Optimization by J. Hoogendoorn:2018`.
    """

    # TODO: Integrate model based on Thermodynamic calculation from D.Juschus
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

        self.add_input(
            "fc_current_density",
            units="A/cm**2",
            val=np.full(number_of_points, np.nan),
        )

        self.add_input(
            name="nominal_pressure",
            units="atm",
            val=1.0,
        )

        self.add_input(
            "operation_pressure",
            units="atm",
            val=np.full(number_of_points, 1.0),
        )

        self.add_output(
            "single_layer_pemfc_voltage",
            units="V",
            val=np.full(number_of_points, MAXIMUM_VOLTAGE_PER_LAYER),
        )

        self.declare_partials(
            of="*",
            wrt=["fc_current_density", "operation_pressure"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="*",
            wrt="nominal_pressure",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        V0 = 0.83
        B = 0.014
        R = 0.24
        m = 5.63 * 10 ** -6
        n = 11.42
        C = 0.06
        i = inputs["fc_current_density"]
        operation_pressure = inputs["operation_pressure"]
        nominal_pressure = inputs["nominal_pressure"]
        outputs["single_layer_pemfc_voltage"] = (
            V0
            - B * np.log(i)
            - R * i
            - m * np.exp(n * i)
            + C * np.log(operation_pressure / nominal_pressure)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        number_of_points = self.options["number_of_points"]
        B = 0.014
        R = 0.24
        m = 5.63 * 10 ** -6
        n = 11.42
        C = 0.06
        i = inputs["fc_current_density"]
        partials["single_layer_pemfc_voltage", "fc_current_density"] = (
            -B / i - R - m * n * np.exp(n * i)
        )
        partials["single_layer_pemfc_voltage", "operation_pressure"] = (
            C / inputs["operation_pressure"]
        )
        partials["single_layer_pemfc_voltage", "nominal_pressure"] = (
            -C * np.ones(number_of_points) / inputs["nominal_pressure"]
        )
