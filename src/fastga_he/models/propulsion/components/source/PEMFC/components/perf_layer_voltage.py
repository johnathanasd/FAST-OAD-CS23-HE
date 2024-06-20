# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


DEFAULT_MAX_CURRENT_DENSITY = 0.7


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

        self.options.declare(
            "open_circuit_voltage",
            default=0.83,
            desc="open_circuit_voltage of one layer of pemfc [V]",
        )

        self.options.declare(
            "activation_loss_coefficient",
            default=0.014,
            desc="activation loss coefficient of one layer of pemfc (V/ln(A/cm**2))",
        )

        self.options.declare(
            "ohmic_resistance",
            default=0.24,
            desc="ohmic resistance of one layer of pemfc [V/ln(A/cm**2)]",
        )

        self.options.declare(
            "coefficient_in_concentration_loss",
            default=5.63 * 10 ** -6,
            desc="coefficient in concentration loss of one layer of pemfc [V]",
        )

        self.options.declare(
            "exponential_coefficient_in_concentration_loss",
            default=11.42,
            desc="exponential coefficient in concentration loss of one layer of pemfc [cm**2/A]",
        )

        self.options.declare(
            "pressure_coefficient", default=0.06, desc="pressure coefficient of one layer of pemfc"
        )

        self.options.declare(
            "max_current_density", default=0.7, desc="maximum current density  of pemfc"
        )

    def setup(self):

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
            val=np.full(number_of_points, DEFAULT_MAX_CURRENT_DENSITY),
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

        voc = self.options["open_circuit_voltage"]
        active_loss_coeff = self.options["activation_loss_coefficient"]
        r = self.options["ohmic_resistance"]
        m = self.options["coefficient_in_concentration_loss"]
        n = self.options["exponential_coefficient_in_concentration_loss"]
        pressure_coeff = self.options["pressure_coefficient"]

        i = np.clip(
            inputs["fc_current_density"],
            np.full_like(inputs["fc_current_density"], 1e-2),
            np.full_like(inputs["fc_current_density"], self.options["max_current_density"]),
        )

        operation_pressure = inputs["operation_pressure"]

        nominal_pressure = inputs["nominal_pressure"]

        outputs["single_layer_pemfc_voltage"] = (
            voc
            - active_loss_coeff * np.log(i)
            - r * i
            - m * np.exp(n * i)
            + pressure_coeff * np.log(operation_pressure / nominal_pressure)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]
        active_loss_coeff = self.options["activation_loss_coefficient"]
        r = self.options["ohmic_resistance"]
        m = self.options["coefficient_in_concentration_loss"]
        n = self.options["exponential_coefficient_in_concentration_loss"]
        pressure_coeff = self.options["pressure_coefficient"]

        i = np.clip(
            inputs["fc_current_density"],
            np.full_like(inputs["fc_current_density"], 1e-2),
            np.full_like(inputs["fc_current_density"], self.options["max_current_density"]),
        )

        partials_j = np.where(
            inputs["fc_current_density"] == i,
            -active_loss_coeff / i - r - m * n * np.exp(n * i),
            1e-6,
        )

        partials["single_layer_pemfc_voltage", "fc_current_density"] = partials_j

        partials["single_layer_pemfc_voltage", "operation_pressure"] = (
            pressure_coeff / inputs["operation_pressure"]
        )

        partials["single_layer_pemfc_voltage", "nominal_pressure"] = (
            -pressure_coeff * np.ones(number_of_points) / inputs["nominal_pressure"]
        )
