# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


from ..components.perf_direct_bus_connection import PerformancesPEMFCDirectBusConnection
from ..components.perf_pemfc_power import PerformancesPEMFCPower
from ..components.perf_maximum import PerformancesMaximum
from ..components.perf_pemfc_current_density import PerformancesCurrentDensity
from ..components.perf_layer_voltage import PerformancesSinglePEMFCVoltage
from ..components.perf_fuel_consumption import PerformancesPEMFCFuelConsumption
from ..components.perf_fuel_consumed import PerformancesPEMFCFuelConsumed
from ..components.perf_pemfc_efficiency import PerformancesPEMFCEfficiency


class PerformancesPEMFCStack(om.Group):
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
        self.options.declare(
            name="direct_bus_connection",
            default=False,
            types=bool,
            desc="If the battery is directly connected to a bus, a special mode is required to "
            "interface the two",
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]
        direct_bus_connection = self.options["direct_bus_connection"]

        self.add_subsystem(
            "single_layer_voltage",
            PerformancesSinglePEMFCVoltage(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )

        if self.options["direct_bus_connection"]:
            self.add_subsystem(
                "direct_bus_connection",
                PerformancesPEMFCDirectBusConnection(number_of_points=number_of_points),
                promotes=["*"],
            )

        self.add_subsystem(
            "pemfc_current_density",
            PerformancesCurrentDensity(
                number_of_points=number_of_points, pemfc_stack_id=pemfc_stack_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "fuel_consumption",
            PerformancesPEMFCFuelConsumption(
                number_of_points=number_of_points, pemfc_stack_id=pemfc_stack_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "fuel_consumed",
            PerformancesPEMFCFuelConsumed(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "pemfc_efficiency",
            PerformancesPEMFCEfficiency(pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "pemfc_power",
            PerformancesPEMFCPower(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesMaximum(number_of_points=number_of_points, pemfc_stack_id=pemfc_stack_id),
            promotes=["*"],
        )

        energy_consumed = om.IndepVarComp()
        energy_consumed.add_output(
            "non_consumable_energy_t", np.full(number_of_points, 0.0), units="W*h"
        )
        self.add_subsystem(
            "energy_consumed",
            energy_consumed,
            promotes=["non_consumable_energy_t"],
        )

    # TODO: Check if this is required
    def guess_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]

        number_of_layers = inputs[
            "module_voltage.data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":module:number_of_layers"
        ]

        # Based on the max voltage and cut off voltage of the battery cell
        fake_layer_voltage = np.linspace(1.2, 0.1, number_of_points)
        stack_voltage = fake_layer_voltage * number_of_layers

        outputs["module_voltage"] = stack_voltage

        if self.options["direct_bus_connection"]:
            outputs["pemfc_voltage"] = stack_voltage
        else:
            outputs["voltage_out"] = stack_voltage
