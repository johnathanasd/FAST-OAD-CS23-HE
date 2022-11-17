# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_modulation_index import PerformancesModulationIndex
from .perf_resistance import PerformancesResistance
from .perf_conduction_loss import PerformancesConductionLosses
from .perf_switching_losses import PerformancesSwitchingLosses
from .perf_total_loss import PerformancesLosses
from .perf_temperature import PerformancesTemperature
from .perf_efficiency import PerformancesEfficiency
from .perf_dc_current import PerformancesDCCurrent


class PerformancesInverter(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 200
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.linear_solver = om.DirectSolver()

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "modulation_idx",
            PerformancesModulationIndex(number_of_points=number_of_points),
            promotes=["ac_voltage_peak_out", "dc_voltage_in"],
        )
        self.add_subsystem(
            "resistance",
            PerformancesResistance(inverter_id=inverter_id, number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "conduction_losses",
            PerformancesConductionLosses(
                inverter_id=inverter_id, number_of_points=number_of_points
            ),
            promotes=["data:*", "ac_current_rms_out_one_phase"],
        )
        self.add_subsystem(
            "switching_losses",
            PerformancesSwitchingLosses(inverter_id=inverter_id, number_of_points=number_of_points),
            promotes=["data:*", "ac_current_rms_out_one_phase", "switching_frequency"],
        )
        self.add_subsystem(
            "total_losses",
            PerformancesLosses(number_of_points=number_of_points),
            promotes=[],
        )
        self.add_subsystem(
            "temperature_inverter",
            PerformancesTemperature(inverter_id=inverter_id, number_of_points=number_of_points),
            promotes=["data:*", "heat_sink_temperature"],
        )
        self.add_subsystem(
            "efficiency",
            PerformancesEfficiency(number_of_points=number_of_points),
            promotes=["ac_current_rms_out_one_phase", "ac_voltage_rms_out"],
        )
        self.add_subsystem(
            "dc_side_current",
            PerformancesDCCurrent(number_of_points=number_of_points),
            promotes=[
                "ac_current_rms_out_one_phase",
                "dc_voltage_in",
                "ac_voltage_rms_out",
                "dc_current_in",
            ],
        )

        self.connect("modulation_idx.modulation_index", "conduction_losses.modulation_index")
        self.connect("resistance.resistance_igbt", "conduction_losses.resistance_igbt")
        self.connect("resistance.resistance_diode", "conduction_losses.resistance_diode")
        self.connect(
            "conduction_losses.conduction_losses_diode", "total_losses.conduction_losses_diode"
        )
        self.connect(
            "conduction_losses.conduction_losses_IGBT", "total_losses.conduction_losses_IGBT"
        )
        self.connect(
            "switching_losses.switching_losses_diode", "total_losses.switching_losses_diode"
        )
        self.connect("switching_losses.switching_losses_IGBT", "total_losses.switching_losses_IGBT")
        self.connect(
            "total_losses.losses_inverter",
            ["temperature_inverter.losses_inverter", "efficiency.losses_inverter"],
        )
        self.connect("temperature_inverter.inverter_temperature", "resistance.inverter_temperature")
        self.connect("efficiency.efficiency", "dc_side_current.efficiency")
