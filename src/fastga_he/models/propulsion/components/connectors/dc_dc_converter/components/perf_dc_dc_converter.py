# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_switching_frequency import PerformancesSwitchingFrequencyMission
from .perf_voltage_out_target import PerformancesVoltageOutTargetMission
from .perf_converter_relations import PerformancesConverterRelations
from .perf_generator_side import PerformancesConverterGeneratorSide
from .perf_load_side import PerformancesConverterLoadSide
from .perf_duty_cycle import PerformancesDutyCycle
from .perf_currents import PerformancesCurrents
from .perf_switching_losses import PerformancesSwitchingLosses
from .perf_conduction_losses import PerformancesConductionLosses
from .perf_total_losses import PerformancesLosses
from .perf_efficiency import PerformancesEfficiency
from .perf_maximum import PerformancesMaximum


class PerformancesDCDCConverter(om.Group):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        self.add_subsystem(
            "switching_frequency",
            PerformancesSwitchingFrequencyMission(
                number_of_points=number_of_points, dc_dc_converter_id=dc_dc_converter_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "voltage_out_target",
            PerformancesVoltageOutTargetMission(
                number_of_points=number_of_points, dc_dc_converter_id=dc_dc_converter_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "load_side",
            PerformancesConverterLoadSide(number_of_points=number_of_points),
            promotes=["dc_voltage_in", "dc_current_in"],
        )
        self.add_subsystem(
            "generator_side",
            PerformancesConverterGeneratorSide(number_of_points=number_of_points),
            promotes=["dc_voltage_out", "dc_current_out"],
        )
        self.add_subsystem(
            "duty_cycle",
            PerformancesDutyCycle(number_of_points=number_of_points),
            promotes=["dc_voltage_in", "dc_voltage_out"],
        )
        self.add_subsystem(
            "currents",
            PerformancesCurrents(number_of_points=number_of_points),
            promotes=["dc_current_out"],
        )
        self.add_subsystem(
            "switching_losses",
            PerformancesSwitchingLosses(
                number_of_points=number_of_points, dc_dc_converter_id=dc_dc_converter_id
            ),
            promotes=["data:*", "switching_frequency"],
        )
        self.add_subsystem(
            "conduction_losses",
            PerformancesConductionLosses(
                number_of_points=number_of_points, dc_dc_converter_id=dc_dc_converter_id
            ),
            promotes=["data:*", "dc_current_out"],
        )
        self.add_subsystem(
            "total_losses",
            PerformancesLosses(number_of_points=number_of_points),
            promotes=[],
        )
        self.add_subsystem(
            "efficiency",
            PerformancesEfficiency(number_of_points=number_of_points),
            promotes=["dc_current_out", "dc_voltage_out"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesMaximum(
                number_of_points=number_of_points, dc_dc_converter_id=dc_dc_converter_id
            ),
            promotes=[
                "data:*",
                "dc_voltage_in",
                "dc_voltage_out",
                "dc_current_in",
                "switching_frequency",
            ],
        )
        self.add_subsystem(
            "converter_relation",
            PerformancesConverterRelations(
                number_of_points=number_of_points,
            ),
            promotes=["dc_voltage_out", "voltage_out_target"],
        )

        self.connect("converter_relation.power_rel", "load_side.power")
        self.connect("dc_current_out", "converter_relation.dc_current_out")
        self.connect("converter_relation.voltage_out_rel", "generator_side.voltage_target")
        self.connect("duty_cycle.duty_cycle", "currents.duty_cycle")
        self.connect(
            "currents.current_IGBT",
            [
                "switching_losses.current_IGBT",
                "conduction_losses.current_IGBT",
                "maximum.current_IGBT",
            ],
        )
        self.connect(
            "currents.current_diode",
            [
                "switching_losses.current_diode",
                "conduction_losses.current_diode",
                "maximum.current_diode",
            ],
        )
        self.connect(
            "currents.current_inductor",
            ["conduction_losses.current_inductor", "maximum.current_inductor"],
        )
        self.connect(
            "currents.current_capacitor",
            ["conduction_losses.current_capacitor", "maximum.current_capacitor"],
        )
        self.connect(
            "switching_losses.switching_losses_diode", "total_losses.switching_losses_diode"
        )
        self.connect("switching_losses.switching_losses_IGBT", "total_losses.switching_losses_IGBT")
        self.connect(
            "conduction_losses.conduction_losses_diode", "total_losses.conduction_losses_diode"
        )
        self.connect(
            "conduction_losses.conduction_losses_IGBT", "total_losses.conduction_losses_IGBT"
        )
        self.connect(
            "conduction_losses.conduction_losses_inductor",
            "total_losses.conduction_losses_inductor",
        )
        self.connect(
            "conduction_losses.conduction_losses_capacitor",
            "total_losses.conduction_losses_capacitor",
        )
        self.connect(
            "total_losses.losses_converter",
            ["efficiency.losses_converter", "maximum.losses_converter"],
        )
        self.connect("efficiency.efficiency", "converter_relation.efficiency")
