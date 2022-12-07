# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import openmdao.api as om
import fastoad.api as oad

from ..constants import HE_SUBMODEL_EQUILIBRIUM
from ..mission.compute_time_step import ComputeTimeStep
from ..mission.performance_per_phase import PerformancePerPhase
from ..mission.reserve_energy import ReserveEnergy
from ..mission.sizing_energy import SizingEnergy
from ..mission.thrust_taxi import ThrustTaxi
from ..mission.update_mass import UpdateMass


class MissionCore(om.Group):
    """Find the conditions necessary for the aircraft equilibrium."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "compute_taxi_thrust",
            ThrustTaxi(),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_time_step",
            ComputeTimeStep(number_of_points=number_of_points),
            promotes_inputs=[],
            promotes_outputs=[],
        )
        options_equilibrium = {
            "number_of_points": number_of_points,
        }
        self.add_subsystem(
            "compute_dep_equilibrium",
            oad.RegisterSubmodel.get_submodel(HE_SUBMODEL_EQUILIBRIUM, options=options_equilibrium),
            promotes_inputs=["data:*"],
            promotes_outputs=[],
        )
        self.add_subsystem(
            "performance_per_phase",
            PerformancePerPhase(number_of_points=number_of_points),
            promotes_inputs=[],
            promotes_outputs=["data:*"],
        )
        self.add_subsystem("reserve_fuel", ReserveEnergy(), promotes=["*"])
        self.add_subsystem("sizing_fuel", SizingEnergy(), promotes=["*"])
        self.add_subsystem(
            "update_mass",
            UpdateMass(number_of_points=number_of_points),
            promotes_inputs=["data:*"],
            promotes_outputs=[],
        )

        self.connect(
            "compute_dep_equilibrium.compute_energy_consumed.fuel_consumed_t_econ",
            "performance_per_phase.fuel_consumed_t_econ",
        )

        self.connect(
            "compute_dep_equilibrium.compute_energy_consumed.non_consumable_energy_t_econ",
            "performance_per_phase.non_consumable_energy_t_econ",
        )

        self.connect(
            "compute_dep_equilibrium.compute_energy_consumed.thrust_rate_t_econ",
            "performance_per_phase.thrust_rate_t_econ",
        )

        self.connect("update_mass.mass", "compute_dep_equilibrium.compute_equilibrium.mass")

        self.connect("performance_per_phase.fuel_consumed_t", "update_mass.fuel_consumed_t")

        self.connect(
            "compute_time_step.time_step",
            "compute_dep_equilibrium.preparation_for_energy_consumption.time_step",
        )
