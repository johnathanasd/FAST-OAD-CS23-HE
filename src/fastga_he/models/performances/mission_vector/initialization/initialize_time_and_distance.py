# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om


class InitializeTimeAndDistance(om.ExplicitComponent):
    """Initializes time and ground distance at each time step."""

    def initialize(self):

        self.options.declare(
            "number_of_points_climb", default=1, desc="number of equilibrium to be treated in climb"
        )
        self.options.declare(
            "number_of_points_cruise",
            default=1,
            desc="number of equilibrium to be treated in " "cruise",
        )
        self.options.declare(
            "number_of_points_descent",
            default=1,
            desc="number of equilibrium to be treated in descent",
        )

    def setup(self):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]

        number_of_points = (
            number_of_points_climb + number_of_points_cruise + number_of_points_descent
        )

        # Cannot use the vertical speed vector previously computed since it is gonna be
        # initialized at 0.0 which will cause a problem for the time computation
        self.add_input("data:TLAR:range", np.nan, units="m")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input(
            "data:mission:sizing:main_route:climb:climb_rate:sea_level", val=np.nan, units="m/s"
        )
        self.add_input(
            "data:mission:sizing:main_route:climb:climb_rate:cruise_level", val=np.nan, units="m/s"
        )
        self.add_input("data:mission:sizing:main_route:descent:descent_rate", np.nan, units="m/s")

        self.add_input(
            "true_airspeed",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="m/s",
        )
        self.add_input(
            "horizontal_speed",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="m/s",
        )
        self.add_input(
            "altitude", val=np.full(number_of_points, np.nan), shape=number_of_points, units="m"
        )

        self.add_output("time", val=np.linspace(0.0, 7200.0, number_of_points), units="s")
        self.add_output("position", val=np.linspace(0.0, 926000.0, number_of_points), units="m")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]

        altitude = inputs["altitude"]
        horizontal_speed = inputs["horizontal_speed"]

        mission_range = inputs["data:TLAR:range"]
        v_tas_cruise = inputs["data:TLAR:v_cruise"]

        climb_rate_sl = float(inputs["data:mission:sizing:main_route:climb:climb_rate:sea_level"])
        climb_rate_cl = float(
            inputs["data:mission:sizing:main_route:climb:climb_rate:cruise_level"]
        )
        descent_rate = -abs(inputs["data:mission:sizing:main_route:descent:descent_rate"])

        altitude_climb = altitude[0:number_of_points_climb]
        horizontal_speed_climb = horizontal_speed[0:number_of_points_climb]
        altitude_descent = altitude[number_of_points_climb + number_of_points_cruise :]
        horizontal_speed_descent = horizontal_speed[
            number_of_points_climb + number_of_points_cruise :
        ]

        # Computing the time evolution during the climb phase, based on the altitude sampling and
        # the desired climb rate
        mid_altitude_climb = (altitude_climb[:-1] + altitude_climb[1:]) / 2.0
        mid_climb_rate = np.interp(
            mid_altitude_climb, [0.0, max(altitude_climb)], [climb_rate_sl, climb_rate_cl]
        )
        mid_horizontal_speed_climb = (
            horizontal_speed_climb[:-1] + horizontal_speed_climb[1:]
        ) / 2.0
        altitude_step_climb = altitude_climb[1:] - altitude_climb[:-1]
        time_to_climb_step = altitude_step_climb / mid_climb_rate
        position_increment_climb = mid_horizontal_speed_climb * time_to_climb_step

        time_climb = np.concatenate((np.array([0]), np.cumsum(time_to_climb_step)))
        position_climb = np.concatenate((np.array([0]), np.cumsum(position_increment_climb)))

        # Computing the time evolution during the descent phase, based on the altitude sampling and
        # the desired descent rate
        mid_descent_rate = np.full_like(altitude_descent[1:], abs(descent_rate))
        mid_horizontal_speed_descent = (
            horizontal_speed_descent[:-1] + horizontal_speed_descent[1:]
        ) / 2.0
        altitude_step_descent = abs(altitude_descent[1:] - altitude_descent[:-1])
        time_to_descend_step = altitude_step_descent / mid_descent_rate
        position_increment_descent = mid_horizontal_speed_descent * time_to_descend_step

        time_descent = np.concatenate((np.array([0]), np.cumsum(time_to_descend_step)))
        position_descent = np.concatenate((np.array([0]), np.cumsum(position_increment_descent)))

        # Cruise position computation
        cruise_range = mission_range - position_climb[-1] - position_descent[-1]
        cruise_distance_step = cruise_range / (number_of_points_cruise + 1)
        position_cruise = np.linspace(
            position_climb[-1] + cruise_distance_step,
            position_climb[-1] + cruise_range - cruise_distance_step,
            number_of_points_cruise,
        )[:, 0]

        cruise_time_array = (position_cruise - position_climb[-1]) / v_tas_cruise + time_climb[-1]
        cruise_time = cruise_range / v_tas_cruise

        position_descent += position_climb[-1] + cruise_range
        time_descent += time_climb[-1] + cruise_time

        position = np.concatenate((position_climb, position_cruise, position_descent))
        time = np.concatenate((time_climb, cruise_time_array, time_descent))

        outputs["position"] = position
        outputs["time"] = time
