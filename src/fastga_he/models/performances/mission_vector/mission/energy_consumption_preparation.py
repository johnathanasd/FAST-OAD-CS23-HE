# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om
from stdatm import Atmosphere


class PrepareForEnergyConsumption(om.ExplicitComponent):
    """
    Prepare the different vector for the energy consumption computation, which means some name
    will be changed because we need to add the point corresponding to the taxi computation.
    """

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
            desc="number of equilibrium to be treated in " "descen",
        )
        self.options.declare(
            "number_of_points_reserve",
            default=1,
            desc="number of equilibrium to be treated in reserve",
        )

    def setup(self):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        number_of_points = (
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        )

        self.add_input("data:mission:sizing:taxi_out:speed", np.nan, units="m/s")
        self.add_input("data:mission:sizing:taxi_out:duration", np.nan, units="s")
        self.add_input("data:mission:sizing:taxi_out:thrust", 1500, units="N")

        self.add_input("data:mission:sizing:taxi_in:speed", np.nan, units="m/s")
        self.add_input("data:mission:sizing:taxi_in:duration", np.nan, units="s")
        self.add_input("data:mission:sizing:taxi_in:thrust", 1500, units="N")

        self.add_input(
            "thrust", shape=number_of_points, val=np.full(number_of_points, np.nan), units="N"
        )
        self.add_input(
            "altitude", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m"
        )
        self.add_input(
            "exterior_temperature",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="degK",
        )
        self.add_input(
            "time_step", shape=number_of_points, val=np.full(number_of_points, np.nan), units="s"
        )
        self.add_input(
            "true_airspeed",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="m/s",
        )
        self.add_input(
            "engine_setting", shape=number_of_points, val=np.full(number_of_points, np.nan)
        )

        # Econ stands for Energy Consumption, this way we separate the vectors used for the
        # computation of the equilibrium from the one used for the computation of the energy
        # consumption
        self.add_output("thrust_econ", shape=number_of_points + 2, units="N")
        self.add_output("altitude_econ", shape=number_of_points + 2, units="m")
        self.add_output("exterior_temperature_econ", shape=number_of_points + 2, units="degK")
        self.add_output("time_step_econ", shape=number_of_points + 2, units="s")
        self.add_output("true_airspeed_econ", shape=number_of_points + 2, units="m/s")
        self.add_output("engine_setting_econ", shape=number_of_points + 2)

    def setup_partials(self):

        self.declare_partials(
            of="thrust_econ",
            wrt=[
                "thrust",
                "data:mission:sizing:taxi_out:thrust",
                "data:mission:sizing:taxi_in:thrust",
            ],
            method="exact",
        )

        self.declare_partials(
            of="altitude_econ",
            wrt=[
                "altitude",
            ],
            method="exact",
        )

        self.declare_partials(
            of="exterior_temperature_econ",
            wrt=[
                "exterior_temperature",
            ],
            method="exact",
        )

        self.declare_partials(
            of="time_step_econ",
            wrt=[
                "time_step",
                "data:mission:sizing:taxi_out:duration",
                "data:mission:sizing:taxi_in:duration",
            ],
            method="exact",
        )

        self.declare_partials(
            of="true_airspeed_econ",
            wrt=[
                "true_airspeed",
                "data:mission:sizing:taxi_out:speed",
                "data:mission:sizing:taxi_in:speed",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        number_of_points = (
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        )

        thrust_taxi_out = float(inputs["data:mission:sizing:taxi_out:thrust"])
        thrust_taxi_in = float(inputs["data:mission:sizing:taxi_in:thrust"])
        outputs["thrust_econ"] = np.concatenate(
            (np.array([thrust_taxi_out]), inputs["thrust"], np.array([thrust_taxi_in]))
        )

        outputs["altitude_econ"] = np.concatenate((np.zeros(1), inputs["altitude"], np.zeros(1)))

        temp_sl = Atmosphere(np.array([0]), altitude_in_feet=True).temperature
        outputs["exterior_temperature_econ"] = np.concatenate(
            (temp_sl, inputs["exterior_temperature"], temp_sl)
        )

        time_step_taxi_out = float(inputs["data:mission:sizing:taxi_out:duration"])
        time_step_taxi_in = float(inputs["data:mission:sizing:taxi_in:duration"])
        # Here we have to do an additional change. Since time step is computed for point i based
        # on time(i+1) - time(i) the last time step of climb will be computed with the first time
        # of cruise which means, since the cruise time step is very wide, that it will be very
        # wide and lead to an overestimation of climb fuel. For this reason we will replace the
        # last time step of climb with the precedent to get a good estimate. This will only serve
        # for the energy consumption calculation. Since this module might be used for something
        # else than performances computation, the array might not be long enough for the index
        # number_of_points_climb - 1 to be reachable. Though it is a clumsy way to do it,
        # we will check that n = number_of_points_climb + number_of_points_cruise +
        # number_of_points_descent and only perform the change above if it is the case.
        time_step = inputs["time_step"]
        if (
            number_of_points
            == number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        ):
            time_step[number_of_points_climb - 1] = time_step[number_of_points_climb - 2]
        outputs["time_step_econ"] = np.concatenate(
            (np.array([time_step_taxi_out]), time_step, np.array([time_step_taxi_in]))
        )

        tas_taxi_out = float(inputs["data:mission:sizing:taxi_out:speed"])
        tas_taxi_in = float(inputs["data:mission:sizing:taxi_in:speed"])
        outputs["true_airspeed_econ"] = np.concatenate(
            (np.array([tas_taxi_out]), inputs["true_airspeed"], np.array([tas_taxi_in]))
        )

        outputs["engine_setting_econ"] = np.concatenate(
            (np.ones(1), inputs["engine_setting"], np.ones(1))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        number_of_points = (
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        )

        d_thrust_econ_d_thrust = np.zeros((number_of_points + 2, number_of_points))
        d_thrust_econ_d_thrust[1 : number_of_points + 1, :] = np.eye(number_of_points)
        partials["thrust_econ", "thrust"] = d_thrust_econ_d_thrust

        d_thrust_econ_d_thrust_to = np.zeros(number_of_points + 2)
        d_thrust_econ_d_thrust_to[0] = 1.0
        partials["thrust_econ", "data:mission:sizing:taxi_out:thrust"] = d_thrust_econ_d_thrust_to

        d_thrust_econ_d_thrust_ti = np.zeros(number_of_points + 2)
        d_thrust_econ_d_thrust_ti[-1] = 1.0
        partials["thrust_econ", "data:mission:sizing:taxi_in:thrust"] = d_thrust_econ_d_thrust_ti

        d_altitude_econ_d_altitude = np.zeros((number_of_points + 2, number_of_points))
        d_altitude_econ_d_altitude[1 : number_of_points + 1, :] = np.eye(number_of_points)
        partials["altitude_econ", "altitude"] = d_altitude_econ_d_altitude

        d_temp_econ_d_temp = np.zeros((number_of_points + 2, number_of_points))
        d_temp_econ_d_temp[1 : number_of_points + 1, :] = np.eye(number_of_points)
        partials["exterior_temperature_econ", "exterior_temperature"] = d_temp_econ_d_temp

        d_ts_econ_d_ts = np.zeros((number_of_points + 2, number_of_points))
        d_ts_econ_d_ts[1 : number_of_points + 1, :] = np.eye(number_of_points)
        if (
            number_of_points
            == number_of_points_climb + number_of_points_cruise + number_of_points_descent
        ):
            d_ts_econ_d_ts[number_of_points_climb - 1, number_of_points_climb - 1] = 0.0
            d_ts_econ_d_ts[number_of_points_climb - 1, number_of_points_climb - 2] = 0.0
        partials["time_step_econ", "time_step"] = d_ts_econ_d_ts

        d_ts_econ_d_ts_to = np.zeros(number_of_points + 2)
        d_ts_econ_d_ts_to[0] = 1.0
        partials["time_step_econ", "data:mission:sizing:taxi_out:duration"] = d_ts_econ_d_ts_to

        d_ts_econ_d_ts_ti = np.zeros(number_of_points + 2)
        d_ts_econ_d_ts_ti[-1] = 1.0
        partials["time_step_econ", "data:mission:sizing:taxi_in:duration"] = d_ts_econ_d_ts_ti

        d_tas_econ_d_tas = np.zeros((number_of_points + 2, number_of_points))
        d_tas_econ_d_tas[1 : number_of_points + 1, :] = np.eye(number_of_points)
        partials["true_airspeed_econ", "true_airspeed"] = d_tas_econ_d_tas

        d_tas_econ_d_tas_to = np.zeros(number_of_points + 2)
        d_tas_econ_d_tas_to[0] = 1.0
        partials["true_airspeed_econ", "data:mission:sizing:taxi_out:speed"] = d_tas_econ_d_tas_to

        d_tas_econ_d_tas_ti = np.zeros(number_of_points + 2)
        d_tas_econ_d_tas_ti[-1] = 1.0
        partials["true_airspeed_econ", "data:mission:sizing:taxi_in:speed"] = d_tas_econ_d_tas_ti
