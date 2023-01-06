"""
Test mission vector module.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import os.path as pth
import pytest

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from fastga_he.models.performances.mission_vector.initialization.initialize_altitude import (
    InitializeAltitude,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_climb_airspeed import (
    InitializeClimbAirspeed,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_descent_airspeed import (
    InitializeDescentAirspeed,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_reserve_airspeed import (
    InitializeReserveAirspeed,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_airspeed import (
    InitializeAirspeed,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_gamma import (
    InitializeGamma,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_airspeed_derivatives import (
    InitializeAirspeedDerivatives,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_time_and_distance import (
    InitializeTimeAndDistance,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_cg import InitializeCoG
from fastga_he.models.performances.mission_vector.mission_vector import MissionVector

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
XML_FILE = "sample_ac.xml"


def test_initialize_altitude():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeAltitude(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        InitializeAltitude(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    expected_altitude = np.array(
        [
            0.0,
            270.9,
            541.9,
            812.8,
            1083.7,
            1354.7,
            1625.6,
            1896.5,
            2167.5,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            1828.8,
            1219.2,
            609.6,
            0.0,
            1000.0,
            1000.0,
            1000.0,
            1000.0,
            1000.0,
        ]
    )
    assert np.max(np.abs(problem.get_val("altitude", units="m") - expected_altitude)) <= 1e-1

    problem.check_partials(compact_print=True)


def test_climb_speed():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeClimbAirspeed(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("mass", units="kg", val=np.full(30, 1700.0))

    problem = run_system(
        InitializeClimbAirspeed(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    climb_eas = problem.get_val("data:mission:sizing:main_route:climb:v_eas", units="m/s")
    assert climb_eas == pytest.approx(44.48, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_descent_speed():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeDescentAirspeed(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("mass", units="kg", val=np.full(30, 1700.0))
    ivc.add_output(
        "altitude",
        units="m",
        val=np.array(
            [
                0.0,
                270.9,
                541.9,
                812.8,
                1083.7,
                1354.7,
                1625.6,
                1896.5,
                2167.5,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                1828.8,
                1219.2,
                609.6,
                0.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ]
        ),
    )

    problem = run_system(
        InitializeDescentAirspeed(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    descent_eas = problem.get_val("data:mission:sizing:main_route:descent:v_eas", units="m/s")
    assert descent_eas == pytest.approx(56.27, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_reserve_speed():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeReserveAirspeed(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("mass", units="kg", val=np.full(30, 1700.0))

    problem = run_system(
        InitializeReserveAirspeed(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    reserve_tas = problem.get_val("data:mission:sizing:main_route:reserve:v_tas", units="m/s")
    assert reserve_tas == pytest.approx(46.69, rel=1e-3)

    problem.check_partials(compact_print=False)


def test_initialize_airspeed():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeAirspeed(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("mass", units="kg", val=np.full(30, 1700.0))
    ivc.add_output(
        "altitude",
        units="m",
        val=np.array(
            [
                0.0,
                270.9,
                541.9,
                812.8,
                1083.7,
                1354.7,
                1625.6,
                1896.5,
                2167.5,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                1828.8,
                1219.2,
                609.6,
                0.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ]
        ),
    )

    problem = run_system(
        InitializeAirspeed(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )
    expected_tas = np.array(
        [
            44.4,
            45.0,
            45.6,
            46.2,
            46.8,
            47.5,
            48.1,
            48.8,
            49.4,
            50.1,
            82.3,
            82.3,
            82.3,
            82.3,
            82.3,
            82.3,
            82.3,
            82.3,
            82.3,
            82.3,
            63.4,
            61.5,
            59.7,
            57.9,
            56.2,
            60.0,
            60.0,
            60.0,
            60.0,
            60.0,
        ]
    )
    expected_eas = np.array(
        [
            44.48,
            44.48,
            44.48,
            44.48,
            44.48,
            44.48,
            44.48,
            44.48,
            44.48,
            44.48,
            72.97,
            72.97,
            72.97,
            72.97,
            72.97,
            72.97,
            72.97,
            72.97,
            72.97,
            72.97,
            56.27,
            56.27,
            56.27,
            56.27,
            56.27,
            57.15,
            57.15,
            57.15,
            57.15,
            57.15,
        ]
    )
    assert np.max(np.abs(problem.get_val("true_airspeed", units="m/s") - expected_tas)) <= 1e-1
    assert (
        np.max(np.abs(problem.get_val("equivalent_airspeed", units="m/s") - expected_eas)) <= 1e-1
    )

    problem.check_partials(compact_print=True)


def test_initialize_airspeed_derivatives():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "altitude",
        units="m",
        val=np.array(
            [
                0.0,
                270.9,
                541.9,
                812.8,
                1083.7,
                1354.7,
                1625.6,
                1896.5,
                2167.5,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                1828.8,
                1219.2,
                609.6,
                0.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ]
        ),
    )
    ivc.add_output(
        "true_airspeed",
        units="m/s",
        val=np.array(
            [
                44.4,
                45.0,
                45.6,
                46.2,
                46.8,
                47.5,
                48.1,
                48.8,
                49.4,
                50.1,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                63.4,
                61.5,
                59.7,
                57.9,
                56.2,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ]
        ),
    )
    ivc.add_output(
        "equivalent_airspeed",
        units="m/s",
        val=np.array(
            [
                44.48,
                44.48,
                44.48,
                44.48,
                44.48,
                44.48,
                44.48,
                44.48,
                44.48,
                44.48,
                72.97,
                72.97,
                72.97,
                72.97,
                72.97,
                72.97,
                72.97,
                72.97,
                72.97,
                72.97,
                56.27,
                56.27,
                56.27,
                56.27,
                56.27,
                57.15,
                57.15,
                57.15,
                57.15,
                57.15,
            ]
        ),
    )
    ivc.add_output(
        "gamma",
        units="deg",
        val=np.array(
            [
                2.0,
                1.9,
                1.8,
                1.7,
                1.6,
                1.5,
                1.4,
                1.3,
                1.2,
                1.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )

    problem = run_system(
        InitializeAirspeedDerivatives(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )
    expected_derivative = np.array(
        [
            0.127,
            0.098,
            0.088,
            0.094,
            0.114,
            0.023,
            0.073,
            0.021,
            0.092,
            0.070,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.079,
            -0.0542,
            -0.0138,
            -0.0560,
            -0.0713,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    assert np.max(np.abs(problem.get_val("d_vx_dt", units="m/s**2") - expected_derivative)) <= 1e-1

    problem.check_partials(compact_print=True)


def test_initialize_gamma():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeGamma(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "altitude",
        units="m",
        val=np.array(
            [
                0.0,
                270.9,
                541.9,
                812.8,
                1083.7,
                1354.7,
                1625.6,
                1896.5,
                2167.5,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                1828.8,
                1219.2,
                609.6,
                0.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ]
        ),
    )
    ivc.add_output(
        "true_airspeed",
        units="m/s",
        val=np.array(
            [
                44.4,
                45.0,
                45.6,
                46.2,
                46.8,
                47.5,
                48.1,
                48.8,
                49.4,
                50.1,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                63.4,
                61.5,
                59.7,
                57.9,
                56.2,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ]
        ),
    )

    problem = run_system(
        InitializeGamma(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    expected_vertical_speed = np.array(
        [
            6.096,
            5.813,
            5.531,
            5.249,
            4.967,
            4.684,
            4.402,
            4.120,
            3.838,
            3.556,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.524,
            -1.524,
            -1.524,
            -1.524,
            -1.524,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    assert (
        np.max(np.abs(problem.get_val("vertical_speed", units="m/s") - expected_vertical_speed))
        <= 1e-1
    )
    expected_gamma = np.array(
        [
            0.1377,
            0.1295,
            0.1216,
            0.1138,
            0.1063,
            0.0987,
            0.0916,
            0.0845,
            0.0777,
            0.0710,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.024,
            -0.024,
            -0.025,
            -0.026,
            -0.027,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    assert np.max(np.abs(problem.get_val("gamma", units="rad") - expected_gamma)) <= 1e-1

    problem.check_partials(compact_print=True)


def test_initialize_time_and_position():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeTimeAndDistance(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "altitude",
        units="m",
        val=np.array(
            [
                0.0,
                270.9,
                541.9,
                812.8,
                1083.7,
                1354.7,
                1625.6,
                1896.5,
                2167.5,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                1828.8,
                1219.2,
                609.6,
                0.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ]
        ),
    )
    ivc.add_output(
        "true_airspeed",
        units="m/s",
        val=np.array(
            [
                44.4,
                45.0,
                45.6,
                46.2,
                46.8,
                47.5,
                48.1,
                48.8,
                49.4,
                50.1,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                63.4,
                61.5,
                59.7,
                57.9,
                56.2,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ]
        ),
    )
    ivc.add_output(
        "horizontal_speed",
        units="m/s",
        val=np.array(
            [
                43.9,
                44.6,
                45.2,
                45.9,
                46.5,
                47.2,
                47.8,
                48.6,
                49.2,
                49.9,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                63.3,
                61.4,
                59.6,
                57.8,
                56.1,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ]
        ),
    )

    problem = run_system(
        InitializeTimeAndDistance(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    expected_position = np.array(
        [
            0.0,
            1.08,
            2.24,
            3.48,
            4.80,
            6.22,
            7.75,
            9.408,
            11.20,
            13.16,
            83.65,
            154.13,
            224.61,
            295.10,
            365.58,
            436.06,
            506.55,
            577.03,
            647.52,
            718.00,
            788.48,
            801.95,
            815.02,
            827.69,
            840.0,
            857.4,
            874.9,
            892.4,
            909.9,
            927.4,
        ]
    )
    assert np.max(np.abs(problem.get_val("position", units="nmi") - expected_position)) <= 1e-1
    expected_time = np.array(
        [
            0.0,
            0.75,
            1.55,
            2.39,
            3.27,
            4.21,
            5.20,
            6.26,
            7.39,
            8.62,
            35.05,
            61.48,
            87.91,
            114.34,
            140.77,
            167.20,
            193.64,
            220.07,
            246.50,
            272.93,
            299.36,
            306.03,
            312.70,
            319.36,
            326.03,
            335.03,
            344.03,
            353.03,
            362.03,
            371.03,
        ]
    )
    assert np.max(np.abs(problem.get_val("time", units="min") - expected_time)) <= 1e-1

    problem.check_partials(compact_print=False)


def test_initialize_cog():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeCoG(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("fuel_consumed_t", units="kg", val=np.linspace(1.0, 2.0, 30))

    problem = run_system(
        InitializeCoG(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )
    expected_cog = np.array(
        [
            3.524,
            3.524,
            3.524,
            3.524,
            3.524,
            3.524,
            3.523,
            3.523,
            3.523,
            3.523,
            3.522,
            3.522,
            3.522,
            3.522,
            3.521,
            3.521,
            3.521,
            3.521,
            3.520,
            3.520,
            3.520,
            3.520,
            3.519,
            3.519,
            3.519,
            3.518,
            3.518,
            3.518,
            3.517,
            3.517,
        ]
    )
    assert np.max(np.abs(problem.get_val("x_cg", units="m") - expected_cog)) <= 1e-3

    problem.check_partials(compact_print=True)


def test_mission_vector():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            MissionVector(
                number_of_points_climb=100,
                number_of_points_cruise=100,
                number_of_points_descent=50,
                number_of_points_reserve=50,
            )
        ),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        MissionVector(
            number_of_points_climb=100,
            number_of_points_cruise=100,
            number_of_points_descent=50,
            number_of_points_reserve=50,
        ),
        ivc,
    )
    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(62.48, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(0.0, abs=1e-2)


def test_mission_vector_from_yml():

    # Define used files depending on options
    xml_file_name = "sample_ac.xml"
    process_file_name = "mission_vector.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()
    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(0.0, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(0.0, abs=1e-2)
