# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
import copy

import openmdao.api as om
import pytest
import numpy as np


from ..components.sizing_pemfc_weight import SizingPEMFCWeight
from ..components.sizing_pemfc_cg_x import SizingPEMFCCGX
from ..components.sizing_pemfc_cg_y import SizingPEMFCCGY
from ..components.sizing_pemfc_volume import SizingPEMFCVolume
from ..components.sizing_pemfc_dimensions import SizingPEMFCDimensions
from ..components.sizing_pemfc_drag import SizingPEMFCDrag

from ..components.perf_direct_bus_connection import PerformancesPEMFCDirectBusConnection
from ..components.perf_fuel_consumption import PerformancesPEMFCFuelConsumption
from ..components.perf_fuel_consumed import PerformancesPEMFCFuelConsumed
from ..components.perf_layer_voltage import PerformancesSinglePEMFCVoltage
from ..components.perf_pemfc_current_density import PerformancesCurrentDensity
from ..components.perf_maximum import PerformancesMaximum
from ..components.perf_pemfc_efficiency import PerformancesPEMFCEfficiency
from ..components.perf_pemfc_power import PerformancesPEMFCPower
from ..components.perf_pemfc_voltage import PerformancesPEMFCVoltage

from ..components.cstr_ensure import ConstraintsEffectiveAreaEnsure
from ..components.cstr_enforce import ConstraintsEffectiveAreaEnforce

from ..components.sizing_pemfc_stack import SizingPEMFCStack
from ..components.perf_pemfc_stack import PerformancesPEMFCStack

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_pemfc_stack.xml"
NB_POINTS_TEST = 10


def test_pemfc_weight():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingPEMFCWeight(pemfc_stack_id="pemfc_stack_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingPEMFCWeight(pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:module:mass", units="kg"
    ) == pytest.approx(15.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_pemfc_volume():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingPEMFCVolume(pemfc_stack_id="pemfc_stack_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingPEMFCVolume(pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:volume", units="L"
    ) == pytest.approx(660.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_pemfc_dimensions():

    expected_length = [0.511, 7.03, 0.81, 0.81, 1.48]
    expected_width = [3.69, 0.47, 1.67, 1.67, 0.95]
    expected_height = [0.82, 0.47, 1.14, 1.14, 1.09]

    for option, length, width, height in zip(
        POSSIBLE_POSITION, expected_length, expected_width, expected_height
    ):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingPEMFCDimensions(pemfc_stack_id="pemfc_stack_1", position=option)),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingPEMFCDimensions(pemfc_stack_id="pemfc_stack_1", position=option),
            ivc,
        )
        assert problem.get_val(
            "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:dimension:length", units="m"
        ) == pytest.approx(length, rel=1e-2)
        assert problem.get_val(
            "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:dimension:width", units="m"
        ) == pytest.approx(width, rel=1e-2)
        assert problem.get_val(
            "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:dimension:height", units="m"
        ) == pytest.approx(height, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_pemfc_cg_x():

    expected_values = [2.88, 2.88, 0.095, 2.38, 1.24]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingPEMFCCGX(pemfc_stack_id="pemfc_stack_1", position=option)),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingPEMFCCGX(pemfc_stack_id="pemfc_stack_1", position=option),
            ivc,
        )
        assert problem.get_val(
            "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_pemfc_cg_y():

    expected_values = [1.57, 1.57, 0.0, 0.0, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingPEMFCCGY(pemfc_stack_id="pemfc_stack_1", position=option)),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingPEMFCCGY(pemfc_stack_id="pemfc_stack_1", position=option),
            ivc,
        )
        assert problem.get_val(
            "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:CG:y", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_pemfc_drag():

    expected_ls_drag = [0.0, 0.021, 0.0, 0.0, 3.77e-3]
    expected_cruise_drag = [0.0, 0.021, 0.0, 0.0, 3.72e-3]

    for option, ls_drag, cruise_drag in zip(
        POSSIBLE_POSITION, expected_ls_drag, expected_cruise_drag
    ):
        # Research independent input value in .xml file
        for ls_option in [True, False]:
            ivc = get_indep_var_comp(
                list_inputs(
                    SizingPEMFCDrag(
                        pemfc_stack_id="pemfc_stack_1", position=option, low_speed_aero=ls_option
                    )
                ),
                __file__,
                XML_FILE,
            )

            # Run problem and check obtained value(s) is/(are) correct
            problem = run_system(
                SizingPEMFCDrag(
                    pemfc_stack_id="pemfc_stack_1", position=option, low_speed_aero=ls_option
                ),
                ivc,
            )

            if ls_option:
                assert (
                    problem.get_val(
                        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:low_speed:CD0",
                    )
                    == pytest.approx(ls_drag, rel=1e-2)
                )
            else:
                assert (
                    problem.get_val(
                        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:cruise:CD0",
                    )
                    == pytest.approx(cruise_drag, rel=1e-2)
                )

            problem.check_partials(compact_print=True)


def test_constraints_enforce_effective_area():

    inputs_list = list_inputs(ConstraintsEffectiveAreaEnforce(pemfc_stack_id="pemfc_stack_1"))
    inputs_list.remove("data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:effective_area")

    # Research independent input value in .xml file
    ivc_base = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    ivc_capacity_con = copy.deepcopy(ivc_base)
    ivc_capacity_con.add_output(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:effective_area",
        val=1.25,
        units="cm**2",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsEffectiveAreaEnforce(pemfc_stack_id="pemfc_stack_1"),
        ivc_capacity_con,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:number_modules",
        )
        == pytest.approx(42.66, rel=1e-2)
    )

    # The error on the capacity multiplier not retained is due to the fact that I've had bad
    # experiences with putting 0 in partials do I take something close enough to 0
    problem.check_partials(compact_print=True)

    # We try with a moderate c_rate to check if it works when remaining in the limiter range
    ivc_c_rate_con = copy.deepcopy(ivc_base)
    ivc_c_rate_con.add_output(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:c_rate_max",
        val=3.0,
        units="h**-1",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsEffectiveAreaEnforce(pemfc_stack_id="pemfc_stack_1"),
        ivc_c_rate_con,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:number_modules",
        )
        == pytest.approx(66.67, rel=1e-2)
    )

    # Partials will be hard to justify here since there is a rounding inside the module
    problem.check_partials(compact_print=True)

    # We try with a high c_rate to check limiter range
    ivc_c_rate_con = copy.deepcopy(ivc_base)
    ivc_c_rate_con.add_output(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:c_rate_max",
        val=6.0,
        units="h**-1",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsEffectiveAreaEnforce(pemfc_stack_id="pemfc_stack_1"),
        ivc_c_rate_con,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:number_modules",
        )
        == pytest.approx(80.0, rel=1e-2)
    )

    # Partials will be hard to justify here since there is a rounding inside the module
    problem.check_partials(compact_print=True)


def test_constraints_ensure_effective_area():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsEffectiveAreaEnsure(pemfc_stack_id="pemfc_stack_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsEffectiveAreaEnsure(pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:min_safe_SOC",
            units="percent",
        )
        == pytest.approx(-20.0, rel=1e-2)
    )
    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:cell:max_c_rate",
            units="percent",
        )
        == pytest.approx(-1.7, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_pemfc_stack_sizing():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingPEMFCStack(pemfc_stack_id="pemfc_stack_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingPEMFCStack(pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:mass", units="kg"
    ) == pytest.approx(3000.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:CG:x", units="m"
    ) == pytest.approx(2.88, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:CG:y", units="m"
    ) == pytest.approx(1.57, rel=1e-2)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:low_speed:CD0",
        )
        == pytest.approx(0.0, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:cruise:CD0",
        )
        == pytest.approx(0.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_pemfc_current_density():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    dc_current_out = np.linspace(1.68, 9.24, NB_POINTS_TEST)
    ivc.add_output("dc_current_out", dc_current_out, units="A")
    ivc.add_output("data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:effective_area",units="cm**2",val=16.8)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCurrentDensity(number_of_points=NB_POINTS_TEST, pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert problem.get_val("fc_current_density", units="A/cm**2") == pytest.approx(
        [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_direct_bus_connection():
    #TODO: Check how to calculate the residuals
    ivc = om.IndepVarComp()
    ivc.add_output("voltage_out", val=np.linspace(400, 400, NB_POINTS_TEST), units="V")
    ivc.add_output("pemfc_voltage", val=np.linspace(390, 400, NB_POINTS_TEST), units="V")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPEMFCDirectBusConnection(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("dc_current_out", units="A") == pytest.approx(
        [4.12, 4.07, 4.03, 3.98, 3.92, 3.86, 3.79, 3.72, 3.66, 3.61], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_single_layer_voltage():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "fc_current_density",
        units="A/cm**2",
        val=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    )
    ivc.add_output(
        name="data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:operation_pressure",
        units="atm",
        val=1.2,
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesSinglePEMFCVoltage(pemfc_stack_id="pemfc_stack_1", number_of_points=7),
        ivc,
    )
    assert problem.get_val("single_layer_pemfc_voltage", units="V") == pytest.approx(
        [0.849, 0.815, 0.786, 0.757, 0.729, 0.699, 0.66], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_pemfc_voltage():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "single_layer_pemfc_voltage",
        val=np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]),
        units="V",
    )
    ivc.add_output("number_of_layers", val=35.)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPEMFCVoltage(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("voltage_out", units="V") == pytest.approx(
        [17.5,19.25,21,22.75,24.5,26.25,28,29.75,31.5,33.25], rel=1e-2
    )

    problem.check_partials(compact_print=True)

    # Check with the other battery mode
    problem = run_system(
        PerformancesPEMFCVoltage(number_of_points=NB_POINTS_TEST, direct_bus_connection=True),
        ivc,
    )
    assert problem.get_val("pemfc_voltage", units="V") == pytest.approx(
        [17.5,19.25,21,22.75,24.5,26.25,28,29.75,31.5,33.25], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_maximum():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "dc_current_out",
        units="A",
        val=np.array([4.01, 3.93, 3.85, 3.8, 3.75, 3.7, 3.67, 3.63, 3.6, 3.57]),
    )


    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(number_of_points=NB_POINTS_TEST, pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:current_min", units="A"
    ) == pytest.approx(
        3.57,
        rel=1e-2,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:current_max", units="A"
    ) == pytest.approx(
        4.01,
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_pemfc_power():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "voltage_out",
        units="V",
        val=np.array([802.0, 786.0, 770.0, 760.0, 750.0, 740.0, 734.0, 726.0, 720.0, 714.0]),
    )
    ivc.add_output("dc_current_out", np.linspace(400, 410, NB_POINTS_TEST), units="A")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPEMFCPower(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("power_out", units="kW") == pytest.approx(
        [320.8, 315.2, 309.7, 306.5, 303.3, 300.1, 298.4, 296.0, 294.4, 292.7],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_pemfc_efficiency():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "single_layer_pemfc_voltage",
        0.9,
        units="V",
    )
    ivc.add_output("data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:operation_pressure",1.2,units="atm")
    ivc.add_output("nominal_pressure",1,units="atm")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPEMFCEfficiency(pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    # Not computed with proper losses, to test only
    assert problem.get_val("efficiency") == pytest.approx(
        0.7253, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_fuel_consumption():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:effective_area",
        units="cm**2",
        val=16.8,
    )
    ivc.add_output(
        "fc_current_density",
        units="A/cm**2",
        val=np.array([0.006,0.0089,0.0119,0.0149,0.0179,0.0208,0.0238,0.0268,0.0298,0.0327]),
        shape=NB_POINTS_TEST,
    )
    ivc.add_output("number_of_layers",val=35.)

    problem = run_system(
        PerformancesPEMFCFuelConsumption(
            number_of_points=NB_POINTS_TEST, pemfc_stack_id="pemfc_stack_1"
        ),
        ivc,
    )
    assert problem.get_val("fuel_consumption", units="kg/h") == pytest.approx(
        [0.000130569948186529,0.000195854922279793,0.000261139896373057,0.000326424870466321,0.000391709844559586,0.000456994818652850,0.000522279792746114,0.000587564766839378,0.000652849740932643,0.000718134715025907], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_fuel_consumed():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "fuel_consumption",
        val=np.array([36.9, 39.6, 42.5, 45.6, 49.0, 52.8, 56.8, 60.8, 65.5, 70.0]),
        units="kg/h",
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesPEMFCFuelConsumed(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("fuel_consumed_t", units="kg") == pytest.approx(
        np.array([5.12, 5.5, 5.9, 6.33, 6.81, 7.33, 7.89, 8.44, 9.1, 9.72]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_performances_pemfc_stack():
    # TODO: Add direct bus connection option test
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesPEMFCStack(number_of_points=NB_POINTS_TEST, pemfc_stack_id="pemfc_stack_1")
        ),
        __file__,
        XML_FILE,
    )
    dc_current_out = np.linspace(400, 410, NB_POINTS_TEST)
    ivc.add_output("dc_current_out", dc_current_out, units="A")
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPEMFCStack(number_of_points=NB_POINTS_TEST, pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert problem.get_val("voltage_out", units="V") == pytest.approx(
        [604.0, 590.5, 580.6, 570.7, 559.3, 546.8, 534.1, 522.5, 512.0, 501.0],
        rel=1e-2,
    )
    assert problem.get_val("state_of_charge", units="percent") == pytest.approx(
        [100.0, 91.5, 83.0, 74.5, 65.9, 57.4, 48.8, 40.1, 31.5, 22.8], rel=1e-2
    )
    assert problem.get_val("losses_battery", units="kW") == pytest.approx(
        [6.32, 7.28, 6.97, 6.15, 5.35, 4.92, 5.02, 5.62, 6.54, 7.44],
        rel=1e-2,
    )
    assert problem.get_val("efficiency") == pytest.approx(
        [0.974, 0.969, 0.97, 0.973, 0.976, 0.978, 0.978, 0.975, 0.971, 0.966],
        rel=1e-2,
    )

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))

    problem.check_partials(compact_print=True)
