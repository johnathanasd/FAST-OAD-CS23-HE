# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
import copy

import openmdao.api as om
import pytest
import numpy as np
import fastoad.api as oad

from ..components.sizing_pemfc_weight import SizingPEMFCWeightAerostak200W
from ..components.sizing_pemfc_weight import SizingPEMFCWeightAdjusted
from ..components.sizing_pemfc_cg_x import SizingPEMFCCGX
from ..components.sizing_pemfc_cg_y import SizingPEMFCCGY
from ..components.sizing_pemfc_volume import SizingPEMFCVolume
from ..components.sizing_pemfc_dimensions import SizingPEMFCDimensions
from ..components.sizing_pemfc_drag import SizingPEMFCDrag

from ..components.perf_fuel_consumption import PerformancesPEMFCFuelConsumption
from ..components.perf_fuel_consumed import PerformancesPEMFCFuelConsumed
from ..components.perf_layer_voltage import PerformancesSinglePEMFCVoltageStatistical
from ..components.perf_layer_voltage import PerformancesSinglePEMFCVoltageAnalytical
from ..components.perf_pemfc_current_density import PerformancesCurrentDensity
from ..components.perf_maximum import PerformancesMaximum
from ..components.perf_pemfc_efficiency import PerformancesPEMFCEfficiency
from ..components.perf_pemfc_power import PerformancesPEMFCPower
from ..components.perf_pemfc_power_density import PerformancesPEMFCPowerDensity
from ..components.perf_pemfc_voltage import PerformancesPEMFCVoltage
from ..components.perf_operation_pressure import PerformancesOperationPressure
from ..components.perf_pemfc_expect_power_density import PerformancesPEMFCMaxPowerDensityAerostak
from ..components.perf_pemfc_expect_power_density import (
    PerformancesPEMFCMaxPowerDensityIntelligentEnergy,
)
from ..components.perf_analytical_voltage_adjustment import PerformancesAnalyticalVoltageAdjustment

from ..components.cstr_ensure import ConstraintsEffectiveAreaEnsure
from ..components.cstr_enforce import ConstraintsEffectiveAreaEnforce

from ..components.sizing_pemfc_stack import SizingPEMFCStack
from ..components.perf_pemfc_stack import PerformancesPEMFCStack

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_pemfc_stack.xml"
NB_POINTS_TEST = 10


def test_pemfc_weight_aerostak_200w():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingPEMFCWeightAerostak200W(pemfc_stack_id="pemfc_stack_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingPEMFCWeightAerostak200W(pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:mass", units="kg"
    ) == pytest.approx(0.5, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_pemfc_weight_adjusted():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingPEMFCWeightAdjusted(pemfc_stack_id="pemfc_stack_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingPEMFCWeightAdjusted(pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:mass", units="kg"
    ) == pytest.approx(0.5, rel=1e-2)

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
    ) == pytest.approx(1.62, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_pemfc_dimensions():

    expected_length = [0.11998, 0.11998, 0.11998, 0.11998]
    expected_width = [0.1162, 0.1162, 0.95824, 0.1162]
    expected_height = [0.1162, 0.1162, 0.01409, 0.1162]

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

    expected_values = [0.44, 2.88466, 1.2387, 2.0374]

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

    expected_values = [0.0, 1.57, 0.0, 0.0]

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

    expected_ls_drag = [0.0, 0.0002985, 3.493e-5, 0.0]
    expected_cruise_drag = [0.0, 0.0002985, 3.445e-5, 0.0]

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


def test_pemfc_stack_sizing():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingPEMFCStack(pemfc_stack_id="pemfc_stack_1")),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:current_max",
        val=11.76,
        units="A",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingPEMFCStack(pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:mass", units="kg"
    ) == pytest.approx(0.5, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:CG:x", units="m"
    ) == pytest.approx(1.2387, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:CG:y", units="m"
    ) == pytest.approx(0.0, rel=1e-2)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:low_speed:CD0",
        )
        == pytest.approx(7.377e-5, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:cruise:CD0",
        )
        == pytest.approx(7.276e-5, rel=1e-2)
    )

    problem.check_partials(compact_print=True)
    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_constraints_enforce_effective_area():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:current_max",
        val=7,
        units="A",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsEffectiveAreaEnforce(pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:effective_area", units="cm**2"
    ) == pytest.approx(10, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_ensure_effective_area():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsEffectiveAreaEnsure(pemfc_stack_id="pemfc_stack_1")),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:current_max",
        val=14,
        units="A",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsEffectiveAreaEnsure(pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:effective_area",
            units="cm**2",
        )
        == pytest.approx(3.2, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_pemfc_current_density():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    dc_current_out = np.linspace(1.68, 9.24, NB_POINTS_TEST)
    ivc.add_output("dc_current_out", dc_current_out, units="A")
    ivc.add_output(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:effective_area",
        units="cm**2",
        val=16.8,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCurrentDensity(number_of_points=NB_POINTS_TEST, pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert problem.get_val("fc_current_density", units="A/cm**2") == pytest.approx(
        [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_operation_pressure():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "altitude",
        units="m",
        val=np.zeros(NB_POINTS_TEST),
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesOperationPressure(
            pemfc_stack_id="pemfc_stack_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    assert problem.get_val("operation_pressure", units="atm") == pytest.approx(
        [1.0] * int(NB_POINTS_TEST), rel=1e-2
    )
    problem.check_partials(compact_print=True)


def test_analytical_voltage_adjustment():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "operation_pressure",
        units="atm",
        val=np.ones(NB_POINTS_TEST),
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesAnalyticalVoltageAdjustment(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("analytical_voltage_adjust_factor") == pytest.approx(
        [1.0] * int(NB_POINTS_TEST), rel=1e-2
    )
    problem.check_partials(compact_print=True)


def test_single_layer_voltage_statistical():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "fc_current_density",
        units="A/cm**2",
        val=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    )
    ivc.add_output(
        name="operation_pressure",
        units="atm",
        val=np.array([1.2] * 7),
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesSinglePEMFCVoltageStatistical(
            pemfc_stack_id="pemfc_stack_1", number_of_points=7
        ),
        ivc,
    )
    assert problem.get_val("single_layer_pemfc_voltage", units="V") == pytest.approx(
        [0.849, 0.815, 0.786, 0.757, 0.729, 0.699, 0.66], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_single_layer_voltage_analytical():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "fc_current_density",
        units="A/cm**2",
        val=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    )
    ivc.add_output(
        name="operation_pressure",
        units="atm",
        val=np.array([1.2] * 7),
    )
    ivc.add_output(
        name="hydrogen_reactant_pressure",
        units="atm",
        val=1.0,
    )
    ivc.add_output(
        name="operation_temperature",
        units="degC",
        val=np.array([50.0] * 7),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesSinglePEMFCVoltageAnalytical(
            pemfc_stack_id="pemfc_stack_1", number_of_points=7
        ),
        ivc,
    )

    assert problem.get_val("single_layer_pemfc_voltage", units="V") == pytest.approx(
        [0.857, 0.799, 0.75, 0.71, 0.66, 0.62, 0.57],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_pemfc_voltage():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "single_layer_pemfc_voltage",
        val=np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]),
        units="V",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:number_of_layers", val=35.0
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPEMFCVoltage(number_of_points=NB_POINTS_TEST, pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert problem.get_val("voltage_out", units="V") == pytest.approx(
        [17.5, 19.25, 21, 22.75, 24.5, 26.25, 28, 29.75, 31.5, 33.25], rel=1e-2
    )

    problem.check_partials(compact_print=True)

    # Check with the other battery mode
    problem = run_system(
        PerformancesPEMFCVoltage(
            number_of_points=NB_POINTS_TEST,
            direct_bus_connection=True,
            pemfc_stack_id="pemfc_stack_1",
        ),
        ivc,
    )
    assert problem.get_val("pemfc_voltage", units="V") == pytest.approx(
        [17.5, 19.25, 21, 22.75, 24.5, 26.25, 28, 29.75, 31.5, 33.25], rel=1e-2
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
    ivc.add_output(
        "power_out",
        units="kW",
        val=np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]),
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
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:power_min", units="kW"
    ) == pytest.approx(
        10.0,
        rel=1e-2,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:power_max", units="kW"
    ) == pytest.approx(
        100.0,
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_max_power_density_aerostak():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:power_max",
        units="kW",
        val=0.2,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPEMFCMaxPowerDensityAerostak(pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:max_power_density", units="kW/kg"
    ) == pytest.approx(
        0.401,
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_max_power_density_intelligent_energy():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:power_max",
        units="kW",
        val=600.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPEMFCMaxPowerDensityIntelligentEnergy(pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:max_power_density", units="kW/kg"
    ) == pytest.approx(
        2.06,
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


def test_pemfc_power_density():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "power_out",
        units="kW",
        val=np.array([802.0, 786.0, 770.0, 760.0, 750.0, 740.0, 734.0, 726.0, 720.0, 714.0]),
    )
    ivc.add_output(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:mass", 1000.0, units="kg"
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPEMFCPowerDensity(
            number_of_points=NB_POINTS_TEST, pemfc_stack_id="pemfc_stack_1"
        ),
        ivc,
    )
    assert problem.get_val("power_density", units="kW/kg") == pytest.approx(
        [0.802, 0.786, 0.77, 0.76, 0.75, 0.74, 0.734, 0.726, 0.72, 0.714],
        rel=1e-2,
    )
    problem.check_partials(compact_print=True)


def test_pemfc_efficiency():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "single_layer_pemfc_voltage",
        val=np.array(
            [0.8492, 0.8315, 0.8154, 0.8002, 0.7856, 0.7713, 0.7572, 0.7431, 0.7289, 0.7143]
        ),
        units="V",
    )
    ivc.add_output(
        "operation_pressure",
        np.array([1.2] * int(NB_POINTS_TEST)),
        units="atm",
    )
    ivc.add_output("nominal_pressure", 1, units="atm")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPEMFCEfficiency(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    # Not computed with proper losses, to test only
    assert problem.get_val("efficiency") == pytest.approx(
        [0.6843, 0.67, 0.6570, 0.6449, 0.6330, 0.6216, 0.6102, 0.5989, 0.5874, 0.5756], rel=1e-2
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
        val=np.array(
            [0.006, 0.0089, 0.0119, 0.0149, 0.0179, 0.0208, 0.0238, 0.0268, 0.0298, 0.0327]
        ),
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:pemfc_stack:pemfc_stack_1:number_of_layers", val=35.0
    )

    problem = run_system(
        PerformancesPEMFCFuelConsumption(
            number_of_points=NB_POINTS_TEST, pemfc_stack_id="pemfc_stack_1"
        ),
        ivc,
    )
    assert problem.get_val("fuel_consumption", units="kg/h") == pytest.approx(
        [
            0.000130569948186529,
            0.000195854922279793,
            0.000261139896373057,
            0.000326424870466321,
            0.000391709844559586,
            0.000456994818652850,
            0.000522279792746114,
            0.000587564766839378,
            0.000652849740932643,
            0.000718134715025907,
        ],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_fuel_consumed():

    ivc = om.IndepVarComp()
    ivc.add_output(
        name="fuel_consumption",
        val=np.array([36.9, 39.6, 42.5, 45.6, 49.0, 52.8, 56.8, 60.8, 65.5, 70.0]),
        units="kg/h",
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPEMFCFuelConsumed(
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val("fuel_consumed_t", units="kg") == pytest.approx(
        np.array([5.12, 5.5, 5.9, 6.33, 6.81, 7.33, 7.89, 8.44, 9.1, 9.72]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_performances_pemfc_stack():
    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.performances.pemfc.layer_voltage"
    ] = "fastga_he.submodel.propulsion.performances.pemfc.layer_voltage.statistical"
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesPEMFCStack(number_of_points=NB_POINTS_TEST, pemfc_stack_id="pemfc_stack_1")
        ),
        __file__,
        XML_FILE,
    )
    dc_current_out = np.linspace(1.68, 9.24, NB_POINTS_TEST)
    ivc.add_output("dc_current_out", dc_current_out, units="A")
    ivc.add_output("time_step", units="h", val=np.full(NB_POINTS_TEST, 1))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPEMFCStack(number_of_points=NB_POINTS_TEST, pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )

    assert problem.get_val("single_layer_pemfc_voltage", units="V") == pytest.approx(
        [
            0.84175546,
            0.82406536,
            0.80801377,
            0.79284721,
            0.77821938,
            0.76392795,
            0.74982252,
            0.73575584,
            0.72154143,
            0.70689839,
        ],
        rel=1e-2,
    )

    assert problem.get_val("fc_current_density", units="A/cm**2") == pytest.approx(
        [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55], rel=1e-2
    )
    assert problem.get_val("fuel_consumption", units="kg/h") == pytest.approx(
        [0.00219, 0.00329, 0.00439, 0.00548, 0.00658, 0.00768, 0.00877, 0.00987, 0.01097, 0.01207],
        rel=1e-2,
    )

    assert problem.get_val("efficiency") == pytest.approx(
        [0.6843, 0.67, 0.6570, 0.6449, 0.6330, 0.6216, 0.6102, 0.5989, 0.5874, 0.5756], rel=1e-2
    )

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))

    problem.check_partials(compact_print=True)


def test_performances_pemfc_stack_analytical():
    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.performances.pemfc.layer_voltage"
    ] = "fastga_he.submodel.propulsion.performances.pemfc.layer_voltage.analytical"
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesPEMFCStack(number_of_points=NB_POINTS_TEST, pemfc_stack_id="pemfc_stack_1")
        ),
        __file__,
        XML_FILE,
    )
    dc_current_out = np.linspace(1.68, 9.24, NB_POINTS_TEST)
    ivc.add_output("dc_current_out", dc_current_out, units="A")
    ivc.add_output("time_step", units="h", val=np.full(NB_POINTS_TEST, 1))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPEMFCStack(number_of_points=NB_POINTS_TEST, pemfc_stack_id="pemfc_stack_1"),
        ivc,
    )

    assert problem.get_val("single_layer_pemfc_voltage", units="V") == pytest.approx(
        [
            0.90045315,
            0.87104179,
            0.84551353,
            0.82201006,
            0.79965299,
            0.77794886,
            0.75658583,
            0.73534818,
            0.71407501,
            0.69263817,
        ],
        rel=1e-2,
    )

    assert problem.get_val("fc_current_density", units="A/cm**2") == pytest.approx(
        [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55], rel=1e-2
    )
    assert problem.get_val("fuel_consumption", units="kg/h") == pytest.approx(
        [0.00219, 0.00329, 0.00439, 0.00548, 0.00658, 0.00768, 0.00877, 0.00987, 0.01097, 0.01207],
        rel=1e-2,
    )

    assert problem.get_val("efficiency") == pytest.approx(
        [
            0.7326714,
            0.70874027,
            0.6879687,
            0.66884464,
            0.65065337,
            0.63299337,
            0.61561092,
            0.59833049,
            0.58102117,
            0.56357866,
        ],
        rel=1e-2,
    )

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))

    problem.check_partials(compact_print=True)
