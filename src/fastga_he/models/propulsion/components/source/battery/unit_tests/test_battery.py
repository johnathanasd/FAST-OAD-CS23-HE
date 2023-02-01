# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import pytest
import numpy as np

from ..components.sizing_module_weight import SizingBatteryModuleWeight
from ..components.sizing_battery_weight import SizingBatteryWeight
from ..components.sizing_number_cells import SizingBatteryNumberCells
from ..components.sizing_battery_cg import SizingBatteryCG
from ..components.sizing_module_volume import SizingBatteryModuleVolume
from ..components.sizing_battery_volume import SizingBatteryVolume
from ..components.sizing_battery_dimensions import SizingBatteryDimensions
from ..components.sizing_battery_drag import SizingBatteryDrag
from ..components.perf_cell_temperature import PerformancesCellTemperatureMission
from ..components.perf_module_current import PerformancesModuleCurrent
from ..components.perf_open_circuit_voltage import PerformancesOpenCircuitVoltage
from ..components.perf_internal_resistance import PerformancesInternalResistance
from ..components.perf_cell_voltage import PerformancesCellVoltage
from ..components.perf_module_voltage import PerformancesModuleVoltage
from ..components.perf_battery_voltage import PerformancesBatteryVoltage
from ..components.perf_battery_c_rate import PerformancesModuleCRate
from ..components.perf_soc_decrease import PerformancesSOCDecrease
from ..components.perf_update_soc import PerformancesUpdateSOC
from ..components.perf_joule_losses import PerformancesCellJouleLosses
from ..components.perf_entropic_heat_coefficient import PerformancesEntropicHeatCoefficient
from ..components.perf_entropic_losses import PerformancesCellEntropicLosses
from ..components.perf_cell_losses import PerformancesCellLosses
from ..components.perf_battery_losses import PerformancesBatteryLosses
from ..components.perf_maximum import PerformancesMaximum
from ..components.perf_energy_consumption import PerformancesEnergyConsumption
from ..components.cstr_ensure import ConstraintsSOCEnsure
from ..components.cstr_enforce import ConstraintsSOCEnforce

from ..components.sizing_battery_pack import SizingBatteryPack
from ..components.perf_battery_pack import PerformancesBatteryPack

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_battery_pack.xml"
NB_POINTS_TEST = 10


def test_module_weight():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingBatteryModuleWeight(battery_pack_id="battery_pack_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingBatteryModuleWeight(battery_pack_id="battery_pack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:module:mass", units="kg"
    ) == pytest.approx(198.4, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_battery_weight():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingBatteryWeight(battery_pack_id="battery_pack_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingBatteryWeight(battery_pack_id="battery_pack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(7936.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_battery_cell_number():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingBatteryNumberCells(battery_pack_id="battery_pack_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingBatteryNumberCells(battery_pack_id="battery_pack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:number_cells"
    ) == pytest.approx(8000.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_module_volume():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingBatteryModuleVolume(battery_pack_id="battery_pack_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingBatteryModuleVolume(battery_pack_id="battery_pack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:module:volume", units="L"
    ) == pytest.approx(38.75, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_battery_volume():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingBatteryVolume(battery_pack_id="battery_pack_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingBatteryVolume(battery_pack_id="battery_pack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:volume", units="L"
    ) == pytest.approx(1550.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_battery_dimensions():

    expected_length = [0.511, 7.03, 0.81, 0.81, 1.48]
    expected_width = [3.69, 0.47, 1.67, 1.67, 0.95]
    expected_height = [0.82, 0.47, 1.14, 1.14, 1.09]

    for option, length, width, height in zip(
        POSSIBLE_POSITION, expected_length, expected_width, expected_height
    ):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingBatteryDimensions(battery_pack_id="battery_pack_1", position=option)),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingBatteryDimensions(battery_pack_id="battery_pack_1", position=option),
            ivc,
        )
        assert problem.get_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:dimension:length", units="m"
        ) == pytest.approx(length, rel=1e-2)
        assert problem.get_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:dimension:width", units="m"
        ) == pytest.approx(width, rel=1e-2)
        assert problem.get_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:dimension:height", units="m"
        ) == pytest.approx(height, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_battery_cg():

    expected_values = [2.88, 2.88, 0.095, 2.38, 1.24]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingBatteryCG(battery_pack_id="battery_pack_1", position=option)),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingBatteryCG(battery_pack_id="battery_pack_1", position=option),
            ivc,
        )
        assert problem.get_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_battery_drag():

    expected_ls_drag = [0.0, 0.021, 0.0, 0.0, 3.77e-3]
    expected_cruise_drag = [0.0, 0.021, 0.0, 0.0, 3.72e-3]

    for option, ls_drag, cruise_drag in zip(
        POSSIBLE_POSITION, expected_ls_drag, expected_cruise_drag
    ):
        # Research independent input value in .xml file
        for ls_option in [True, False]:
            ivc = get_indep_var_comp(
                list_inputs(
                    SizingBatteryDrag(
                        battery_pack_id="battery_pack_1", position=option, low_speed_aero=ls_option
                    )
                ),
                __file__,
                XML_FILE,
            )

            # Run problem and check obtained value(s) is/(are) correct
            problem = run_system(
                SizingBatteryDrag(
                    battery_pack_id="battery_pack_1", position=option, low_speed_aero=ls_option
                ),
                ivc,
            )

            if ls_option:
                assert (
                    problem.get_val(
                        "data:propulsion:he_power_train:battery_pack:battery_pack_1:low_speed:CD0",
                    )
                    == pytest.approx(ls_drag, rel=1e-2)
                )
            else:
                assert (
                    problem.get_val(
                        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cruise:CD0",
                    )
                    == pytest.approx(cruise_drag, rel=1e-2)
                )

            problem.check_partials(compact_print=True)


def test_constraints_enforce_soc():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsSOCEnforce(battery_pack_id="battery_pack_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsSOCEnforce(battery_pack_id="battery_pack_1"),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:number_modules",
        )
        == pytest.approx(32.0, rel=1e-2)
    )

    # Partials will be hard to justify here since there is a rounding inside the module
    problem.check_partials(compact_print=True)


def test_constraints_ensure_soc():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsSOCEnsure(battery_pack_id="battery_pack_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsSOCEnsure(battery_pack_id="battery_pack_1"),
        ivc,
    )
    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:battery_pack:battery_pack_1:min_safe_SOC",
            units="percent",
        )
        == pytest.approx(-20.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_battery_pack_sizing():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingBatteryPack(battery_pack_id="battery_pack_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingBatteryPack(battery_pack_id="battery_pack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(7936.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:CG:x", units="m"
    ) == pytest.approx(2.88, rel=1e-2)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:low_speed:CD0",
        )
        == pytest.approx(0.0, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:battery_pack:battery_pack_1:cruise:CD0",
        )
        == pytest.approx(0.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_cell_temperature_mission():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell_temperature_mission",
        val=290.0,
        units="degK",
    )

    problem = run_system(
        PerformancesCellTemperatureMission(
            battery_pack_id="battery_pack_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("cell_temperature", units="degK") == pytest.approx(
        np.full(NB_POINTS_TEST, 290), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc3 = om.IndepVarComp()
    ivc3.add_output(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell_temperature_mission",
        val=[290, 270, 290, 290, 270, 290, 290, 270, 290, 42],
        units="degK",
    )

    problem3 = run_system(
        PerformancesCellTemperatureMission(
            battery_pack_id="battery_pack_1", number_of_points=NB_POINTS_TEST
        ),
        ivc3,
    )

    assert problem3.get_val("cell_temperature", units="degK") == pytest.approx(
        np.array([290, 270, 290, 290, 270, 290, 290, 270, 290, 42]),
        rel=1e-2,
    )

    problem3.check_partials(compact_print=True)


def test_current_module():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesModuleCurrent(
                number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"
            )
        ),
        __file__,
        XML_FILE,
    )
    dc_current_out = np.linspace(400, 410, NB_POINTS_TEST)
    ivc.add_output("dc_current_out", dc_current_out, units="A")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesModuleCurrent(
            number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"
        ),
        ivc,
    )
    assert problem.get_val("current_one_module", units="A") == pytest.approx(
        [10.0, 10.03, 10.06, 10.08, 10.11, 10.14, 10.17, 10.19, 10.22, 10.25], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_open_circuit_voltage():

    ivc = om.IndepVarComp()
    ivc.add_output("state_of_charge", val=np.linspace(100, 40, NB_POINTS_TEST), units="percent")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesOpenCircuitVoltage(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("open_circuit_voltage", units="V") == pytest.approx(
        [4.12, 4.07, 4.03, 3.98, 3.92, 3.86, 3.79, 3.72, 3.66, 3.61], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_internal_resistance():

    ivc = om.IndepVarComp()
    ivc.add_output("state_of_charge", val=np.linspace(100, 40, NB_POINTS_TEST), units="percent")
    ivc.add_output(
        "cell_temperature",
        units="degK",
        val=np.linspace(288.15, 298.15, NB_POINTS_TEST),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesInternalResistance(
            number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"
        ),
        ivc,
    )
    assert problem.get_val("internal_resistance", units="ohm") * 1e3 == pytest.approx(
        [46.59, 56.13, 58.11, 56.15, 52.7, 49.3, 46.76, 45.32, 44.82, 44.73], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_cell_voltage():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "current_one_module",
        units="A",
        val=np.array([10.0, 10.03, 10.06, 10.08, 10.11, 10.14, 10.17, 10.19, 10.22, 10.25]),
    )
    ivc.add_output(
        "open_circuit_voltage",
        units="V",
        val=np.array([4.04, 3.96, 3.89, 3.84, 3.79, 3.75, 3.72, 3.68, 3.65, 3.62]),
    )
    ivc.add_output(
        "internal_resistance",
        units="ohm",
        val=np.array([2.96, 3.44, 3.83, 4.13, 4.35, 4.51, 4.62, 4.68, 4.71, 4.71]) * 1e-3,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCellVoltage(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("terminal_voltage", units="V") == pytest.approx(
        [4.01, 3.93, 3.85, 3.8, 3.75, 3.7, 3.67, 3.63, 3.6, 3.57], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_module_voltage():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesModuleVoltage(
                number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "terminal_voltage",
        units="V",
        val=np.array([4.01, 3.93, 3.85, 3.8, 3.75, 3.7, 3.67, 3.63, 3.6, 3.57]),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesModuleVoltage(
            number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"
        ),
        ivc,
    )
    assert problem.get_val("module_voltage", units="V") == pytest.approx(
        [802.0, 786.0, 770.0, 760.0, 750.0, 740.0, 734.0, 726.0, 720.0, 714.0], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_battery_voltage():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "module_voltage",
        val=np.array([802.0, 786.0, 770.0, 760.0, 750.0, 740.0, 734.0, 726.0, 720.0, 714.0]),
        units="V",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesBatteryVoltage(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("voltage_out", units="V") == pytest.approx(
        [802.0, 786.0, 770.0, 760.0, 750.0, 740.0, 734.0, 726.0, 720.0, 714.0], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_module_c_rate():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesModuleCRate(
                number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "current_one_module",
        units="A",
        val=np.array([10.0, 10.03, 10.06, 10.08, 10.11, 10.14, 10.17, 10.19, 10.22, 10.25]),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesModuleCRate(number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"),
        ivc,
    )
    assert problem.get_val("c_rate", units="h**-1") == pytest.approx(
        [0.5, 0.5015, 0.503, 0.504, 0.5055, 0.507, 0.5085, 0.5095, 0.511, 0.5125], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_module_soc_decrease():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "c_rate",
        units="h**-1",
        val=np.array([0.5, 0.5015, 0.503, 0.504, 0.5055, 0.507, 0.5085, 0.5095, 0.511, 0.5125]),
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesSOCDecrease(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("state_of_charge_decrease", units="percent") == pytest.approx(
        [6.94, 6.97, 6.99, 7.0, 7.02, 7.04, 7.06, 7.08, 7.1, 7.12],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_update_soc():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "state_of_charge_decrease",
        units="percent",
        val=np.array([6.94, 6.97, 6.99, 7.0, 7.02, 7.04, 7.06, 7.08, 7.1, 7.12]),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesUpdateSOC(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("state_of_charge", units="percent") == pytest.approx(
        [100.0, 93.06, 86.09, 79.1, 72.1, 65.08, 58.04, 50.98, 43.9, 36.8],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_cell_joules_losses():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "current_one_module",
        units="A",
        val=np.array([10.0, 10.03, 10.06, 10.08, 10.11, 10.14, 10.17, 10.19, 10.22, 10.25]),
    )
    ivc.add_output(
        "internal_resistance",
        units="ohm",
        val=np.array([2.96, 3.44, 3.83, 4.13, 4.35, 4.51, 4.62, 4.68, 4.71, 4.71]) * 1e-3,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCellJouleLosses(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("joule_losses_cell", units="mW") == pytest.approx(
        [296.0, 346.07, 387.61, 419.63, 444.62, 463.72, 477.84, 485.95, 491.95, 494.84], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_cell_entropic_heat_coefficient():

    ivc = om.IndepVarComp()
    ivc.add_output("state_of_charge", val=np.linspace(100, 40, NB_POINTS_TEST), units="percent")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesEntropicHeatCoefficient(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("entropic_heat_coefficient", units="mV/degK") == pytest.approx(
        [-0.0099, 0.0305, -0.0312, -0.0894, -0.0958, -0.0454, 0.0381, 0.1157, 0.148, 0.1091],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_cell_entropic_losses():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "current_one_module",
        units="A",
        val=np.array([10.0, 10.03, 10.06, 10.08, 10.11, 10.14, 10.17, 10.19, 10.22, 10.25]),
    )
    ivc.add_output(
        "entropic_heat_coefficient",
        units="mV/degK",
        val=np.array(
            [-0.0099, 0.0305, -0.0312, -0.0894, -0.0958, -0.0454, 0.0381, 0.1157, 0.148, 0.1091]
        ),
    )
    ivc.add_output(
        "cell_temperature",
        units="degK",
        val=np.full(NB_POINTS_TEST, 288.15),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCellEntropicLosses(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("entropic_losses_cell", units="mW") == pytest.approx(
        [28.53, -88.15, 90.44, 259.67, 279.08, 132.65, -111.65, -339.72, -435.84, -322.23], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_cell_total_losses():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "entropic_losses_cell",
        units="mW",
        val=np.array(
            [28.53, -88.15, 90.44, 259.67, 279.08, 132.65, -111.65, -339.72, -435.84, -322.23]
        ),
    )
    ivc.add_output(
        "joule_losses_cell",
        units="mW",
        val=np.array(
            [296.0, 346.07, 387.61, 419.63, 444.62, 463.72, 477.84, 485.95, 491.95, 494.84]
        ),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCellLosses(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("losses_cell", units="mW") == pytest.approx(
        [324.53, 257.92, 478.05, 679.3, 723.7, 596.37, 366.19, 146.23, 56.11, 172.61], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_battery_losses():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesBatteryLosses(
                number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "losses_cell",
        units="mW",
        val=np.array([324.53, 257.92, 478.05, 679.3, 723.7, 596.37, 366.19, 146.23, 56.11, 172.61]),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesBatteryLosses(
            number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"
        ),
        ivc,
    )
    assert problem.get_val("losses_battery", units="W") == pytest.approx(
        [2596.24, 2063.36, 3824.4, 5434.4, 5789.6, 4770.96, 2929.52, 1169.84, 448.88, 1380.88],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_maximum():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "terminal_voltage",
        units="V",
        val=np.array([4.01, 3.93, 3.85, 3.8, 3.75, 3.7, 3.67, 3.63, 3.6, 3.57]),
    )
    ivc.add_output("state_of_charge", val=np.linspace(100, 40, NB_POINTS_TEST), units="percent")
    ivc.add_output(
        "c_rate",
        units="h**-1",
        val=np.array([0.5, 0.5015, 0.503, 0.504, 0.5055, 0.507, 0.5085, 0.5095, 0.511, 0.5125]),
    )
    ivc.add_output(
        "losses_cell",
        units="mW",
        val=np.array([324.53, 257.92, 478.05, 679.3, 723.7, 596.37, 366.19, 146.23, 56.11, 172.61]),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:voltage_min", units="V"
    ) == pytest.approx(
        3.57,
        rel=1e-2,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:voltage_max", units="V"
    ) == pytest.approx(
        4.01,
        rel=1e-2,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    ) == pytest.approx(
        40.0,
        rel=1e-2,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:c_rate_max", units="h**-1"
    ) == pytest.approx(
        0.5125,
        rel=1e-2,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:losses_max", units="mW"
    ) == pytest.approx(
        723.7,
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_energy_consumption():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "voltage_out",
        units="V",
        val=np.array([802.0, 786.0, 770.0, 760.0, 750.0, 740.0, 734.0, 726.0, 720.0, 714.0]),
    )
    ivc.add_output("dc_current_out", np.linspace(400, 410, NB_POINTS_TEST), units="A")
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    problem = run_system(
        PerformancesEnergyConsumption(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("non_consumable_energy_t", units="kW*h") == pytest.approx(
        [44.556, 43.788, 43.015, 42.574, 42.13, 41.682, 41.457, 41.118, 40.889, 40.658], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_performances_battery_pack():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesBatteryPack(
                number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"
            )
        ),
        __file__,
        XML_FILE,
    )
    dc_current_out = np.linspace(400, 410, NB_POINTS_TEST)
    ivc.add_output("dc_current_out", dc_current_out, units="A")
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesBatteryPack(number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"),
        ivc,
    )
    assert problem.get_val("voltage_out", units="V") == pytest.approx(
        [730.8, 695.4, 676.7, 666.0, 657.5, 647.6, 635.2, 620.4, 604.5, 589.7], rel=1e-2
    )
    assert problem.get_val("state_of_charge", units="percent") == pytest.approx(
        [100.0, 93.06, 86.09, 79.1, 72.1, 65.08, 58.04, 50.98, 43.9, 36.8], rel=1e-2
    )
    assert problem.get_val("component.battery_losses.losses_battery", units="kW") == pytest.approx(
        [37.5, 46.8, 52.1, 53.7, 52.3, 49.5, 46.9, 45.8, 47.0, 50.4],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)
