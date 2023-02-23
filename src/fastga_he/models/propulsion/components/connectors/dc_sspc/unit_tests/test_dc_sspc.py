# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import pytest
import numpy as np

from ..components.sizing_resistance_scaling import SizingDCSSPCResistanceScaling
from ..components.sizing_reference_resistance import SizingDCSSPCResistances
from ..components.sizing_weight import SizingDCSSPCWeight
from ..components.sizing_dc_sspc_cg import SizingDCSSPCCG
from ..components.perf_resistance import PerformancesDCSSPCResistance
from ..components.perf_current import PerformancesDCSSPCCurrent
from ..components.perf_voltage_out import PerformancesDCSSPCVoltageOut
from ..components.perf_losses import PerformancesDCSSPCLosses
from ..components.perf_power import PerformancesDCSSPCPower
from ..components.perf_efficiency import PerformancesDCSSPCEfficiency
from ..components.perf_maximum import PerformancesDCSSPCMaximum

from ..components.cstr_ensure import ConstraintsCurrentEnsure, ConstraintsVoltageEnsure
from ..components.cstr_enforce import ConstraintsCurrentEnforce, ConstraintsVoltageEnforce

from ..components.sizing_dc_sspc import SizingDCSSPC
from ..components.perf_dc_sspc import PerformancesDCSSPC

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from ..constants import POSSIBLE_POSITION

XML_FILE = "sample_dc_sspc.xml"
NB_POINTS_TEST = 10

# Note: The PerformancesConverterRelations, PerformancesConverterGeneratorSide and
# PerformancesDCDCConverter components cannot be easily tested on its own as it is meant to act
# jointly with the other components of the converter, so it won't be tested here but rather in
# the assemblies


def test_energy_coefficients_scaling_ratio():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCSSPCResistanceScaling(dc_sspc_id="dc_sspc_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingDCSSPCResistanceScaling(dc_sspc_id="dc_sspc_1"), ivc)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:scaling:resistance"
    ) == pytest.approx(1.125, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCSSPCResistances(dc_sspc_id="dc_sspc_1")), __file__, XML_FILE
    )

    problem = run_system(SizingDCSSPCResistances(dc_sspc_id="dc_sspc_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:igbt:resistance", units="ohm"
    ) == pytest.approx(0.00169875, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:diode:resistance", units="ohm"
    ) == pytest.approx(0.00210375, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_weight():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCSSPCWeight(dc_sspc_id="dc_sspc_1")), __file__, XML_FILE
    )

    problem = run_system(SizingDCSSPCWeight(dc_sspc_id="dc_sspc_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:mass", units="kg"
    ) == pytest.approx(10.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_dc_sspc_cg():

    expected_cg = [2.69, 0.45, 2.54]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingDCSSPCCG(dc_sspc_id="dc_sspc_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(SizingDCSSPCCG(dc_sspc_id="dc_sspc_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_sizing_dc_sspc():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(SizingDCSSPC(dc_sspc_id="dc_sspc_1")), __file__, XML_FILE)

    problem = run_system(SizingDCSSPC(dc_sspc_id="dc_sspc_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:mass", units="kg"
    ) == pytest.approx(10.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:CG:x", units="m"
    ) == pytest.approx(2.69, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:low_speed:CD0"
    ) == pytest.approx(0.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:cruise:CD0"
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_resistance():

    # Research independent input value in .xml file, should be the same regardless of option
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesDCSSPCResistance(
                dc_sspc_id="dc_sspc_1", number_of_points=NB_POINTS_TEST, closed=True
            )
        ),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        PerformancesDCSSPCResistance(
            dc_sspc_id="dc_sspc_1", number_of_points=NB_POINTS_TEST, closed=True
        ),
        ivc,
    )
    expected_resistance = np.full(NB_POINTS_TEST, 0.00380)
    assert problem.get_val("resistance_sspc", units="ohm") == pytest.approx(
        expected_resistance, rel=1e-2
    )

    problem = run_system(
        PerformancesDCSSPCResistance(
            dc_sspc_id="dc_sspc_1", number_of_points=NB_POINTS_TEST, closed=False
        ),
        ivc,
    )
    expected_resistance = np.full(NB_POINTS_TEST, np.inf)
    assert problem.get_val("resistance_sspc", units="ohm") == pytest.approx(
        expected_resistance, rel=1e-2
    )


def test_perf_currents():

    ivc = om.IndepVarComp()
    output_current = np.linspace(400, 390, NB_POINTS_TEST)
    ivc.add_output("dc_current_out", units="A", val=output_current)

    problem = run_system(
        PerformancesDCSSPCCurrent(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("dc_current_in", units="A") == pytest.approx(output_current, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_voltage_out():

    ivc = om.IndepVarComp()
    output_current = np.linspace(400, 390, NB_POINTS_TEST)
    ivc.add_output("dc_current_in", units="A", val=output_current)
    resistance = np.full(NB_POINTS_TEST, 0.00380)
    ivc.add_output("resistance_sspc", units="ohm", val=resistance)
    input_voltage = np.full(NB_POINTS_TEST, 600.0)
    ivc.add_output("dc_voltage_in", units="V", val=input_voltage)

    problem = run_system(
        PerformancesDCSSPCVoltageOut(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    expected_voltage = np.array(
        [598.48, 598.48, 598.49, 598.49, 598.5, 598.5, 598.51, 598.51, 598.51, 598.52]
    )
    assert problem.get_val("dc_voltage_out", units="V") == pytest.approx(expected_voltage, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_losses():

    ivc = om.IndepVarComp()
    output_current = np.linspace(400, 390, NB_POINTS_TEST)
    ivc.add_output("dc_current_in", units="A", val=output_current)
    resistance = np.full(NB_POINTS_TEST, 0.00380)
    ivc.add_output("resistance_sspc", units="ohm", val=resistance)

    problem = run_system(
        PerformancesDCSSPCLosses(number_of_points=NB_POINTS_TEST, closed=True),
        ivc,
    )
    expected_losses = np.array(
        [608.0, 604.6, 601.3, 597.9, 594.6, 591.2, 587.9, 584.6, 581.3, 578.0]
    )
    assert problem.get_val("power_losses", units="W") == pytest.approx(expected_losses, rel=1e-2)

    problem.check_partials(compact_print=True)

    problem = run_system(
        PerformancesDCSSPCLosses(number_of_points=NB_POINTS_TEST, closed=False),
        ivc,
    )
    assert problem.get_val("power_losses", units="W") == pytest.approx(
        np.zeros(NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_perf_power():

    ivc = om.IndepVarComp()
    output_current = np.linspace(400, 390, NB_POINTS_TEST)
    ivc.add_output("dc_current_in", units="A", val=output_current)
    input_voltage = np.full(NB_POINTS_TEST, 600.0)
    ivc.add_output("dc_voltage_in", units="V", val=input_voltage)
    output_voltage = np.array(
        [598.48, 598.48, 598.49, 598.49, 598.5, 598.5, 598.51, 598.51, 598.51, 598.52]
    )
    ivc.add_output("dc_voltage_out", units="V", val=output_voltage)

    problem = run_system(
        PerformancesDCSSPCPower(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    expected_power = np.array(
        [240.0, 239.3, 238.7, 238.0, 237.3, 236.7, 236.0, 235.3, 234.7, 234.0]
    )
    assert problem.get_val("power_flow", units="kW") == pytest.approx(expected_power, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_efficiency():

    ivc = om.IndepVarComp()
    losses = np.array([608.0, 604.6, 601.3, 597.9, 594.6, 591.2, 587.9, 584.6, 581.3, 578.0])
    ivc.add_output("power_losses", units="W", val=losses)
    power = np.array([240.0, 239.3, 238.7, 238.0, 237.3, 236.7, 236.0, 235.3, 234.7, 234.0])
    ivc.add_output("power_flow", units="kW", val=power)

    problem = run_system(
        PerformancesDCSSPCEfficiency(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    expected_efficiency = np.array(
        [0.99747, 0.99747, 0.99748, 0.99749, 0.99749, 0.9975, 0.99751, 0.99752, 0.99752, 0.99753]
    )
    assert problem.get_val("efficiency") == pytest.approx(expected_efficiency, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_maximum():

    ivc = om.IndepVarComp()
    output_current = np.linspace(400, 390, NB_POINTS_TEST)
    ivc.add_output("dc_current_in", units="A", val=output_current)
    input_voltage = np.full(NB_POINTS_TEST, 600.0)
    ivc.add_output("dc_voltage_in", units="V", val=input_voltage)
    output_voltage = np.array(
        [598.48, 598.48, 598.49, 598.49, 598.5, 598.5, 598.51, 598.51, 598.51, 598.52]
    )
    ivc.add_output("dc_voltage_out", units="V", val=output_voltage)

    problem = run_system(
        PerformancesDCSSPCMaximum(dc_sspc_id="dc_sspc_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:current_max", units="A"
    ) == pytest.approx(400, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:voltage_max", units="V"
    ) == pytest.approx(600, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_current_ensure():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentEnsure(dc_sspc_id="dc_sspc_1")), __file__, XML_FILE
    )

    problem = run_system(ConstraintsCurrentEnsure(dc_sspc_id="dc_sspc_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:DC_SSPC:dc_sspc_1:current_caliber", units="A"
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_voltage_ensure():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnsure(dc_sspc_id="dc_sspc_1")), __file__, XML_FILE
    )

    problem = run_system(ConstraintsVoltageEnsure(dc_sspc_id="dc_sspc_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:DC_SSPC:dc_sspc_1:voltage_caliber", units="V"
    ) == pytest.approx(-150.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_current_enforce():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentEnforce(dc_sspc_id="dc_sspc_1")), __file__, XML_FILE
    )

    problem = run_system(ConstraintsCurrentEnforce(dc_sspc_id="dc_sspc_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:current_caliber", units="A"
    ) == pytest.approx(400.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_voltage_enforce():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnforce(dc_sspc_id="dc_sspc_1")), __file__, XML_FILE
    )

    problem = run_system(ConstraintsVoltageEnforce(dc_sspc_id="dc_sspc_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:voltage_caliber", units="V"
    ) == pytest.approx(600.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_performances():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesDCSSPC(dc_sspc_id="dc_sspc_1", number_of_points=NB_POINTS_TEST, closed=True)
        ),
        __file__,
        XML_FILE,
    )
    output_current = np.linspace(400, 390, NB_POINTS_TEST)
    ivc.add_output("dc_current_out", units="A", val=output_current)
    input_voltage = np.full(NB_POINTS_TEST, 600.0)
    ivc.add_output("dc_voltage_in", units="V", val=input_voltage)

    problem = run_system(
        PerformancesDCSSPC(dc_sspc_id="dc_sspc_1", number_of_points=NB_POINTS_TEST, closed=True),
        ivc,
    )
    expected_voltage = np.array(
        [598.48, 598.48, 598.49, 598.49, 598.5, 598.5, 598.51, 598.51, 598.51, 598.52]
    )
    assert problem.get_val("dc_voltage_out", units="V") == pytest.approx(expected_voltage, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:current_max", units="A"
    ) == pytest.approx(400, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:voltage_max", units="V"
    ) == pytest.approx(600, rel=1e-2)

    expected_efficiency = np.array(
        [0.99747, 0.99747, 0.99748, 0.99749, 0.99749, 0.9975, 0.99751, 0.99752, 0.99752, 0.99753]
    )
    assert problem.get_val("efficiency") == pytest.approx(expected_efficiency, rel=1e-3)

    problem.check_partials(compact_print=True)
