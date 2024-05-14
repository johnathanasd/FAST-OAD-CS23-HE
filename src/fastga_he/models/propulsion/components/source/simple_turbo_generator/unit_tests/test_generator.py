# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest

import openmdao.api as om

from ..components.perf_mission_rpm import PerformancesRPMMission
from ..components.perf_voltage_out_target import PerformancesVoltageOutTargetMission
from ..components.perf_current_rms_3_phases import PerformancesCurrentRMS3Phases
from ..components.perf_torque import PerformancesTorque
from ..components.perf_shaft_power_in import PerformancesShaftPowerIn
from ..components.perf_active_power import PerformancesActivePower
from ..components.perf_apparent_power import PerformancesApparentPower
from ..components.perf_voltage_peak import PerformancesVoltagePeak
from ..components.perf_maximum import PerformancesMaximum

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_simple_generator.xml"
NB_POINTS_TEST = 10


def test_rpm_mission():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesRPMMission(
                turbo_generator_id="turbo_generator_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesRPMMission(
            turbo_generator_id="turbo_generator_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("rpm", units="min**-1") == pytest.approx(
        np.full(NB_POINTS_TEST, 2350.0), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_voltage_out_target():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesVoltageOutTargetMission(
                turbo_generator_id="turbo_generator_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesVoltageOutTargetMission(
            turbo_generator_id="turbo_generator_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("voltage_out_target", units="V") == pytest.approx(
        np.full(NB_POINTS_TEST, 800.0), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_rms_current_3_phases():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "ac_current_rms_out_one_phase", val=np.linspace(250, 275, NB_POINTS_TEST), units="A"
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesCurrentRMS3Phases(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("ac_current_rms_out", units="A") == pytest.approx(
        np.linspace(750.0, 825.0, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_apparent_power():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "ac_current_rms_out",
        val=np.linspace(750.0, 825.0, NB_POINTS_TEST),
        units="A",
    )
    ivc.add_output(
        "ac_voltage_rms_out",
        val=np.full(NB_POINTS_TEST, 800.0),
        units="V",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesApparentPower(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("apparent_power", units="kW") == pytest.approx(
        np.linspace(600.0, 660.0, NB_POINTS_TEST),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_active_power():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesActivePower(
                turbo_generator_id="turbo_generator_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "apparent_power",
        val=np.linspace(600.0, 660.0, NB_POINTS_TEST),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesActivePower(
            turbo_generator_id="turbo_generator_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("active_power", units="kW") == pytest.approx(
        np.linspace(570.0, 627.0, NB_POINTS_TEST),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_shaft_power_in():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesShaftPowerIn(
                turbo_generator_id="turbo_generator_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "active_power",
        units="kW",
        val=np.linspace(570.0, 627.0, NB_POINTS_TEST),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesShaftPowerIn(
            turbo_generator_id="turbo_generator_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("shaft_power_in", units="kW") == pytest.approx(
        np.linspace(575.75, 633.33, NB_POINTS_TEST),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_torque():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "shaft_power_in",
        val=np.linspace(575.75, 633.33, NB_POINTS_TEST),
        units="kW",
    )
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2350.0), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesTorque(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("torque_in", units="N*m") == pytest.approx(
        np.linspace(2339.6, 2573.6, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_voltage_peak():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "ac_voltage_rms_out",
        val=np.full(NB_POINTS_TEST, 800.0),
        units="V",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesVoltagePeak(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("ac_voltage_peak_out", units="V") == pytest.approx(
        np.full(NB_POINTS_TEST, 979.79),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_maximum():

    ivc = om.IndepVarComp()

    ivc.add_output(
        "shaft_power_in",
        val=np.linspace(575.75, 633.33, NB_POINTS_TEST),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(
            turbo_generator_id="turbo_generator_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:turbo_generator:turbo_generator_1:shaft_power_max",
            units="W",
        )
        == pytest.approx(633.33e3, rel=1e-2)
    )

    problem.check_partials(compact_print=True)
