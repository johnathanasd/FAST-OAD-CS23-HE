# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import pytest
import numpy as np

from ..components.perf_voltage_out_target import PerformancesVoltageOutTargetMission
from ..components.perf_efficiency import PerformancesEfficiencyMission
from ..components.perf_maximum import PerformancesMaximum

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from ..constants import POSSIBLE_POSITION

XML_FILE = "sample_rectifier.xml"
NB_POINTS_TEST = 10


def test_voltage_out_target_mission():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesVoltageOutTargetMission(
                rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        PerformancesVoltageOutTargetMission(
            rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("voltage_out_target", units="V") == pytest.approx(
        np.full(NB_POINTS_TEST, 800), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc2 = om.IndepVarComp()
    ivc2.add_output(
        "data:propulsion:he_power_train:rectifier:rectifier_1:voltage_out_target_mission",
        val=[850.0, 800.0, 700.0, 850.0, 800.0, 700.0, 850.0, 800.0, 700.0, 420.0],
        units="V",
    )

    problem2 = run_system(
        PerformancesVoltageOutTargetMission(
            rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
        ),
        ivc2,
    )

    assert problem2.get_val("voltage_out_target", units="V") == pytest.approx(
        np.array([850.0, 800.0, 700.0, 850.0, 800.0, 700.0, 850.0, 800.0, 700.0, 420.0]),
        rel=1e-2,
    )

    problem2.check_partials(compact_print=True)


def test_efficiency():
    # Will eventually disappear

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesEfficiencyMission(
                rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        PerformancesEfficiencyMission(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("efficiency") == pytest.approx(np.full(NB_POINTS_TEST, 0.98), rel=1e-2)

    problem.check_partials(compact_print=True)

    ivc2 = om.IndepVarComp()
    ivc2.add_output(
        "data:propulsion:he_power_train:rectifier:rectifier_1:efficiency",
        val=[0.98, 0.97, 0.96, 0.95, 0.94, 0.94, 0.95, 0.96, 0.97, 0.98],
    )

    problem2 = run_system(
        PerformancesEfficiencyMission(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST),
        ivc2,
    )

    assert problem2.get_val("efficiency") == pytest.approx(
        np.array([0.98, 0.97, 0.96, 0.95, 0.94, 0.94, 0.95, 0.96, 0.97, 0.98]),
        rel=1e-2,
    )

    problem2.check_partials(compact_print=True)


def test_maximum():

    ivc = om.IndepVarComp()
    ivc.add_output("dc_current_out", units="A", val=np.linspace(300.0, 280.0, NB_POINTS_TEST))
    ivc.add_output("dc_voltage_out", units="V", val=np.linspace(500.0, 480.0, NB_POINTS_TEST))
    ivc.add_output("ac_voltage_peak_in", units="V", val=np.linspace(500.0, 480.0, NB_POINTS_TEST))
    ivc.add_output(
        "ac_current_rms_in_one_phase", units="A", val=np.linspace(133.0, 120.0, NB_POINTS_TEST)
    )

    problem = run_system(
        PerformancesMaximum(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:current_ac_max", units="A"
    ) == pytest.approx(133.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:voltage_ac_max", units="V"
    ) == pytest.approx(500.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:current_dc_max", units="A"
    ) == pytest.approx(300.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:voltage_dc_max", units="V"
    ) == pytest.approx(500.0, rel=1e-2)

    problem.check_partials(compact_print=True)
