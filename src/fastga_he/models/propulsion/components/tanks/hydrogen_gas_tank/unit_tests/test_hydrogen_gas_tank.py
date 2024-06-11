# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO


import openmdao.api as om
import pytest
import numpy as np

from ..components.sizing_tank_inner_volume import SizingHydrogenGasTankInnerVolume
from ..components.sizing_tank_total_hydrogen_mission import (
    SizingHydrogenGasTankTotalHydrogenMission,
)
from ..components.sizing_tank_unusable_hydrogen import SizingHydrogenGasTankUnusableHydrogen
from ..components.sizing_tank_cg_x import SizingHydrogenGasTankCGX
from ..components.sizing_tank_cg_y import SizingHydrogenGasTankCGY
from ..components.sizing_tank_length import SizingHydrogenGasTankLength
from ..components.sizing_tank_inner_diameter import SizingHydrogenGasTankInnerDiameter
from ..components.sizing_tank_outer_diameter import SizingHydrogenGasTankOuterDiameter
from ..components.sizing_tank_weight import SizingHydrogenGasTankWeight
from ..components.sizing_tank_drag import SizingHydrogenGasTankDrag
from ..components.sizing_tank_wall_thickness import SizingHydrogenGasTankWallThickness

from ..components.cstr_enforce import ConstraintsHydrogenGasTankCapacityEnforce
from ..components.cstr_ensure import ConstraintsHydrogenGasTankCapacityEnsure

from ..components.perf_fuel_mission_consumed import PerformancesHydrogenGasConsumedMission
from ..components.perf_fuel_remaining import PerformancesHydrogenGasRemainingMission

from ..components.sizing_tank import SizingHydrogenGasTank
from ..components.perf_hydrogen_gas_tank import PerformancesHydrogenGasTank

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_tank.xml"
NB_POINTS_TEST = 10


def test_unusable_hydrogen_gas_mission():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingHydrogenGasTankUnusableHydrogen(hydrogen_gas_tank_id="hydrogen_gas_tank_1")
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHydrogenGasTankUnusableHydrogen(hydrogen_gas_tank_id="hydrogen_gas_tank_1"),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:unusable_fuel_mission",
            units="kg",
        )
        == pytest.approx(1.4, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_total_hydrogen_gas_mission():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingHydrogenGasTankTotalHydrogenMission(hydrogen_gas_tank_id="hydrogen_gas_tank_1")
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHydrogenGasTankTotalHydrogenMission(hydrogen_gas_tank_id="hydrogen_gas_tank_1"),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:fuel_total_mission",
            units="kg",
        )
        == pytest.approx(141.4, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_inner_volume_hydrogen_gas_tank():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHydrogenGasTankInnerVolume(hydrogen_gas_tank_id="hydrogen_gas_tank_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHydrogenGasTankInnerVolume(hydrogen_gas_tank_id="hydrogen_gas_tank_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:volume", units="L"
    ) == pytest.approx(393.4, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_tank_cg_x():

    expected_values = [2.88, 2.88, 1.24]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                SizingHydrogenGasTankCGX(
                    hydrogen_gas_tank_id="hydrogen_gas_tank_1", position=option
                )
            ),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingHydrogenGasTankCGX(hydrogen_gas_tank_id="hydrogen_gas_tank_1", position=option),
            ivc,
        )
        assert problem.get_val(
            "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_tank_cg_y():

    expected_values = [1.85, 1.85, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                SizingHydrogenGasTankCGY(
                    hydrogen_gas_tank_id="hydrogen_gas_tank_1", position=option
                )
            ),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingHydrogenGasTankCGY(hydrogen_gas_tank_id="hydrogen_gas_tank_1", position=option),
            ivc,
        )
        assert problem.get_val(
            "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:CG:y", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_tank_length():

    expected_values = [0.467, 2.68, 1.477]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                SizingHydrogenGasTankLength(
                    hydrogen_gas_tank_id="hydrogen_gas_tank_1", position=option
                )
            ),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingHydrogenGasTankLength(
                hydrogen_gas_tank_id="hydrogen_gas_tank_1", position=option
            ),
            ivc,
        )
        assert (
            problem.get_val(
                "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:dimension:length",
                units="m",
            )
            == pytest.approx(expected_value, rel=1e-2)
        )

        if option == "inside_the_wing":
            assert (
                problem.get_val(
                    "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:dimension:ref_chord",
                    units="m",
                )
                == pytest.approx(0.9346, rel=1e-2)
            )

        problem.check_partials(compact_print=True, step=1e-7)


def test_tank_outer_diameter():

    expected_values = [0.127, 0.383, 0.133]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                SizingHydrogenGasTankOuterDiameter(
                    hydrogen_gas_tank_id="hydrogen_gas_tank_1", position=option
                )
            ),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingHydrogenGasTankOuterDiameter(
                hydrogen_gas_tank_id="hydrogen_gas_tank_1", position=option
            ),
            ivc,
        )
        assert (
            problem.get_val(
                "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:dimension:outer_diameter",
                units="m",
            )
            == pytest.approx(expected_value, rel=1e-2)
        )

        problem.check_partials(compact_print=True, step=1e-7)


def test_tank_inner_diameter():

    ivc = get_indep_var_comp(
        list_inputs(SizingHydrogenGasTankInnerDiameter(hydrogen_gas_tank_id="hydrogen_gas_tank_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHydrogenGasTankInnerDiameter(hydrogen_gas_tank_id="hydrogen_gas_tank_1"),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:dimension:inner_diameter",
            units="m",
        )
        == pytest.approx(6.62, rel=1e-2)
    )

    problem.check_partials(compact_print=True, step=1e-7)


def test_tank_wall_thickness():

    ivc = get_indep_var_comp(
        list_inputs(SizingHydrogenGasTankWallThickness(hydrogen_gas_tank_id="hydrogen_gas_tank_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHydrogenGasTankWallThickness(hydrogen_gas_tank_id="hydrogen_gas_tank_1"),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:dimension:wall_thickness",
            units="m",
        )
        == pytest.approx(6.62, rel=1e-2)
    )

    problem.check_partials(compact_print=True, step=1e-7)


def test_hydrogen_gas_tank_weight():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHydrogenGasTankWeight(hydrogen_gas_tank_id="hydrogen_gas_tank_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHydrogenGasTankWeight(hydrogen_gas_tank_id="hydrogen_gas_tank_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:mass", units="kg"
    ) == pytest.approx(1.4, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_hydrogen_gas_tank_drag():

    expected_ls_drag = [0.0, 0.00421, 0.0]
    expected_cruise_drag = [0.0, 0.00421, 0.0]

    for option, ls_drag, cruise_drag in zip(
        POSSIBLE_POSITION, expected_ls_drag, expected_cruise_drag
    ):
        # Research independent input value in .xml file
        for ls_option in [True, False]:
            ivc = get_indep_var_comp(
                list_inputs(
                    SizingHydrogenGasTankDrag(
                        hydrogen_gas_tank_id="hydrogen_gas_tank_1",
                        position=option,
                        low_speed_aero=ls_option,
                    )
                ),
                __file__,
                XML_FILE,
            )

            # Run problem and check obtained value(s) is/(are) correct
            problem = run_system(
                SizingHydrogenGasTankDrag(
                    hydrogen_gas_tank_id="hydrogen_gas_tank_1",
                    position=option,
                    low_speed_aero=ls_option,
                ),
                ivc,
            )

            if ls_option:
                assert (
                    problem.get_val(
                        "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:low_speed:CD0",
                    )
                    == pytest.approx(ls_drag, rel=1e-2)
                )
            else:
                assert (
                    problem.get_val(
                        "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:cruise:CD0",
                    )
                    == pytest.approx(cruise_drag, rel=1e-2)
                )

            problem.check_partials(compact_print=True)


def test_sizing_tank():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHydrogenGasTank(hydrogen_gas_tank_id="hydrogen_gas_tank_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHydrogenGasTank(hydrogen_gas_tank_id="hydrogen_gas_tank_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:mass", units="kg"
    ) == pytest.approx(1.4, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:cruise:CD0"
    ) == pytest.approx(0.0, rel=1e-2)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:dimension:width",
            units="m",
        )
        == pytest.approx(3.33, rel=1e-2)
    )

    problem.check_partials(compact_print=True, step=1e-7)


def test_constraints_enforce_tank_capacity():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            ConstraintsHydrogenGasTankCapacityEnforce(hydrogen_gas_tank_id="hydrogen_gas_tank_1")
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsHydrogenGasTankCapacityEnforce(hydrogen_gas_tank_id="hydrogen_gas_tank_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:capacity", units="kg"
    ) == pytest.approx(141.4, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_ensure_tank_capacity():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            ConstraintsHydrogenGasTankCapacityEnsure(hydrogen_gas_tank_id="hydrogen_gas_tank_1")
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsHydrogenGasTankCapacityEnsure(hydrogen_gas_tank_id="hydrogen_gas_tank_1"),
        ivc,
    )
    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:capacity",
            units="kg",
        )
        == pytest.approx(-141.4, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_hydrogen_gas_consumed_mission():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fuel_consumed_t", val=np.linspace(13.37, 42.0, NB_POINTS_TEST))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHydrogenGasConsumedMission(
            hydrogen_gas_tank_id="hydrogen_gas_tank_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:fuel_consumed_mission",
            units="kg",
        )
        == pytest.approx(276.85, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_hydrogen_gas_remaining_mission():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:fuel_consumed_mission", units="kg",val=140.)
    ivc.add_output("fuel_consumed_t", val=np.full(NB_POINTS_TEST, 14.0))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHydrogenGasRemainingMission(
            hydrogen_gas_tank_id="hydrogen_gas_tank_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    assert problem.get_val("fuel_remaining_t", units="kg") == pytest.approx(
        np.array([140.0, 126.0, 112.0, 98.0, 84.0, 70.0, 56.0, 42.0, 28.0, 14.0]), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_performances_hydrogen_gas_tank():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fuel_consumed_t", val=np.linspace(13.37, 42.0, NB_POINTS_TEST))

    problem = run_system(
        PerformancesHydrogenGasTank(
            hydrogen_gas_tank_id="hydrogen_gas_tank_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    assert problem.get_val("fuel_remaining_t", units="kg") == pytest.approx(
        np.array([276.85, 263.48, 246.93, 227.2, 204.28, 178.19, 148.91, 116.46, 80.82, 42.0]),
        rel=1e-2,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:hydrogen_gas_tank:hydrogen_gas_tank_1:fuel_consumed_mission",
            units="kg",
        )
        == pytest.approx(279.62, rel=1e-2)
    )
