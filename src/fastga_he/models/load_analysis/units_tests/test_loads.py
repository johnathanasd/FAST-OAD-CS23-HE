"""
Test load_analysis module.
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

import numpy as np
import pytest


from ..wing.aerostructural_loads import AerostructuralLoadHE
from ..wing.structural_loads import StructuralLoadsHE
from ..wing.aerodynamic_loads import AerodynamicLoadsHE
from ..wing.loads import WingLoadsHE

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs


XML_FILE = "data.xml"


def test_compute_shear_stress():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerostructuralLoadHE()), __file__, XML_FILE)
    cl_vector_only_prop = [
        0.04,
        0.13,
        0.21,
        0.3,
        0.39,
        0.47,
        0.56,
        0.68,
        0.84,
        1.01,
        1.17,
        1.34,
        1.51,
        1.67,
        1.84,
        2.01,
        2.18,
        2.35,
        2.51,
        2.68,
        2.85,
        3.02,
        3.19,
        3.35,
        3.52,
        3.68,
        3.85,
        4.01,
        4.18,
        4.34,
        4.5,
        4.66,
        4.81,
        4.97,
        5.13,
        5.28,
        5.43,
        5.58,
        5.73,
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
        0.0,
    ]
    y_vector = [
        1.43,
        1.42,
        1.42,
        1.42,
        1.42,
        1.41,
        1.42,
        1.41,
        1.4,
        1.4,
        1.41,
        1.4,
        1.4,
        1.46,
        1.5,
        1.38,
        1.35,
        1.36,
        1.35,
        1.34,
        1.33,
        1.32,
        1.31,
        1.3,
        1.28,
        1.26,
        1.24,
        1.21,
        1.19,
        1.16,
        1.13,
        1.09,
        1.05,
        1.01,
        0.95,
        0.88,
        0.79,
        0.67,
        0.63,
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
        0.0,
    ]
    ivc.add_output(
        "data:aerodynamics:slipstream:wing:cruise:only_prop:CL_vector", cl_vector_only_prop
    )
    ivc.add_output("data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector", y_vector, units="m")
    ivc.add_output(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:velocity", 82.3111, units="m/s"
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(AerostructuralLoadHE(), ivc)
    shear_max_mass_condition = problem.get_val("data:loads:max_shear:mass", units="kg")
    assert shear_max_mass_condition == pytest.approx(1747.15, abs=1e-1)
    shear_max_lf_condition = problem.get_val("data:loads:max_shear:load_factor")
    assert shear_max_lf_condition == pytest.approx(3.8, abs=1e-2)
    lift_shear_diagram = problem.get_val("data:loads:max_shear:lift_shear", units="N")
    lift_root_shear = lift_shear_diagram[0]
    assert lift_root_shear == pytest.approx(142889.41, abs=1)
    weight_shear_diagram = problem.get_val("data:loads:max_shear:weight_shear", units="N")
    weight_root_shear = weight_shear_diagram[0]
    assert weight_root_shear == pytest.approx(-21607.77, abs=1)


def test_compute_root_bending_moment():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerostructuralLoadHE()), __file__, XML_FILE)
    cl_vector_only_prop = [
        0.04,
        0.13,
        0.21,
        0.3,
        0.39,
        0.47,
        0.56,
        0.68,
        0.84,
        1.01,
        1.17,
        1.34,
        1.51,
        1.67,
        1.84,
        2.01,
        2.18,
        2.35,
        2.51,
        2.68,
        2.85,
        3.02,
        3.19,
        3.35,
        3.52,
        3.68,
        3.85,
        4.01,
        4.18,
        4.34,
        4.5,
        4.66,
        4.81,
        4.97,
        5.13,
        5.28,
        5.43,
        5.58,
        5.73,
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
        0.0,
    ]
    y_vector = [
        1.43,
        1.42,
        1.42,
        1.42,
        1.42,
        1.41,
        1.42,
        1.41,
        1.4,
        1.4,
        1.41,
        1.4,
        1.4,
        1.46,
        1.5,
        1.38,
        1.35,
        1.36,
        1.35,
        1.34,
        1.33,
        1.32,
        1.31,
        1.3,
        1.28,
        1.26,
        1.24,
        1.21,
        1.19,
        1.16,
        1.13,
        1.09,
        1.05,
        1.01,
        0.95,
        0.88,
        0.79,
        0.67,
        0.63,
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
        0.0,
    ]
    ivc.add_output(
        "data:aerodynamics:slipstream:wing:cruise:only_prop:CL_vector", cl_vector_only_prop
    )
    ivc.add_output("data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector", y_vector, units="m")
    ivc.add_output(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:velocity", 82.3111, units="m/s"
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(AerostructuralLoadHE(), ivc)
    max_rbm_mass_condition = problem.get_val("data:loads:max_rbm:mass", units="kg")
    assert max_rbm_mass_condition == pytest.approx(1559.71, abs=1e-1)
    max_rbm_lf_condition = problem.get_val("data:loads:max_rbm:load_factor")
    assert max_rbm_lf_condition == pytest.approx(3.8, abs=1e-2)
    lift_rbm_diagram = problem.get_val("data:loads:max_rbm:lift_rbm", units="N*m")
    lift_rbm = lift_rbm_diagram[0]
    assert lift_rbm == pytest.approx(369153, abs=1)
    weight_rbm_diagram = problem.get_val("data:loads:max_rbm:weight_rbm", units="N*m")
    weight_rbm = weight_rbm_diagram[0]
    assert weight_rbm == pytest.approx(-34854, abs=1)


def test_compute_mass_distribution():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(StructuralLoadsHE()), __file__, XML_FILE)
    load_factor_shear = 4.0
    ivc.add_output("data:loads:max_shear:load_factor", load_factor_shear)
    load_factor_rbm = 4.0
    ivc.add_output("data:loads:max_rbm:load_factor", load_factor_rbm)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(StructuralLoadsHE(), ivc)
    point_mass_array = problem.get_val(
        "data:loads:structure:ultimate:force_distribution:point_mass", units="N/m"
    )
    point_mass_result = np.array(
        [
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -24546.6436177,
            -24546.6436177,
            -24546.6436177,
            -24546.6436177,
            -24546.6436177,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -145568.22319297,
            -145568.22319297,
            -145568.22319297,
            -145568.22319297,
            -145568.22319297,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
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
            0.0,
            0.0,
        ]
    )
    assert point_mass_array == pytest.approx(point_mass_result, abs=1e-1)
    wing_mass_array = problem.get_val(
        "data:loads:structure:ultimate:force_distribution:wing", units="N/m"
    )
    wing_mass_result = np.array(
        [
            -591.87840868,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -589.0481754,
            -591.87840868,
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
            0.0,
            0.0,
        ]
    )

    assert wing_mass_array == pytest.approx(wing_mass_result, abs=1e-1)
    fuel_mass_array = problem.get_val(
        "data:loads:structure:ultimate:force_distribution:fuel", units="N/m"
    )
    fuel_mass_result = np.array(
        [
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -1312.61537039,
            -1268.00134823,
            -1223.38732608,
            -1178.77330392,
            -1146.90614523,
            -1127.78585002,
            -1108.66555481,
            -1089.5452596,
            -1070.42496439,
            -0.0,
            -0.0,
            -0.0,
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
            0.0,
            0.0,
        ]
    )

    assert fuel_mass_array == pytest.approx(fuel_mass_result, abs=1e-1)

    distributed_mass_array = problem.get_val(
        "data:loads:structure:ultimate:force_distribution:distributed_mass", units="N/m"
    )
    distributed_mass_result = np.zeros_like(distributed_mass_array)
    assert np.max(np.abs(distributed_mass_result - distributed_mass_array)) <= 1e-1


def test_compute_structure_shear():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(StructuralLoadsHE()), __file__, XML_FILE)
    ivc.add_output("data:loads:max_shear:load_factor", 4.0)
    ivc.add_output("data:loads:max_rbm:load_factor", 4.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(StructuralLoadsHE(), ivc)
    point_mass_array = problem.get_val("data:loads:structure:ultimate:shear:point_mass", units="N")
    point_mass_result = np.array(
        [
            -8067.287414,
            -8067.287414,
            -8067.287414,
            -8067.287414,
            -8067.287414,
            -8055.01409219,
            -7770.1344051,
            -7485.25471801,
            -7200.37503093,
            -6915.49534384,
            -6903.22202203,
            -6903.22202203,
            -6903.22202203,
            -6903.22202203,
            -6830.43791043,
            -5141.02446073,
            -3451.61101102,
            -1762.19756131,
            -72.7841116,
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
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert point_mass_array == pytest.approx(point_mass_result, abs=1e-1)
    wing_mass_array = problem.get_val("data:loads:structure:ultimate:shear:wing", units="N")
    wing_mass_result = np.array(
        [
            -3418.51620139,
            -3342.31619729,
            -3190.28143412,
            -3038.24667094,
            -2966.48137654,
            -2965.89232836,
            -2959.05604302,
            -2952.21975768,
            -2945.38347235,
            -2938.54718701,
            -2937.95813883,
            -2814.94461094,
            -2520.37525412,
            -2270.4266901,
            -2269.83764192,
            -2263.00135658,
            -2256.16507124,
            -2249.3287859,
            -2242.49250056,
            -2241.90345239,
            -2225.80589729,
            -1931.23654047,
            -1636.66718364,
            -1342.09782681,
            -1047.52846999,
            -837.12178654,
            -710.87777647,
            -584.6337664,
            -458.38975634,
            -332.14574627,
            -205.9017362,
            -79.65772613,
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
            0.0,
            0.0,
            0.0,
        ]
    )

    assert wing_mass_array == pytest.approx(wing_mass_result, abs=1e-1)
    fuel_mass_array = problem.get_val("data:loads:structure:ultimate:shear:fuel", units="N")
    fuel_mass_result = np.array(
        [
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3677.53356,
            -3349.32927929,
            -2704.07593774,
            -2081.13303591,
            -1480.50057381,
            -1065.13683949,
            -821.38238005,
            -581.72575648,
            -346.16696878,
            -114.70601695,
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
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert fuel_mass_array == pytest.approx(fuel_mass_result, abs=1e-1)

    distributed_mass_array = problem.get_val(
        "data:loads:structure:ultimate:shear:distributed_mass", units="N"
    )
    distributed_mass_result = np.zeros_like(distributed_mass_array)
    assert np.max(np.abs(distributed_mass_result - distributed_mass_array)) <= 1e-1


def test_compute_structure_bending():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(StructuralLoadsHE()), __file__, XML_FILE)
    ivc.add_output("data:loads:max_shear:load_factor", 4.0)
    ivc.add_output("data:loads:max_rbm:load_factor", 4.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(StructuralLoadsHE(), ivc)
    point_mass_array = problem.get_val(
        "data:loads:structure:ultimate:root_bending:point_mass", units="N*m"
    )
    point_mass_result = np.array(
        [
            -14540.90609382,
            -13499.81286692,
            -11417.62641314,
            -9335.43995935,
            -8352.58103458,
            -8344.51374716,
            -8252.68320157,
            -8164.15886913,
            -8078.94074984,
            -7997.02884372,
            -7990.1256217,
            -6548.49532693,
            -3096.35379422,
            -167.13594142,
            -160.2327194,
            -90.76443471,
            -40.90288658,
            -10.64807501,
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
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    assert point_mass_array == pytest.approx(point_mass_result, abs=1e-1)
    wing_mass_array = problem.get_val(
        "data:loads:structure:ultimate:root_bending:wing", units="N*m"
    )
    wing_mass_result = np.array(
        [
            -9918.54733458,
            -9482.31230723,
            -8639.27265845,
            -7835.47355091,
            -7469.6876151,
            -7466.72142825,
            -7432.33999785,
            -7398.03790697,
            -7363.8151556,
            -7329.67174376,
            -7326.73349109,
            -6726.0314667,
            -5391.99648726,
            -4375.56513539,
            -4373.29500323,
            -4346.99173799,
            -4320.76781227,
            -4294.62322606,
            -4268.55797937,
            -4266.3157814,
            -4205.26882427,
            -3165.84847774,
            -2273.73544766,
            -1528.92973404,
            -931.43133688,
            -594.83492041,
            -428.9523317,
            -290.12618887,
            -178.35649192,
            -93.64324085,
            -35.98643566,
            -5.38607634,
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
            0.0,
            0.0,
            0.0,
        ]
    )

    assert wing_mass_array == pytest.approx(wing_mass_result, abs=1e-1)
    fuel_mass_array = problem.get_val(
        "data:loads:structure:ultimate:root_bending:fuel", units="N*m"
    )
    fuel_mass_result = np.array(
        [
            -13830.5481984,
            -13355.95803141,
            -12406.77769742,
            -11457.59736344,
            -11009.55374046,
            -11005.8762069,
            -10963.19604964,
            -10920.51589238,
            -10877.83573511,
            -10835.15557785,
            -10831.47804429,
            -10063.4824948,
            -8224.4330915,
            -6663.95931218,
            -6660.28177862,
            -6617.60162136,
            -6574.9214641,
            -6532.24130684,
            -6489.56114958,
            -6485.88361602,
            -6385.3836882,
            -4546.33428491,
            -3035.53960818,
            -1841.84272357,
            -954.08669664,
            -500.45520182,
            -298.51663578,
            -148.38007524,
            -49.16727753,
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
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert fuel_mass_array == pytest.approx(fuel_mass_result, abs=1e-1)

    distributed_mass_array = problem.get_val(
        "data:loads:structure:ultimate:root_bending:distributed_mass", units="N*m"
    )
    distributed_mass_result = np.zeros_like(distributed_mass_array)
    assert np.max(np.abs(distributed_mass_result - distributed_mass_array)) <= 1e-1


def test_non_nil_distributed_masses():
    # Research independent input value in .xml file
    input_list = list_inputs(StructuralLoadsHE())
    input_list.remove("data:weight:airframe:wing:distributed_mass:y_ratio_start")
    input_list.remove("data:weight:airframe:wing:distributed_mass:y_ratio_end")
    input_list.remove("data:weight:airframe:wing:distributed_mass:start_chord")
    input_list.remove("data:weight:airframe:wing:distributed_mass:chord_slope")
    input_list.remove("data:weight:airframe:wing:distributed_mass:mass")

    ivc = get_indep_var_comp(input_list, __file__, XML_FILE)
    ivc.add_output("data:loads:max_shear:load_factor", 4.0)
    ivc.add_output("data:loads:max_rbm:load_factor", 4.0)

    # We add two distributed mass, one going from 0.15 to 0.3, without slope and weighing 50 kg,
    # and one going from 0.45 to 0.6, with a slight negative slope and weighing 30 kg
    ivc.add_output("data:weight:airframe:wing:distributed_mass:y_ratio_start", val=[0.15, 0.45])
    ivc.add_output("data:weight:airframe:wing:distributed_mass:y_ratio_end", val=[0.45, 0.6])
    ivc.add_output(
        "data:weight:airframe:wing:distributed_mass:start_chord",
        val=[1.4541595355959134, 1.4541595355959134],
        units="m",
    )
    ivc.add_output(
        "data:weight:airframe:wing:distributed_mass:chord_slope",
        val=[0.0, -0.1],
    )
    ivc.add_output("data:weight:airframe:wing:distributed_mass:mass", val=[50.0, 30.0], units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(StructuralLoadsHE(), ivc)

    distributed_mass_array = problem.get_val(
        "data:loads:structure:ultimate:force_distribution:distributed_mass", units="N/m"
    )
    distributed_mass_result = np.array(
        [
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -1057.86090204,
            -1057.86090204,
            -1057.86090204,
            -1057.86090204,
            -1057.86090204,
            -1057.86090204,
            -1057.86090204,
            -1057.86090204,
            -1057.86090204,
            -1057.86090204,
            -1057.86090204,
            -2354.03817813,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
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
            0.0,
            0.0,
        ]
    )
    assert distributed_mass_array == pytest.approx(distributed_mass_result, rel=1e-2)

    distributed_mass_shear_array = problem.get_val(
        "data:loads:structure:ultimate:shear:distributed_mass", units="N"
    )
    distributed_mass_shear_result = np.array(
        [
            -3139.2,
            -3139.2,
            -3139.2,
            -3139.2,
            -3139.2,
            -3139.2,
            -3139.2,
            -3139.2,
            -3139.2,
            -3139.2,
            -3139.2,
            -3028.74112158,
            -2499.72937256,
            -2050.85129358,
            -2049.79343268,
            -2037.5162722,
            -2025.23911173,
            -2012.96195126,
            -2000.68479079,
            -1999.62692988,
            -1970.71762354,
            -1441.70587451,
            -588.6,
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
            0.0,
            0.0,
        ]
    )
    assert distributed_mass_shear_array == pytest.approx(distributed_mass_shear_result, rel=1e-2)

    distributed_mass_bending_array = problem.get_val(
        "data:loads:structure:ultimate:root_bending:distributed_mass", units="N*m"
    )
    distributed_mass_bending_result = np.array(
        [
            -7157.66709014,
            -6752.54952209,
            -5942.31438601,
            -5132.07924992,
            -4749.62222585,
            -4746.48302585,
            -4710.05057799,
            -4673.61813014,
            -4637.18568228,
            -4600.75323442,
            -4597.61403442,
            -3942.04105605,
            -2559.71099739,
            -1594.24573456,
            -1592.1954122,
            -1568.47747487,
            -1544.90202194,
            -1521.4690534,
            -1498.17856926,
            -1496.1784134,
            -1441.9274685,
            -588.69046937,
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
            0.0,
            0.0,
            0.0,
        ]
    )
    assert distributed_mass_bending_array == pytest.approx(
        distributed_mass_bending_result, rel=1e-2
    )


def test_compute_lift_distribution():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerodynamicLoadsHE()), __file__, XML_FILE)
    ivc.add_output("data:loads:max_shear:load_factor", 4.0)
    ivc.add_output("data:loads:max_shear:mass", 1747.0, units="kg")
    ivc.add_output("data:loads:max_rbm:load_factor", 4.0)
    ivc.add_output("data:loads:max_rbm:mass", 1568.0, units="kg")
    cl_vector_only_prop = [
        0.04,
        0.13,
        0.21,
        0.3,
        0.39,
        0.47,
        0.56,
        0.68,
        0.84,
        1.01,
        1.17,
        1.34,
        1.51,
        1.67,
        1.84,
        2.01,
        2.18,
        2.35,
        2.51,
        2.68,
        2.85,
        3.02,
        3.19,
        3.35,
        3.52,
        3.68,
        3.85,
        4.01,
        4.18,
        4.34,
        4.5,
        4.66,
        4.81,
        4.97,
        5.13,
        5.28,
        5.43,
        5.58,
        5.73,
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
        0.0,
    ]
    y_vector = [
        1.43,
        1.42,
        1.42,
        1.42,
        1.42,
        1.41,
        1.42,
        1.41,
        1.4,
        1.4,
        1.41,
        1.4,
        1.4,
        1.46,
        1.5,
        1.38,
        1.35,
        1.36,
        1.35,
        1.34,
        1.33,
        1.32,
        1.31,
        1.3,
        1.28,
        1.26,
        1.24,
        1.21,
        1.19,
        1.16,
        1.13,
        1.09,
        1.05,
        1.01,
        0.95,
        0.88,
        0.79,
        0.67,
        0.63,
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
        0.0,
    ]
    ivc.add_output(
        "data:aerodynamics:slipstream:wing:cruise:only_prop:CL_vector", cl_vector_only_prop
    )
    ivc.add_output("data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector", y_vector, units="m")
    ivc.add_output(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:velocity", 82.3111, units="m/s"
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(AerodynamicLoadsHE(), ivc)
    lift_array = problem.get_val("data:loads:aerodynamic:ultimate:force_distribution", units="N/m")
    lift_result = np.array(
        [
            6727.59259473,
            6695.42271624,
            6685.44026375,
            6816.29714678,
            7336.55818301,
            7339.40232362,
            7372.41041626,
            7407.34031907,
            7457.50522008,
            7507.67012108,
            7511.99257721,
            8671.59604842,
            31594.73004791,
            28954.36210253,
            28948.13960661,
            28875.92351389,
            28803.70742117,
            28731.49132845,
            28659.27523573,
            28653.0527398,
            28483.00387936,
            25335.0582747,
            22148.37271376,
            18914.70126684,
            15621.72989116,
            13196.30076336,
            11661.97256886,
            10099.7748881,
            8484.62970505,
            6785.47570978,
            4945.07012295,
            2815.90482822,
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
            0.0,
            0.0,
            0.0,
        ]
    )

    assert np.max(np.abs(lift_array - lift_result)) <= 1e-1


def test_load_group():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(WingLoadsHE()), __file__, XML_FILE)
    cl_vector_only_prop = [
        0.04,
        0.13,
        0.21,
        0.3,
        0.39,
        0.47,
        0.56,
        0.68,
        0.84,
        1.01,
        1.17,
        1.34,
        1.51,
        1.67,
        1.84,
        2.01,
        2.18,
        2.35,
        2.51,
        2.68,
        2.85,
        3.02,
        3.19,
        3.35,
        3.52,
        3.68,
        3.85,
        4.01,
        4.18,
        4.34,
        4.5,
        4.66,
        4.81,
        4.97,
        5.13,
        5.28,
        5.43,
        5.58,
        5.73,
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
        0.0,
    ]
    y_vector = [
        1.43,
        1.42,
        1.42,
        1.42,
        1.42,
        1.41,
        1.42,
        1.41,
        1.4,
        1.4,
        1.41,
        1.4,
        1.4,
        1.46,
        1.5,
        1.38,
        1.35,
        1.36,
        1.35,
        1.34,
        1.33,
        1.32,
        1.31,
        1.3,
        1.28,
        1.26,
        1.24,
        1.21,
        1.19,
        1.16,
        1.13,
        1.09,
        1.05,
        1.01,
        0.95,
        0.88,
        0.79,
        0.67,
        0.63,
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
        0.0,
    ]
    ivc.add_output(
        "data:aerodynamics:slipstream:wing:cruise:only_prop:CL_vector", cl_vector_only_prop
    )
    ivc.add_output("data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector", y_vector, units="m")
    ivc.add_output(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:velocity", 82.3111, units="m/s"
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(WingLoadsHE(), ivc)

    lift_shear_diagram = problem.get_val("data:loads:max_shear:lift_shear", units="N")
    lift_root_shear = lift_shear_diagram[0]
    assert lift_root_shear == pytest.approx(142889.45, abs=1)
