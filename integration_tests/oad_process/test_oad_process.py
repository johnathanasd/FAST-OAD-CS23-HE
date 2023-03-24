import os
import os.path as pth
import logging

import pytest

import fastoad.api as oad

from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")


def test_pipistrel_like():

    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)

    # Define used files depending on options
    xml_file_name = "full_sizing.xml"
    process_file_name = "full_sizing.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=1000.0)
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        606.0, rel=1e-2
    )

    problem.write_outputs()


def test_fuel_and_battery():

    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)

    # Define used files depending on options
    xml_file_name = "full_sizing_fuel_and_battery.xml"
    process_file_name = "full_sizing_fuel_and_battery.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()
    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=1200.0)
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        722.0, rel=1e-2
    )
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(19.32, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(126.0, rel=1e-2)


def test_sizing_sr22():

    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)

    # Define used files depending on options
    xml_file_name = "input_sr22.xml"
    process_file_name = "full_sizing_fuel.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()

    # om.n2(problem, show_browser=True)

    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=1000.0)
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        1662.0, rel=1e-2
    )
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(
        257.90, rel=1e-2
    )

    problem.write_outputs()
