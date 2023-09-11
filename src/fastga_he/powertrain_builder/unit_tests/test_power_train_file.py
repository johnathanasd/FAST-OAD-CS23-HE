# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import os.path as pth

import pytest

import networkx as nx
import matplotlib.pyplot as plt

from ..powertrain import FASTGAHEPowerTrainConfigurator
from ..exceptions import FASTGAHESingleSSPCAtEndOfLine, FASTGAHEImpossiblePair

YML_FILE = "sample_power_train_file.yml"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
NB_POINTS_TEST = 10


def test_power_train_file_components_sizing():

    sample_power_train_file_path = pth.join(pth.dirname(__file__), "data", YML_FILE)
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        components_name,
        components_name_id,
        components_type,
        components_om_type,
        components_position,
    ) = power_train_configurator.get_sizing_element_lists()

    # Check that they are not empty
    assert components_name
    assert components_name_id
    assert components_type
    assert components_om_type
    assert components_position


def test_power_train_file_components_performances():

    sample_power_train_file_path = pth.join(pth.dirname(__file__), "data", YML_FILE)
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        components_name,
        components_name_id,
        components_type,
        components_om_type,
        components_options,
        components_connection_outputs,
        components_connection_inputs,
        components_promotes,
        sspc_list,
        sspc_default_state,
    ) = power_train_configurator.get_performances_element_lists()

    # Check that they are not empty
    assert components_name
    assert components_name_id
    assert components_type
    assert components_om_type
    assert components_options
    assert components_connection_outputs
    assert components_connection_inputs
    assert components_promotes
    assert sspc_list
    assert sspc_default_state


def test_power_train_file_components_slipstream():

    sample_power_train_file_path = pth.join(pth.dirname(__file__), "data", YML_FILE)
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        components_name,
        components_name_id,
        components_type,
        components_om_type,
        components_slipstream_promotes,
        components_slipstream_flaps,
        components_slipstream_wing_lift,
    ) = power_train_configurator.get_slipstream_element_lists()

    # Check that they are not empty
    assert components_name
    assert components_name_id
    assert components_type
    assert components_om_type
    assert components_slipstream_promotes
    assert components_slipstream_flaps
    assert components_slipstream_wing_lift


def test_power_train_file_components_performances_sspc_last():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_sspc_last.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        cp_name,
        cp_id,
        cp_type,
        cp_om_type,
        cp_option,
        _,
        _,
        cp_promotes,
        _,
        _,
    ) = power_train_configurator.get_performances_element_lists()

    (components_name, components_name_id, _, _, _,) = power_train_configurator.enforce_sspc_last(
        cp_name,
        cp_id,
        cp_om_type,
        cp_option,
        cp_promotes,
    )

    # Check that they are not empty
    assert components_name == ["dc_bus_1", "dc_line_1", "dc_bus_2", "dc_sspc_1", "dc_sspc_2"]
    assert components_name_id == [
        "dc_bus_id",
        "harness_id",
        "dc_bus_id",
        "dc_sspc_id",
        "dc_sspc_id",
    ]


def test_power_train_file_connections():

    sample_power_train_file_path = pth.join(pth.dirname(__file__), "data", YML_FILE)
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    print("\n")
    power_train_configurator._get_components()
    power_train_configurator._get_connections()

    for om_output, om_input in zip(
        power_train_configurator._components_connection_outputs,
        power_train_configurator._components_connection_inputs,
    ):

        print("[" + om_output + ", " + om_input + "]")


def test_power_train_file_connections_splitter():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    print("\n")
    power_train_configurator._get_components()
    power_train_configurator._get_connections()

    for om_output, om_input in zip(
        power_train_configurator._components_connection_outputs,
        power_train_configurator._components_connection_inputs,
    ):

        print("[" + om_output + ", " + om_input + "]")


def test_power_train_watcher_path():

    sample_power_train_file_path = pth.join(pth.dirname(__file__), "data", YML_FILE)
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    print("\n")
    print(power_train_configurator.get_watcher_file_path())


def test_power_train_logic_check():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_sspc_fail.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    print("\n")
    power_train_configurator._get_components()
    power_train_configurator._get_connections()

    with pytest.raises(FASTGAHESingleSSPCAtEndOfLine):
        power_train_configurator.check_sspc_states({})

    sample_power_train_file_path = pth.join(pth.dirname(__file__), "data", YML_FILE)
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )
    print("\n")
    power_train_configurator._get_components()
    power_train_configurator._get_connections()

    # This state should be correct, let's check it
    actual_state = power_train_configurator.check_sspc_states(
        power_train_configurator._sspc_default_state
    )

    assert actual_state == power_train_configurator._sspc_default_state

    # Check that if we open one end, the other will be opened
    new_state = {"dc_sspc_1": True, "dc_sspc_2": False, "dc_sspc_3": True}
    state_check = {"dc_sspc_1": True, "dc_sspc_2": False, "dc_sspc_3": False}

    assert power_train_configurator.check_sspc_states(new_state) == state_check


def test_distance_from_propulsive_load():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    print("\n")
    power_train_configurator._get_components()
    power_train_configurator._get_connections()

    print("\n")
    distance_from_prop_load, _ = power_train_configurator.get_distance_from_propulsive_load()

    assert distance_from_prop_load["battery_pack_1"] == 9
    assert distance_from_prop_load["ice_1"] == 10

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_tri_prop.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    print("\n")
    power_train_configurator._get_components()
    power_train_configurator._get_connections()

    print("\n")
    distance_from_prop_load, _ = power_train_configurator.get_distance_from_propulsive_load()

    assert distance_from_prop_load["dc_bus_1"] == 3
    assert distance_from_prop_load["dc_bus_2"] == 3
    assert distance_from_prop_load["dc_bus_3"] == 3

    assert distance_from_prop_load["dc_bus_4"] == 7


def test_independent_voltage_subgraph():
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    print("\n")
    sub_graphs = power_train_configurator.get_graphs_connected_voltage()

    # Skip drawing if in GitHub actions
    if not IN_GITHUB_ACTIONS:
        for i, sub_graph in enumerate(sub_graphs):
            fig = plt.figure(figsize=(12, 9), dpi=80)

            nx.draw_kamada_kawai(sub_graph, ax=fig.add_subplot(), with_labels=True)
            fig.savefig("powertrain_builder/unit_tests/outputs/graph_" + str(i + 1) + ".png")


def test_voltage_setter_list():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    voltage_setter_list = power_train_configurator._list_voltage_coherence_to_check()

    # In the .yml that serves for this test, there are 3 subgraphs, one with a DC/DC converter
    # and a rectifier (both sets voltage), one with a generator (which sets voltage) and one with
    # nothing that sets the voltage
    assert [] in voltage_setter_list
    assert ["dc_dc_converter_1_out", "rectifier_1_out"] in voltage_setter_list
    assert ["generator_1_out"] in voltage_setter_list


def test_wing_punctual_mass_identification():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter_position.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        wing_punctual_mass_list,
        wing_punctual_mass_type_list,
        _,
    ) = power_train_configurator.get_wing_punctual_mass_element_list()

    assert "dc_dc_converter_1" in wing_punctual_mass_list
    assert "DC_DC_converter" in wing_punctual_mass_type_list
    assert "rectifier_1" in wing_punctual_mass_list
    assert "rectifier" in wing_punctual_mass_type_list
    assert "generator_1" in wing_punctual_mass_list
    assert "generator" in wing_punctual_mass_type_list
    assert "ice_1" in wing_punctual_mass_list
    assert "ICE" in wing_punctual_mass_type_list

    assert not ("battery_pack_1" in wing_punctual_mass_list)

    assert not ("propeller_1" in wing_punctual_mass_list)
    assert not ("motor_1" in wing_punctual_mass_list)
    assert not ("inverter_1" in wing_punctual_mass_list)
    assert not ("dc_sspc_1" in wing_punctual_mass_list)
    assert not ("dc_bus_1" in wing_punctual_mass_list)
    assert not ("dc_sspc_2" in wing_punctual_mass_list)
    assert not ("dc_line_1" in wing_punctual_mass_list)
    assert not ("dc_sspc_3" in wing_punctual_mass_list)
    assert not ("dc_splitter_1" in wing_punctual_mass_list)


def test_wing_distributed_mass_identification():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter_position.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        wing_distributed_mass_list,
        wing_distributed_mass_type_list,
        _,
    ) = power_train_configurator.get_wing_distributed_mass_element_list()

    assert len(wing_distributed_mass_list) == 1
    assert "battery_pack_1" in wing_distributed_mass_list
    assert "battery_pack" in wing_distributed_mass_type_list


def test_no_distributed_mass():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_propeller_symmetrical.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        wing_distributed_mass_list,
        _,
        pairs_list,
    ) = power_train_configurator.get_wing_distributed_mass_element_list()

    assert not wing_distributed_mass_list
    assert not pairs_list


def test_wing_punctual_mass_symmetrical():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_propeller_symmetrical.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        wing_punctual_mass_list,
        _,
        symmetrical_pair_list,
    ) = power_train_configurator.get_wing_punctual_mass_element_list()

    assert "propeller_1" in wing_punctual_mass_list
    assert "propeller_2" in wing_punctual_mass_list

    assert ["propeller_1", "propeller_2"] in symmetrical_pair_list
    assert ["propeller_2", "propeller_1"] not in symmetrical_pair_list


def test_wing_distributed_mass_symmetrical():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_battery_symmetrical.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        wing_distributed_mass_list,
        _,
        symmetrical_pair_list,
    ) = power_train_configurator.get_wing_distributed_mass_element_list()

    assert "propeller_1" not in wing_distributed_mass_list
    assert "propeller_2" not in wing_distributed_mass_list

    assert "battery_pack_1" in wing_distributed_mass_list
    assert "battery_pack_2" in wing_distributed_mass_list

    assert ["propeller_1", "propeller_2"] not in symmetrical_pair_list
    assert ["battery_pack_1", "battery_pack_2"] in symmetrical_pair_list
    assert ["battery_pack_2", "battery_pack_1"] not in symmetrical_pair_list


def test_bad_pair():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_bad_pair.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    with pytest.raises(FASTGAHEImpossiblePair) as e_info:

        # This should prompt the error
        (
            _,
            _,
            _,
        ) = power_train_configurator.get_wing_punctual_mass_element_list()

    assert (
        e_info.value.args[0]
        == "Cannot pair propeller_1 with propeller_3 because propeller_3 does not exist. Valid pair choice are among the following list: propeller_1, propeller_2. \nBest regards."
    )


def test_independent_subgraph():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_tri_prop_two_chainz.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    print("\n")
    sub_graphs = power_train_configurator.get_independent_sub_propulsion_chain()

    assert len(sub_graphs) == 2


def test_mass_variation_identification():

    # Mass should vary
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    assert power_train_configurator.will_aircraft_mass_vary()

    # Mass shouldn't vary
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_tri_prop.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    assert not power_train_configurator.will_aircraft_mass_vary()


def test_identification_unconsumable_source():

    # Mass should vary but there also is an unconsumable energy source
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    assert power_train_configurator.will_aircraft_mass_vary()
    assert power_train_configurator.has_fuel_non_consumable_energy_source()

    # Mass shouldn't vary and there is an unconsumable energy source
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_tri_prop.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    assert not power_train_configurator.will_aircraft_mass_vary()
    assert power_train_configurator.has_fuel_non_consumable_energy_source()

    # Mass should vary and there is no unconsumable energy sources
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_full_turbo.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    assert power_train_configurator.will_aircraft_mass_vary()
    assert not power_train_configurator.has_fuel_non_consumable_energy_source()


def test_get_power_on_each_node():

    # Very simple power train
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    propulsive_power_dict = {"propeller_1": 50e3}

    power_at_each_node = power_train_configurator.get_current_to_set(
        inputs=None, number_of_points=10, propulsive_power_dict=propulsive_power_dict
    )[0][0]

    # All values should be unique
    assert len(power_at_each_node) == len(set(power_at_each_node.values()))

    # With a splitter
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    propulsive_power_dict = {"propeller_1": 50e3}

    power_at_each_node = power_train_configurator.get_current_to_set(
        inputs=None, number_of_points=10, propulsive_power_dict=propulsive_power_dict
    )[0][0]

    # Since there is a splitter we will check that the two input are indeed created
    assert "dc_splitter_1_in_1" in list(power_at_each_node.keys())
    assert "dc_splitter_1_in_2" in list(power_at_each_node.keys())

    # We will also check that the DC/DC converter has the same value in the dict as the rectifier
    assert power_at_each_node["dc_dc_converter_1_out"] == power_at_each_node["rectifier_1_out"]

    # Then we try with a bus with uniform input
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_tri_prop.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    propulsive_power_dict = {"propeller_1": 50e3, "propeller_2": 25e3, "propeller_3": 25e3}

    power_at_each_node = power_train_configurator.get_current_to_set(
        inputs=None, number_of_points=10, propulsive_power_dict=propulsive_power_dict
    )[0][0]

    assert power_at_each_node["dc_sspc_3_3_in"] == power_at_each_node["dc_sspc_2_3_in"]
    assert power_at_each_node["dc_sspc_1_3_in"] == power_at_each_node["dc_sspc_2_3_in"]

    assert power_at_each_node["dc_bus_4_out"] == power_at_each_node["dc_sspc_1_3_in"] + 1

    # Then we try with a bus with a non-uniform input
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_tri_prop_shorter_nose.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    propulsive_power_dict = {"propeller_1": 50e3, "propeller_2": 25e3, "propeller_3": 25e3}

    power_at_each_node = power_train_configurator.get_current_to_set(
        inputs=None, number_of_points=10, propulsive_power_dict=propulsive_power_dict
    )[0][0]

    assert power_at_each_node["dc_sspc_3_3_in"] == power_at_each_node["dc_sspc_1_3_in"]
    assert power_at_each_node["dc_sspc_3_3_in"] < power_at_each_node["dc_sspc_2_3_in"]

    assert power_at_each_node["dc_bus_4_out"] == power_at_each_node["dc_sspc_2_3_in"] + 1

    # Then we try with a bus and a splitter
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_spliter_and_bus.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    power_at_each_node = power_train_configurator.get_current_to_set(
        inputs=None, number_of_points=10, propulsive_power_dict=propulsive_power_dict
    )[0][0]

    # Assert that they exist and have the same value
    assert power_at_each_node["battery_pack_2_out"] == power_at_each_node["battery_pack_1_out"]

    # Assert that they are properly order

    assert power_at_each_node["dc_bus_0_in"] < power_at_each_node["dc_splitter_0_out"]

    # Test with multiple independent power train
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_tri_prop_two_chainz.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )
    power_at_each_node = power_train_configurator.get_current_to_set(
        inputs=None, number_of_points=10, propulsive_power_dict=propulsive_power_dict
    )

    # Two independent subgraph
    assert len(power_at_each_node) == 2

    first_power_dict = power_at_each_node[0][0]
    second_power_dict = power_at_each_node[1][0]

    # All values should be unique in the second dict
    assert len(second_power_dict) == len(set(second_power_dict.values()))

    assert "propeller_1_out" in list(first_power_dict.keys())
    assert "propeller_3_out" in list(first_power_dict.keys())

    assert "propeller_2_out" in list(second_power_dict.keys())
