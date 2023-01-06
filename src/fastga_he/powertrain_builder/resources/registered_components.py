# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

ID = "id"
# CN is used to translate the ID to the prefix used in the OpenMDAO names for the components,
# e.g: PerformancesPropeller = Performances + Propeller
CN = "OM_components_name"
# CN_ID is used to store the id of the components in question, "battery_pack_id", "propeller_id"
CN_ID = "OM_components_id"
# CT is used to store the type of the components in question, "battery_pack", "propeller"
CT = "components_type"
# CTC is used to store the class of the components type type, like "propulsor", "connector", ...
CTC = "components_type_class"
ATT = "attributes"
# The IN and OUT field contain the input and output in the system sense of the term,
# meaning the output of the prop is the propulsive power while its input is the mechanical power.
# In those inputs, there will be tuple of two element: the first filled when the system
# input/output is an openmdao input the other if the opposite
IN = "inputs"
OUT = "outputs"
# The PT field contains the variables that must be promoted from aircraft level for the component
# to work
PT = "promoted_variables"

PROPELLER = {
    ID: "fastga_he.pt_component.propeller",
    CN: "Propeller",
    CN_ID: "propeller_id",
    CT: "propeller",
    ATT: None,
    PT: ["true_airspeed", "altitude"],
    IN: [(None, "rpm"), (None, "shaft_power_in")],
    OUT: None,
    CTC: "propulsor",
}
PMSM = {
    ID: "fastga_he.pt_component.pmsm",
    CN: "PMSM",
    CN_ID: "motor_id",
    CT: "PMSM",
    ATT: None,
    PT: ["settings:*"],
    IN: [
        (None, "ac_current_rms_in_one_phase"),
        (None, "ac_voltage_peak_in"),
        (None, "ac_voltage_rms_in"),
    ],
    OUT: [("rpm", None), ("shaft_power_out", None)],
    CTC: "propulsive_load",
}
INVERTER = {
    ID: "fastga_he.pt_component.inverter",
    CN: "Inverter",
    CN_ID: "inverter_id",
    CT: "inverter",
    ATT: None,
    PT: ["settings:*"],
    IN: [("dc_voltage_in", None), (None, "dc_current_in")],
    OUT: [
        ("ac_current_rms_out_one_phase", None),
        ("ac_voltage_peak_out", None),
        ("ac_voltage_rms_out", None),
    ],
    CTC: "connector",
}
DC_BUS = {
    ID: "fastga_he.pt_component.dc_bus",
    CN: "DCBus",
    CN_ID: "dc_bus_id",
    CT: "DC_bus",
    ATT: ["number_of_inputs", "number_of_outputs"],
    PT: [],
    IN: [(None, "dc_voltage"), ("dc_current_in_", None)],
    OUT: [(None, "dc_voltage"), ("dc_current_out_", None)],
    CTC: "connector",
}
DC_LINE = {
    ID: "fastga_he.pt_component.dc_line",
    CN: "Harness",
    CN_ID: "harness_id",
    CT: "DC_cable_harness",
    ATT: None,
    PT: ["exterior_temperature", "settings:*"],
    IN: [("dc_voltage_in", None), (None, "dc_current")],
    OUT: [("dc_voltage_out", None), (None, "dc_current")],
    CTC: "connector",
}
DC_DC_CONVERTER = {
    ID: "fastga_he.pt_component.dc_dc_converter",
    CN: "DCDCConverter",
    CN_ID: "dc_dc_converter_id",
    CT: "DC_DC_converter",
    ATT: None,
    PT: [],
    IN: [("dc_voltage_in", None), (None, "dc_current_in")],
    OUT: [("dc_voltage_out", None), (None, "dc_current_out")],
    CTC: "connector",
}
BATTERY_PACK = {
    ID: "fastga_he.pt_component.battery_pack",
    CN: "BatteryPack",
    CN_ID: "battery_pack_id",
    CT: "battery_pack",
    ATT: None,
    PT: ["time_step"],
    IN: None,
    OUT: [(None, "voltage_out"), ("dc_current_out", None)],
    CTC: "source",
}

KNOWN_COMPONENTS = [PROPELLER, PMSM, INVERTER, DC_BUS, DC_LINE, DC_DC_CONVERTER, BATTERY_PACK]

KNOWN_ID = []

DICTIONARY_CN = {}
DICTIONARY_CN_ID = {}
DICTIONARY_CT = {}
DICTIONARY_ATT = {}
DICTIONARY_PT = {}
DICTIONARY_IN = {}
DICTIONARY_OUT = {}
DICTIONARY_CTC = {}

for known_component in KNOWN_COMPONENTS:
    KNOWN_ID.append(known_component[ID])
    DICTIONARY_CN[known_component[ID]] = known_component[CN]
    DICTIONARY_CN_ID[known_component[ID]] = known_component[CN_ID]
    DICTIONARY_CT[known_component[ID]] = known_component[CT]
    DICTIONARY_ATT[known_component[ID]] = known_component[ATT]
    DICTIONARY_PT[known_component[ID]] = known_component[PT]
    DICTIONARY_IN[known_component[ID]] = known_component[IN]
    DICTIONARY_OUT[known_component[ID]] = known_component[OUT]
    DICTIONARY_CTC[known_component[ID]] = known_component[CTC]
