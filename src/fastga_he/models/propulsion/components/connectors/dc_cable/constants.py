# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

SUBMODEL_CONSTRAINTS_DC_LINE_CURRENT = "submodel.propulsion.constraints.dc_line.current"
SUBMODEL_CONSTRAINTS_DC_LINE_VOLTAGE = "submodel.propulsion.constraints.dc_line.voltage"

SUBMODEL_DC_LINE_SIZING_LENGTH = "submodel.propulsion.sizing.dc_line.length"
SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE = (
    "submodel.propulsion.performances.dc_line.temperature_profile"
)
SUBMODEL_DC_LINE_PERFORMANCES_RESISTANCE_PROFILE = (
    "submodel.propulsion.performances.dc_line.resistance_profile"
)

POSSIBLE_POSITION = [
    "inside_the_wing",
    "from_rear_to_front",
    "from_rear_to_wing",
    "from_front_to_wing",
    "from_rear_to_nose",
    "from_front_to_nose",
    "from_wing_to_nose",
]
