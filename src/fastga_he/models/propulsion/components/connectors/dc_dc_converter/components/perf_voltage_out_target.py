# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from fastga_he.exceptions import ControlParameterInconsistentShapeError

from fastga.models.performances.mission.mission_components import (
    POINTS_NB_CLIMB,
    POINTS_NB_CRUISE,
    POINTS_NB_DESCENT,
)


class PerformancesVoltageOutTargetMission(om.ExplicitComponent):
    """
    Component which takes the desired voltage output target for converter operation from the data
    and gives it the right format for the mission. It was deemed best to put it this way rather
    than the original way to simplify the construction of the power train file.

    The input voltage target can either be a float (then during the whole mission the
    voltage is going to be the same), an array of three element (different voltage for the
    whole climb, whole cruise and whole descent) or an array of number of points elements for the
    individual control of each point.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )

    def setup(self):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_out_target_mission",
            val=np.nan,
            units="V",
            desc="Target output voltage of the DC/DC converter for the points",
            shape_by_conn=True,
        )

        self.add_output("voltage_out_target", units="V", val=850.0, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        number_of_points = self.options["number_of_points"]

        v_out_tgt_mission = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_out_target_mission"
        ]

        if len(v_out_tgt_mission) == 1:
            outputs["voltage_out_target"] = np.full(number_of_points, v_out_tgt_mission)

        elif len(v_out_tgt_mission) == 3:
            outputs["voltage_out_target"] = np.concatenate(
                (
                    np.full(POINTS_NB_CLIMB, v_out_tgt_mission[0]),
                    np.full(POINTS_NB_CRUISE, v_out_tgt_mission[1]),
                    np.full(POINTS_NB_DESCENT, v_out_tgt_mission[2]),
                )
            )

        elif len(v_out_tgt_mission) == number_of_points:
            outputs["voltage_out_target"] = v_out_tgt_mission

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_out_target_mission"
                + " should be 1, 3 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        number_of_points = self.options["number_of_points"]

        v_out_tgt_mission = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_out_target_mission"
        ]

        if len(v_out_tgt_mission) == 1:
            partials[
                "voltage_out_target",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_out_target_mission",
            ] = np.full(number_of_points, 1.0)

        elif len(v_out_tgt_mission) == 3:
            tmp_partials = np.zeros((number_of_points, 3))
            tmp_partials[:POINTS_NB_CLIMB, 0] = 1
            tmp_partials[POINTS_NB_CLIMB : POINTS_NB_CLIMB + POINTS_NB_CRUISE, 1] = 1
            tmp_partials[POINTS_NB_CLIMB + POINTS_NB_CRUISE :, 2] = 1
            partials[
                "voltage_out_target",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_out_target_mission",
            ] = tmp_partials

        elif len(v_out_tgt_mission) == number_of_points:
            partials[
                "voltage_out_target",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_out_target_mission",
            ] = np.eye(number_of_points)
