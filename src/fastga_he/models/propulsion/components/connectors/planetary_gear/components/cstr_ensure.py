# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import (
    SUBMODEL_CONSTRAINTS_PLANETARY_GEAR_TORQUE,
)

import openmdao.api as om
import numpy as np

import fastoad.api as oad


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PLANETARY_GEAR_TORQUE,
    "fastga_he.submodel.propulsion.constraints.planetary_gear.torque.ensure",
)
class ConstraintsTorqueEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum torque seen by the planetary gear
    during the mission and the value used for sizing, ensuring each component works below its
    maximum.
    """

    def initialize(self):
        self.options.declare(
            name="planetary_gear_id",
            default=None,
            desc="Identifier of the speed reducer",
            allow_none=False,
        )

    def setup(self):

        planetary_gear_id = self.options["planetary_gear_id"]

        self.add_input(
            "data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":torque_out_max",
            units="N*m",
            val=np.nan,
            desc="Maximum value of the output torque of the gearbox",
        )
        self.add_input(
            "data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":torque_out_rating",
            units="N*m",
            val=np.nan,
            desc="Max continuous output torque of the gearbox",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":torque_out_rating",
            units="N*m",
            val=250.0,
            desc="Respected if <0",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":torque_out_rating",
            wrt=[
                "data:propulsion:he_power_train:planetary_gear:"
                + planetary_gear_id
                + ":torque_out_max",
                "data:propulsion:he_power_train:planetary_gear:"
                + planetary_gear_id
                + ":torque_out_rating",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        planetary_gear_id = self.options["planetary_gear_id"]

        outputs[
            "constraints:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":torque_out_rating"
        ] = (
            inputs[
                "data:propulsion:he_power_train:planetary_gear:"
                + planetary_gear_id
                + ":torque_out_max"
            ]
            - inputs[
                "data:propulsion:he_power_train:planetary_gear:"
                + planetary_gear_id
                + ":torque_out_rating"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        planetary_gear_id = self.options["planetary_gear_id"]

        partials[
            "constraints:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":torque_out_rating",
            "data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":torque_out_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":torque_out_rating",
            "data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":torque_out_rating",
        ] = -1.0
