# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import SUBMODEL_CONSTRAINTS_PEMFC_NL_POWER


import fastoad.api as oad


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PEMFC_NL_POWER,
    "fastga_he.submodel.propulsion.constraints.pemfc_stack.nominal_power.ensure",
)
class ConstraintsNominalPowerEnsure(om.ExplicitComponent):
    """
    Class that ensures that the maximum power seen by the PEMFC stack during the mission is below the
    one used for sizing, ensuring each component works below its minimum.
    """

    def initialize(self):

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            units="kW",
            val=np.nan,
            desc="Maximum power the pemfc has to provide in whole mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_nominal",
            units="kW",
            val=np.nan,
            desc="Maximum power the pemfc can provide with nominal pressure condition",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":power_nominal",
            units="kW",
            val=-0.0,
            desc="Respected if <0",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":power_nominal",
            wrt="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            val=1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":power_nominal",
            wrt="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_nominal",
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs[
            "constraints:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_nominal"
        ] = (
            inputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"]
            - inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_nominal"
            ]
        )
