# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import SUBMODEL_CONSTRAINTS_PEMFC_NL_POWER

import fastoad.api as oad

oad.RegisterSubmodel.active_models[
    SUBMODEL_CONSTRAINTS_PEMFC_NL_POWER
] = "submodel.propulsion.constraints.pemfc_stack.norminal_power.enforce"


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PEMFC_NL_POWER,
    "fastga_he.submodel.propulsion.constraints.pemfc_stack.nominal_power.enforce",
)
class ConstraintsNorminalPowerEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum power seen by the pemfc during the mission is used for
    the sizing.
    """

    def initialize(self):

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the pemfc stacks",
            allow_none=False,
        )

    def setup(self):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack_stack:" + pemfc_stack_id + ":power_max",
            units="kW",
            val=np.nan,
            desc="Maximum power the PEMFC stack has to provide during mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_nominal",
            units="kW",
            val=25.0,
            desc="Maximum power the pemfc_stack can provide at Nominal pressure condition",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_nominal",
            wrt="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_nominal"
        ] = inputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"]
