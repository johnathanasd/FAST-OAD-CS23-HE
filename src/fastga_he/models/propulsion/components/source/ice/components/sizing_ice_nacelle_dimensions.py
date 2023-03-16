# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingICENacelleDimensions(om.ExplicitComponent):
    """
    Computation of the dimensions of the ICE nacelle. Based on some very simple geometric ratio
    which depend on the engine position. When not on the wing, the nacelle dimensions will purely
    be indicative.
    """

    def initialize(self):

        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the generator, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        ice_id = self.options["ice_id"]

        self.add_input(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":engine:length",
            val=np.nan,
            desc="Length of the ICE",
            units="m",
        )
        self.add_input(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":engine:width",
            val=np.nan,
            desc="Width of the ICE",
            units="m",
        )
        self.add_input(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":engine:height",
            val=np.nan,
            desc="Height of the ICE",
            units="m",
        )

        self.add_output(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":nacelle:length",
            val=1.5,
            desc="Length of the ICE nacelle",
            units="m",
        )
        self.add_output(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":nacelle:width",
            val=1.1,
            desc="Width of the ICE nacelle",
            units="m",
        )
        self.add_output(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":nacelle:height",
            val=0.6,
            desc="Height of the ICE nacelle",
            units="m",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ICE:" + ice_id + ":nacelle:length",
            wrt="data:propulsion:he_power_train:ICE:" + ice_id + ":engine:length",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:ICE:" + ice_id + ":nacelle:width",
            wrt="data:propulsion:he_power_train:ICE:" + ice_id + ":engine:width",
            val=1.1,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:ICE:" + ice_id + ":nacelle:height",
            wrt="data:propulsion:he_power_train:ICE:" + ice_id + ":engine:height",
            val=1.1,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ice_id = self.options["ice_id"]

        if self.options["position"] == "on_the_wing":

            outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":nacelle:length"] = (
                inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":engine:length"] * 2.0
            )

        else:

            outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":nacelle:length"] = (
                inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":engine:length"] * 1.15
            )

        outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":nacelle:width"] = (
            inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":engine:width"] * 1.1
        )
        outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":nacelle:height"] = (
            inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":engine:height"] * 1.1
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        ice_id = self.options["ice_id"]

        if self.options["position"] == "on_the_wing":

            partials[
                "data:propulsion:he_power_train:ICE:" + ice_id + ":nacelle:length",
                "data:propulsion:he_power_train:ICE:" + ice_id + ":engine:length",
            ] = 2.0

        else:

            partials[
                "data:propulsion:he_power_train:ICE:" + ice_id + ":nacelle:length",
                "data:propulsion:he_power_train:ICE:" + ice_id + ":engine:length",
            ] = 1.15
