# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingPEMFCDrag(om.ExplicitComponent):
    """Class that computes the contribution to profile drag of the pemfc according to the
    position given in the options. For now this will be 0.0 regardless of the option except when
    in a pod."""

    def initialize(self):

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="in_the_back",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the pemfc, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )
        # Not as useful as the ones in aerodynamics, here it will just be run twice in the sizing
        # group
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]
        # For refractoring purpose we just match the option to the tag in the variable name and
        # use it
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        # At least one input is needed regardless of the case
        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width",
            units="m",
            val=np.nan,
            desc="Width of the pemfc, as in the size of the pemfc along the Y-axis",
        )

        if position == "underbelly":

            self.add_input(
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":dimension:length",
                units="m",
                val=np.nan,
                desc="Length of the pemfc, as in the size of the pemfc along the X-axis",
            )
            self.add_input("data:geometry:fuselage:wet_area", val=np.nan, units="m**2")
            self.add_input("data:aerodynamics:fuselage:" + ls_tag + ":CD0", val=np.nan)

        if position == "wing_pod":

            self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":" + ls_tag + ":CD0",
            val=0.0,
        )

        # Should not work but actually does. I expected the value to be zero everywhere but it
        # seems like this value is overwritten by the compute_partials function
        self.declare_partials(of="*", wrt="*", val=0.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        if position == "wing_pod":

            # According to :cite:`gudmundsson:2013`. the drag of a streamlined external tank,
            # which more or less resemble a podded pemfc can be computed using the following
            # formula. It highly depends on the tank/wing interface so we will take a middle.
            # Also, there is no dependency on the tank length

            wing_area = inputs["data:geometry:wing:area"]

            frontal_area = (
                np.pi
                * inputs[
                    "data:propulsion:he_power_train:pemfc_stack:"
                    + pemfc_stack_id
                    + ":dimension:width"
                ]
                ** 2
                / 2.0
            )

            cd0 = 0.10 * frontal_area / wing_area

        elif position == "underbelly":

            # For now we will just consider the addition of wetted area and not the change in
            # form factor, ...

            cd0_fus = inputs["data:aerodynamics:fuselage:" + ls_tag + ":CD0"]

            wet_area = inputs["data:geometry:fuselage:wet_area"]
            belly_width = inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width"
            ]
            belly_length = inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:length"
            ]
            belly_height = inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:height"
            ]
            added_wet_area = (
                belly_length * belly_width
                + 2.0 * belly_length * belly_height
                + 2.0 * belly_height * belly_width
            )
            cd0 = added_wet_area / wet_area * cd0_fus

        else:

            cd0 = 0.0

        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":" + ls_tag + ":CD0"
        ] = cd0

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]
        low_speed_aero = self.options["low_speed_aero"]
        ls_tag = "low_speed" if low_speed_aero else "cruise"

        if position == "wing_pod":

            frontal_area = (
                np.pi
                * inputs[
                    "data:propulsion:he_power_train:pemfc_stack:"
                    + pemfc_stack_id
                    + ":dimension:width"
                ]
                ** 2
                / 2.0
            )

            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:geometry:wing:area",
            ] = (
                -0.10 * frontal_area / inputs["data:geometry:wing:area"] ** 2.0
            )
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width",
            ] = (
                0.10
                * np.pi
                * inputs[
                    "data:propulsion:he_power_train:pemfc_stack:"
                    + pemfc_stack_id
                    + ":dimension:width"
                ]
                / inputs["data:geometry:wing:area"]
            )

        elif position == "underbelly":

            # For now we will just consider the addition of wetted area and not the change in
            # form factor, ...

            cd0_fus = inputs["data:aerodynamics:fuselage:" + ls_tag + ":CD0"]
            wet_area = inputs["data:geometry:fuselage:wet_area"]

            belly_width = inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width"
            ]
            belly_length = inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:length"
            ]
            belly_height = inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:height"
            ]

            added_wet_area = (
                belly_length * belly_width
                + 2.0 * belly_length * belly_height
                + 2.0 * belly_height * belly_width
            )
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width",
            ] = (
                (belly_length + 2.0 * belly_height) / wet_area * cd0_fus
            )
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":dimension:height",
            ] = (
                (2.0 * belly_width + 2.0 * belly_length) / wet_area * cd0_fus
            )
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":dimension:length",
            ] = (
                (belly_width + 2.0 * belly_height) / wet_area * cd0_fus
            )
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:geometry:fuselage:wet_area",
            ] = (
                -added_wet_area / wet_area ** 2.0 * cd0_fus
            )
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:aerodynamics:fuselage:" + ls_tag + ":CD0",
            ] = (
                added_wet_area / wet_area
            )

        else:

            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width",
            ] = 0.0
