# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from ..constants import (
    SUBMODEL_CONSTRAINTS_TURBO_GENERATOR_POWER,
)


class ConstraintsTurboGenerator(om.Group):
    """
    Class that gather the different constraints for the generator be they ensure or enforce.
    """

    def initialize(self):

        self.options.declare(
            name="turbo_generator_id",
            default=None,
            desc="Identifier of the turbo generator",
            allow_none=False,
        )

    def setup(self):

        turbo_generator_id = self.options["turbo_generator_id"]

        option_turbo_generator_id = {"turbo_generator_id": turbo_generator_id}

        self.add_subsystem(
            name="constraints_torque_generator",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_TURBO_GENERATOR_POWER, options=option_turbo_generator_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="power_for_power_rate",
            subsys=ConstraintTurboGeneratorPowerRateMission(turbo_generator_id=turbo_generator_id),
            promotes=["*"],
        )


class ConstraintTurboGeneratorPowerRateMission(om.ExplicitComponent):
    """
    This class will define the value of the maximum power we use to get the power rate inside the
    mission, it is mandatory that we compute it outside the mission when sizing the power train
    or else when recomputing the wing area it will be stuck at one which we don't want. It's
    nothing complex but we need it done outside of the mission and we need consistent naming.
    """

    def initialize(self):
        self.options.declare(
            name="turbo_generator_id",
            default=None,
            desc="Identifier of the turbo generator",
            allow_none=False,
        )

    def setup(self):

        turbo_generator_id = self.options["turbo_generator_id"]

        self.add_input(
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":shaft_power_max",
            units="kW",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":shaft_power_rating",
            units="kW",
            val=42000.0,
            desc="Value of the maximum power the generator can provide, used for sizing",
        )

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        turbo_generator_id = self.options["turbo_generator_id"]

        outputs[
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":shaft_power_rating"
        ] = inputs[
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":shaft_power_max"
        ]
