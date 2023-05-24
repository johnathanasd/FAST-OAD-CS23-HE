# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_RECTIFIER_WEIGHT

oad.RegisterSubmodel.active_models[
    SUBMODEL_CONSTRAINTS_RECTIFIER_WEIGHT
] = "fastga_he.submodel.propulsion.rectifier.weight.sum"


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_RECTIFIER_WEIGHT,
    "fastga_he.submodel.propulsion.rectifier.weight.power_to_mass",
)
class SizingRectifierWeight(om.ExplicitComponent):
    """
    Computation of the rectifier weight, based on power density. Default value of power density
    is based on the 2025 target in :cite:`pettes:2021.`
    """

    def initialize(self):

        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            types=str,
            allow_none=False,
        )

    def setup(self):

        rectifier_id = self.options["rectifier_id"]

        self.add_input(
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
            val=np.nan,
            units="A",
        )
        self.add_input(
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber",
            val=np.nan,
            units="V",
        )
        self.add_input(
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":power_density",
            val=15.0,
            units="kW/kg",
        )

        self.add_output(
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":mass",
            val=15.0,
            units="kg",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        rectifier_id = self.options["rectifier_id"]

        current_rms_1_phase = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber"
        ]
        voltage_rms = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber"
        ] / np.sqrt(2.0)
        power_density = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":power_density"
        ]

        outputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":mass"] = (
            3.0 * current_rms_1_phase * voltage_rms / (power_density * 1000.0)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        rectifier_id = self.options["rectifier_id"]

        current_rms_1_phase = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber"
        ]
        voltage_rms = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber"
        ] / np.sqrt(2.0)
        power_density = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":power_density"
        ]

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":mass",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
        ] = (
            3.0 * voltage_rms / (power_density * 1000.0)
        )
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":mass",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber",
        ] = (
            3.0 * current_rms_1_phase / (power_density * np.sqrt(2.0) * 1000.0)
        )
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":mass",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":power_density",
        ] = -(3.0 * current_rms_1_phase * voltage_rms / (1000.0 * power_density ** 2.0))


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_RECTIFIER_WEIGHT,
    "fastga_he.submodel.propulsion.rectifier.weight.sum",
)
class SizingRectifierWeightBySum(om.ExplicitComponent):
    """
    Computation of the rectifier weight, based on the sum of individual components.
    """

    def initialize(self):
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            types=str,
            allow_none=False,
        )

    def setup(self):
        rectifier_id = self.options["rectifier_id"]

        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":casing:mass",
            units="kg",
            val=np.nan,
            desc="Weight of the casings (3 of them in the rectifier)",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":heat_sink:mass",
            units="kg",
            val=np.nan,
            desc="Mass of the heat sink, includes tubes and core",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":capacitor:mass",
            val=np.nan,
            units="kg",
            desc="Mass of the capacitor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":control_card:mass",
            val=1.0,
            units="kg",
            desc="Weight of the control card, is generally constant, taken at 1 kg",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":contactor:mass",
            units="kg",
            val=np.nan,
            desc="Mass of the 3 contactors",
        )

        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":mass",
            units="kg",
            val=40,
            desc="Mass of the rectifier",
        )

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        outputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":mass"] = (
            inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":casing:mass"]
            + inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":heat_sink:mass"]
            + inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":capacitor:mass"]
            + inputs[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":control_card:mass"
            ]
            + inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":contactor:mass"]
        )
