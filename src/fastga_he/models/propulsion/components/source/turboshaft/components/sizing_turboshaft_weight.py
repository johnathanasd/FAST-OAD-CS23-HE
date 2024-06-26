# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingTurboshaftWeight(om.ExplicitComponent):
    """
    Computation of the installed turboshaft weight. Based on a statistical analysis, see
    :cite:`raymer:2012`.
    """

    def initialize(self):
        self.options.declare(
            name="turboshaft_id",
            default=None,
            desc="Identifier of the turboshaft",
            allow_none=False,
        )

    def setup(self):

        turboshaft_id = self.options["turboshaft_id"]

        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":uninstalled_mass",
            units="kg",
            val=np.nan,
            desc="Uninstalled weight of the turboshaft",
        )

        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":mass_ratio",
            val=1.2,
            desc="Ratio between the uninstall mass and installed mass",
        )

        self.add_output(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":mass",
            units="kg",
            val=250.0,
            desc="Installed weight of the turboshaft",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        turboshaft_id = self.options["turboshaft_id"]

        # We take 1.2 for the installed mass as now, the propeller weight is properly computed as
        # are the fuel lines
        outputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":mass"] = (
            inputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":mass_ratio"]
            * inputs[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":uninstalled_mass"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        turboshaft_id = self.options["turboshaft_id"]

        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":mass",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":mass_ratio",
        ] = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":uninstalled_mass"
        ]

        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":mass",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":uninstalled_mass",
        ] = inputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":mass_ratio"]
