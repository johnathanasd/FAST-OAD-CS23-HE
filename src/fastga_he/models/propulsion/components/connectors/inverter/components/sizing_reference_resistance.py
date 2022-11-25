# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingInverterResistances(om.ExplicitComponent):
    """
    Computation of resistances of the diodes and IGBT, reference IGBT module for this is the
    SEMiX453GB12M7p.
    """

    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

        self.options.declare(
            name="R_igbt_ref",
            types=float,
            default=1.51e-3,
            desc="Reference IGBT resistance (Ohm)",
        )
        self.options.declare(
            name="R_diode_ref",
            types=float,
            default=1.87e-3,
            desc="Reference diode resistance (Ohm)",
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]

        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:resistance",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:resistance",
            val=1e-3,
            units="ohm",
        )
        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:resistance",
            val=1e-3,
            units="ohm",
        )

        self.declare_partials(
            of=[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:resistance",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:resistance",
            ],
            wrt="data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:resistance",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:resistance"] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:resistance"]
            * self.options["R_igbt_ref"]
        )

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:resistance"] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:resistance"]
            * self.options["R_diode_ref"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:resistance",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:resistance",
        ] = self.options["R_igbt_ref"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:resistance",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:resistance",
        ] = self.options["R_diode_ref"]
