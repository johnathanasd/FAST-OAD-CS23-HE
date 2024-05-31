# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesSinglePEMFCStackCurrent(om.ExplicitComponent):
    """
    Computation of the current of single layer proton exchange membrane fuel cell stack from the fuel cell effective area.
    Model based on existing pemfc, Aerostack Ultralight 200, details can be found in:
    cite:`Fuel Cell and Battery Hybrid System Optimization by J. Hoogendoorn:2018`.
    """

    # TODO: Integrate model based on Thermodynamic calculation from D.Juschus

    def initialize(self):

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        self.add_input(
            "fc_current_density",
            units="A/cm**2",
            val=np.nan,
        )

        self.add_input(
            "effective_area",
            units="cm**2",
            val=16.8,
        )

        self.add_output(
            "single_stack_pemfc_current",
            units="A",
            val=10,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]
        outputs[
            "single_stack_pemfc_current"
        ] = (inputs["fc_current_density"] * inputs["effective_area"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + "single_stack_pemfc_current",
            "fc_current_density",
        ] = inputs["effective_area"]
        partials[
            "data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + "single_stack_pemfc_current",
            "effective_area",
        ] = inputs["fc_current_density"]
