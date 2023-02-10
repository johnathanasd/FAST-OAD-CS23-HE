# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingDCDCConverterInductorCoreScaling(om.ExplicitComponent):
    """
    Computation of the scaling coefficients for the core of the filter inductor necessary to
    store the energy, implementation in the openMDAO format of :cite:`budinger_sizing_2023`.
    """

    def initialize(self):
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )
        self.options.declare(
            name="iron_surface_ref",
            types=float,
            default=738e-6,
            desc="Iron surface of the reference component [m**2]",
        )

    def setup(self):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_surface",
            units="m**2",
            val=np.nan,
            desc="Iron surface of the E-core inductor",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:scaling:core_dimension",
            val=1.0,
            desc="Scaling factor for the dimensions of the E-core",
        )
        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:scaling:core_mass",
            val=1.0,
            desc="Scaling factor for the mass of the E-core",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        iron_area_ref = self.options["iron_surface_ref"]
        iron_area = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_surface"
        ]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:scaling:core_dimension"
        ] = np.sqrt(iron_area / iron_area_ref)

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:scaling:core_mass"
        ] = (iron_area / iron_area_ref) ** (3.0 / 2.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        iron_area_ref = self.options["iron_surface_ref"]
        iron_area = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_surface"
        ]

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:scaling:core_dimension",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_surface",
        ] = 0.5 * np.sqrt(1.0 / (iron_area_ref * iron_area))

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:scaling:core_mass",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:iron_surface",
        ] = (
            3.0 / 2.0 * np.sqrt(iron_area / iron_area_ref ** 3.0)
        )
