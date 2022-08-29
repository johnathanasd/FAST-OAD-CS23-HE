# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

# TODO: In this file we will check the results compare to NACA N640 report considering the
#  2-blades, 5868-9 propeller

import numpy as np
import matplotlib.pyplot as plt
from stdatm import Atmosphere

ELEMENTS_NUMBER = 100


def cp_from_ct(j, tip_mach, re_d, solidity, ct, activity_factor, aspect_ratio):

    cp = (
        10 ** 0.36560
        * j
        ** (
            -4.61201 * np.log10(ct)
            - 0.02274 * np.log10(j) * np.log10(activity_factor)
            + 0.01441 * np.log10(re_d) ** 2
            - 0.27259 * np.log10(ct) ** 2
            - 0.69104 * np.log10(tip_mach) * np.log10(solidity)
            - 0.29060 * np.log10(j) * np.log10(ct)
            - 0.50680 * np.log10(j) ** 2
            + 0.15905 * np.log10(solidity) ** 2
            - 2.13139 * np.log10(solidity) * np.log10(activity_factor)
            - 0.00610 * np.log10(re_d) * np.log10(aspect_ratio)
            + 6.36947 * np.log10(solidity)
            + 1.22123 * np.log10(ct) * np.log10(aspect_ratio)
            + 1.23502 * np.log10(ct) * np.log10(activity_factor)
            - 2.09630 * np.log10(solidity) * np.log10(aspect_ratio)
        )
        * tip_mach
        ** (
            -0.14528 * np.log10(tip_mach) * np.log10(activity_factor)
            - 0.45985 * np.log10(solidity) * np.log10(ct)
            - 0.77646 * np.log10(tip_mach) * np.log10(solidity)
            + 1.54416 * np.log10(solidity)
            + 0.38145 * np.log10(solidity) ** 2
            - 0.03329 * np.log10(re_d) * np.log10(ct)
            - 0.21297 * np.log10(re_d) * np.log10(solidity)
            - 0.20773 * np.log10(solidity) * np.log10(aspect_ratio)
            - 0.20054 * np.log10(solidity) * np.log10(activity_factor)
            + 0.02692 * np.log10(ct) ** 2
        )
        * re_d
        ** (
            +0.01092 * np.log10(ct) ** 2
            + 0.00145 * np.log10(re_d) ** 2
            - 0.02079 * np.log10(aspect_ratio)
            - 0.04205 * np.log10(solidity) * np.log10(ct)
        )
        * solidity
        ** (
            -0.13955 * np.log10(ct) * np.log10(activity_factor)
            + 0.12848 * np.log10(solidity) * np.log10(activity_factor)
            + 0.05448 * np.log10(solidity) * np.log10(ct)
            - 0.00950 * np.log10(activity_factor) ** 2
            - 0.07964 * np.log10(solidity) ** 2
        )
        * ct
        ** (
            2.70875
            + 0.12148 * np.log10(ct) * np.log10(activity_factor)
            - 0.64370 * np.log10(aspect_ratio)
            - 0.41614 * np.log10(activity_factor)
        )
        * activity_factor
        ** (-0.00180 * np.log10(activity_factor) ** 2 - 0.06746 * np.log10(aspect_ratio))
    )
    return cp


radius_ratio_chord = np.array(
    [
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.427,
        0.457,
        0.487,
        0.527,
        0.581,
        0.637,
        0.681,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ]
)
chord_to_diameter_ratio = np.array(
    [
        0.0377,
        0.0451,
        0.0532,
        0.0611,
        0.0687,
        0.0725,
        0.0747,
        0.0760,
        0.0763,
        0.0744,
        0.0710,
        0.0671,
        0.0620,
        0.0565,
        0.0514,
        0.0448,
        0.0380,
    ]
)

prop_diameter = 3.048
hub_diameter = 0.2 * prop_diameter
n_blades = 4.0

radius_max = prop_diameter / 2.0
radius_min = hub_diameter / 2.0

length = radius_max - radius_min
elements_number = np.arange(ELEMENTS_NUMBER)
element_length = length / ELEMENTS_NUMBER
radius_ratio = (radius_min + (elements_number + 0.5) * element_length) / radius_max
radius = radius_ratio * radius_max

chord_array = np.interp(radius_ratio, radius_ratio_chord, chord_to_diameter_ratio) * prop_diameter
solidity_naca = n_blades / np.pi / radius_max ** 2.0 * np.sum(chord_array * element_length)
activity_factor_naca = (
    100000 / 32 / radius_max ** 5.0 * np.sum(chord_array * radius ** 3.0 * element_length)
)
c_star = np.sum(chord_array * radius ** 2.0 * element_length) / np.sum(
    radius ** 2.0 * element_length
)
aspect_ratio_naca = radius_max / c_star

atm = Atmosphere(0.0, altitude_in_feet=False)
prop_rps = 1200.0 / 60.0

v = np.linspace(11.30, 50.0)
j_naca = v / prop_rps / prop_diameter
tip_mach_naca = (
    v ** 2.0 + (prop_rps * 2.0 * np.pi) ** 2.0 * (prop_diameter / 2.0) ** 2.0
) / atm.speed_of_sound ** 2.0
re_d_naca = v * prop_diameter / atm.kinematic_viscosity
pitch_ratio_naca = 15.0 * np.pi / 180.0 / 0.087


ct_list = np.array(
    [
        0.19,
        0.18,
        0.17,
        0.16,
        0.15,
        0.14,
        0.13,
        0.12,
        0.11,
        0.10,
        0.09,
        0.08,
        0.07,
        0.06,
        0.05,
        0.04,
        0.03,
        0.02,
        0.01,
    ]
)
j_list_25_deg = np.array(
    [
        0.39375,
        0.54375,
        0.60000,
        0.65000,
        0.68750,
        0.73750,
        0.78125,
        0.82500,
        0.86250,
        0.90625,
        0.95000,
        0.98750,
        1.0250,
        1.0625,
        1.1000,
        1.1375,
        1.1687,
        1.2125,
        1.2500,
    ]
)
cp_list_verif_25_deg = np.array(
    [
        0.16353,
        0.15668,
        0.15356,
        0.14920,
        0.14484,
        0.13924,
        0.13363,
        0.12678,
        0.11993,
        0.11183,
        0.10374,
        0.095640,
        0.086298,
        0.076955,
        0.067612,
        0.056401,
        0.045813,
        0.035848,
        0.025260,
    ]
)
cp_list = np.zeros_like(ct_list)
for idx, (j, ct) in enumerate(zip(j_list_25_deg, ct_list)):
    cp_list[idx] = cp_from_ct(
        j,
        (j ** 2.0 + np.pi ** 2.0) * (900 / 60 * 3.048) ** 2.0 / atm.speed_of_sound ** 2.0,
        j * 900 / 60 * 3.048 ** 2.0 / atm.kinematic_viscosity,
        solidity_naca,
        ct,
        activity_factor_naca,
        aspect_ratio_naca,
    )

plt.plot(j_list_25_deg, cp_list, label="VPLM repgression on BEMT")
plt.plot(j_list_25_deg, cp_list_verif_25_deg, label="NACA report")

# plt.plot(j_list_35_deg, cp_list, label="VPLM repgression on BEMT")
# plt.plot(j_list_35_deg, cp_list_verif_35_deg, label="NACA report")

# plt.plot(j_list_45_deg, cp_list, label="VPLM repgression on BEMT")
# plt.plot(j_list_45_deg, cp_list_bemt_45_deg, label="BEMT")
# plt.plot(j_list_45_deg, cp_list_verif_45_deg, label="NACA report")
print((cp_list - cp_list_verif_25_deg) / cp_list_verif_25_deg * 100.0)
plt.legend()
plt.show()
