# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pyvplm.addon.variablepowerlaw import regression_models, perform_regression

if __name__ == "__main__":

    # Data correspond to what is found inside data/FourCylindersAtmospheric.csv

    np.set_printoptions(suppress=True)

    rpm_data = np.array(
        [2200.0, 2250.0, 2300.0, 2350.0, 2400.0, 2450.0, 2500.0, 2550.0, 2600.0, 2650.0, 2700.0]
    )
    pme_data = np.array(
        [
            8.08191746,
            8.51454368,
            8.9471699,
            9.37979611,
            9.81242233,
            10.24504855,
            10.67767477,
            11.11030098,
            11.5429272,
            11.97555342,
            12.40817964,
            12.84080586,
            13.27343207,
            13.70605829,
            14.13868451,
            14.57131073,
            15.00393695,
            15.43656316,
            15.86918938,
            16.3018156,
            16.73444182,
            17.16706803,
            17.59969425,
            18.03232047,
            18.46494669,
            18.89757291,
            19.33019912,
            19.76282534,
            20.19545156,
            20.62807778,
        ]
    )
    sfc_data = np.array(
        [
            [
                283.45360944,
                290.51392435,
                294.90730834,
                292.47356457,
                295.70505775,
                298.1595,
                296.86047431,
                298.52567236,
                300.27779009,
                303.4296168,
                305.41687499,
            ],
            [
                276.62219804,
                282.49095164,
                286.22389294,
                284.0203279,
                287.07808638,
                289.48048848,
                288.51623285,
                290.06508358,
                291.87139366,
                295.14707034,
                297.57388711,
            ],
            [
                270.81098137,
                275.59314125,
                278.72480776,
                276.69500127,
                279.61803897,
                281.96462223,
                281.33597613,
                282.78119663,
                284.65226448,
                288.04640252,
                290.87977503,
            ],
            [
                265.77499659,
                269.58774045,
                272.29186651,
                270.41318562,
                273.23439279,
                275.45410084,
                275.17970247,
                276.51094216,
                278.4557159,
                281.96371792,
                285.17528258,
            ],
            [
                261.40471098,
                264.36240155,
                266.69256101,
                264.9824409,
                267.69570551,
                269.81825309,
                269.86974363,
                271.11983683,
                273.14611974,
                276.76409322,
                280.32928697,
            ],
            [
                257.61972762,
                259.82229682,
                261.82578728,
                260.2684284,
                262.89814713,
                264.94785404,
                265.30006488,
                266.49630421,
                268.61101835,
                272.33546707,
                276.23278021,
            ],
            [
                254.35158723,
                255.88697964,
                257.60575649,
                256.18714609,
                258.75376485,
                260.7509716,
                261.38180553,
                262.56595072,
                264.75597653,
                268.58396694,
                272.79452235,
            ],
            [
                251.54188475,
                252.48785659,
                253.95930824,
                252.66693704,
                255.18789107,
                257.14974104,
                258.03960494,
                259.23191673,
                261.50149343,
                265.43041662,
                269.93752575,
            ],
            [
                249.14057394,
                249.56611969,
                250.82362069,
                249.64641824,
                252.13676741,
                254.07785123,
                255.20926314,
                256.4213543,
                258.77984435,
                262.80737037,
                267.59634602,
            ],
            [
                247.10434601,
                247.0710535,
                248.14456697,
                247.07264172,
                249.54545196,
                251.47841147,
                252.83578041,
                254.0779094,
                256.53316435,
                260.65724028,
                265.71501392,
            ],
            [
                245.39817464,
                244.98251499,
                245.87543567,
                244.89975389,
                247.36645788,
                249.30234561,
                250.87166969,
                252.15310208,
                254.71186529,
                258.9306171,
                264.24538339,
            ],
            [
                244.02232841,
                243.25556967,
                243.97541512,
                243.08800841,
                245.55867737,
                247.50714537,
                249.27579813,
                250.60488671,
                253.27774854,
                257.58455411,
                263.14570475,
            ],
            [
                242.90530276,
                241.83147835,
                242.43359403,
                241.6025255,
                244.08620066,
                246.05574152,
                248.01233873,
                249.39650186,
                252.19091958,
                256.58178302,
                262.37983895,
            ],
            [
                242.02374578,
                240.68270035,
                241.20195375,
                240.43163589,
                242.91757142,
                244.91572391,
                247.04995077,
                248.49604067,
                251.41480272,
                255.88990446,
                261.91618795,
            ],
            [
                241.35693848,
                239.78502666,
                240.23863087,
                239.53847176,
                242.02547266,
                244.05866481,
                246.36115572,
                247.87534697,
                250.9208944,
                255.4801099,
                261.72689503,
            ],
            [
                240.8864896,
                239.11691071,
                239.52070248,
                238.88523281,
                241.40714827,
                243.45944837,
                245.921675,
                247.50952972,
                250.68421336,
                255.32959342,
                261.78752247,
            ],
            [
                240.59600022,
                238.65907552,
                239.0276742,
                238.45178219,
                241.01489157,
                243.1043116,
                245.70996575,
                247.37672258,
                250.68251906,
                255.41738428,
                262.07650527,
            ],
            [
                240.47077691,
                238.39432063,
                238.74118059,
                238.22034242,
                240.8301462,
                242.97027639,
                245.7069914,
                247.45730974,
                250.89610938,
                255.71914593,
                262.57453427,
            ],
            [
                240.49764108,
                238.30728545,
                238.6447762,
                238.17477633,
                240.83629382,
                243.0320168,
                245.89567295,
                247.73392131,
                251.30748208,
                256.21759956,
                263.26451543,
            ],
            [
                240.66897655,
                238.39541093,
                238.72370241,
                238.30061865,
                241.01847854,
                243.27428327,
                246.26078867,
                248.19102968,
                251.90090474,
                256.89708976,
                264.13114144,
            ],
            [
                240.98260568,
                238.64220629,
                238.96464016,
                238.58495207,
                241.36328103,
                243.68319547,
                246.78868694,
                248.81461783,
                252.66220573,
                257.74357543,
                265.16082872,
            ],
            [
                241.41348423,
                239.02604533,
                239.35600899,
                239.01597882,
                241.85866011,
                244.24639979,
                247.46970645,
                249.59213544,
                253.57882115,
                258.74446263,
                266.3410868,
            ],
            [
                241.95331863,
                239.53724397,
                239.90270669,
                239.58308931,
                242.49367606,
                244.95258008,
                248.29497426,
                250.51222632,
                254.63923726,
                259.88851129,
                267.66085531,
            ],
            [
                242.59468856,
                240.16701551,
                240.57615279,
                240.2830816,
                243.25834231,
                245.7915657,
                249.24850846,
                251.56470299,
                255.83318864,
                261.16527743,
                269.11008813,
            ],
            [
                243.3306811,
                240.9073058,
                241.36774468,
                241.10705236,
                244.14375647,
                246.75413297,
                250.3214515,
                252.74027033,
                257.15122863,
                262.56556575,
                270.67967043,
            ],
            [
                244.1549737,
                241.75075189,
                242.2696389,
                242.03887908,
                245.14174251,
                247.83183768,
                251.50566228,
                254.03051306,
                258.58492817,
                264.08093974,
                272.36139126,
            ],
            [
                245.06188205,
                242.69064666,
                243.27470337,
                243.07148461,
                246.24479225,
                249.017051,
                252.79370954,
                255.42762648,
                260.1264737,
                265.70359071,
                274.14761916,
            ],
            [
                246.0461119,
                243.72079058,
                244.37631997,
                244.19850439,
                247.45464617,
                250.3027821,
                254.17880522,
                256.92470597,
                261.76882877,
                267.42654221,
                276.03162995,
            ],
            [
                247.10658274,
                244.84208085,
                245.56844113,
                245.41402278,
                248.75570276,
                251.68250648,
                255.65479219,
                258.51524165,
                263.50549315,
                269.24332124,
                278.0069874,
            ],
            [
                248.24017028,
                246.0462909,
                246.84561277,
                246.71259601,
                250.14185676,
                253.15051445,
                257.21603221,
                260.19351911,
                265.3305648,
                271.14800246,
                280.06799999,
            ],
        ]
    )

    max_rpm = np.max(rpm_data)
    max_pme = np.max(pme_data)

    rpm_mesh_grid, pme_mesh_grid = np.meshgrid(rpm_data, pme_data)
    rpm_for_regression = rpm_mesh_grid.flatten() / max_rpm
    pme_for_regression = pme_mesh_grid.flatten() / max_pme
    sfc_for_regression = sfc_data.flatten()

    A = np.c_[
        np.ones_like(rpm_for_regression),
        rpm_for_regression,
        pme_for_regression,
        rpm_for_regression ** 2.0,
        pme_for_regression * rpm_for_regression,
        pme_for_regression ** 2.0,
        rpm_for_regression ** 3.0,
        rpm_for_regression ** 2.0 * pme_for_regression,
        rpm_for_regression * pme_for_regression ** 2.0,
        pme_for_regression ** 3.0,
    ]
    B = sfc_for_regression

    # Solve the system of equations.
    result, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c, d, e, f, g, h, i, j = result
    print(result)

    sfc_test = (
        a * np.ones_like(rpm_for_regression)
        + b * rpm_for_regression
        + c * pme_for_regression
        + d * rpm_for_regression ** 2.0
        + e * pme_for_regression * rpm_for_regression
        + f * pme_for_regression ** 2.0
        + g * rpm_for_regression ** 3.0
        + h * rpm_for_regression ** 2.0 * pme_for_regression
        + i * rpm_for_regression * pme_for_regression ** 2.0
        + j * pme_for_regression ** 3.0
    )
    error_percent = (sfc_test - sfc_for_regression) / sfc_for_regression * 100.0
    print(np.mean(abs(error_percent)))
    print(np.max(abs(error_percent)))

    new_sfc_for_contour = (
        a * np.ones_like(rpm_mesh_grid)
        + b * (rpm_mesh_grid / max_rpm)
        + c * (pme_mesh_grid / max_pme)
        + d * (rpm_mesh_grid / max_rpm) ** 2.0
        + e * (pme_mesh_grid / max_pme) * (rpm_mesh_grid / max_rpm)
        + f * (pme_mesh_grid / max_pme) ** 2.0
        + g * (rpm_mesh_grid / max_rpm) ** 3.0
        + h * (rpm_mesh_grid / max_rpm) ** 2.0 * (pme_mesh_grid / max_pme)
        + i * (rpm_mesh_grid / max_rpm) * (pme_mesh_grid / max_pme) ** 2.0
        + j * (pme_mesh_grid / max_pme) ** 3.0
    )

    fig2 = go.Figure()
    sfc_contour_new = go.Contour(
        x=rpm_data,
        y=pme_data,
        z=new_sfc_for_contour,
        ncontours=20,
        contours=dict(
            coloring="heatmap",
            showlabels=True,  # show labels on contours
            labelfont=dict(  # label font properties
                size=12,
                color="white",
            ),
        ),
        zmax=np.max(new_sfc_for_contour),
        zmin=np.min(new_sfc_for_contour),
    )
    fig2.add_trace(sfc_contour_new)
    fig2.update_layout(
        title_text="Interpolated data for ICE SFC",
        title_x=0.5,
        xaxis_title="Mean Effective Pressure [bar]",
        yaxis_title="RPM [min**-1]",
    )
    # fig2.show()

    sfc_diff = (new_sfc_for_contour - sfc_data) / sfc_data * 100.0
    fig3 = go.Figure()
    sfc_diff_contour_new = go.Contour(
        x=rpm_data,
        y=pme_data,
        z=sfc_diff,
        ncontours=20,
        contours=dict(
            coloring="heatmap",
            showlabels=True,  # show labels on contours
            labelfont=dict(  # label font properties
                size=12,
                color="white",
            ),
        ),
        zmax=np.max(sfc_diff),
        zmin=np.min(sfc_diff),
    )
    fig3.add_trace(sfc_diff_contour_new)
    fig3.update_layout(
        title_text="Percent difference with data SFC",
        title_x=0.5,
        xaxis_title="Mean Effective Pressure [bar]",
        yaxis_title="RPM [min**-1]",
    )
    fig3.show()
