# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from .connectors.dc_bus import SizingDCBus, PerformancesDCBus, SlipstreamDCBus
from .connectors.dc_cable import SizingHarness, PerformancesHarness, SlipstreamHarness
from .connectors.dc_dc_converter import (
    SizingDCDCConverter,
    PerformancesDCDCConverter,
    SlipstreamDCDCConverter,
)
from .connectors.inverter import SizingInverter, PerformancesInverter, SlipstreamInverter
from .connectors.dc_sspc import SizingDCSSPC, PerformancesDCSSPC, SlipstreamDCSSPC
from .connectors.dc_splitter import SizingDCSplitter, PerformancesDCSplitter, SlipstreamDCSplitter
from .connectors.rectifier import SizingRectifier, PerformancesRectifier, SlipstreamRectifier
from .connectors.fuel_system import SizingFuelSystem, PerformancesFuelSystem, SlipstreamFuelSystem
from .connectors.speed_reducer import (
    SizingSpeedReducer,
    PerformancesSpeedReducer,
    SlipstreamSpeedReducer,
)
from .connectors.planetary_gear import (
    SizingPlanetaryGear,
    PerformancesPlanetaryGear,
    SlipstreamPlanetaryGear,
)
from .connectors.gearbox import SizingGearbox, PerformancesGearbox, SlipstreamGearbox

from .loads.pmsm import SizingPMSM, PerformancesPMSM, SlipstreamPMSM
from .loads.simple_pmsm import SizingSimplePMSM, PerformancesSimplePMSM, SlipstreamSimplePMSM

from .propulsor.propeller import SizingPropeller, PerformancesPropeller, SlipstreamPropeller

from .source.battery import SizingBatteryPack, PerformancesBatteryPack, SlipstreamBatteryPack
from .source.generator import SizingGenerator, PerformancesGenerator, SlipstreamGenerator
from .source.ice import SizingICE, PerformancesICE, SlipstreamICE
from .source.turboshaft import SizingTurboshaft, PerformancesTurboshaft, SlipstreamTurboshaft
from .source.simple_turbo_generator import (
    SizingTurboGenerator,
    PerformancesTurboGenerator,
    SlipstreamTurboGenerator,
)
from .source.pemfc import SizingPEMFCStack, PerformancesPEMFCStack, SlipstreamPEMFCStack

from .tanks.fuel_tanks import SizingFuelTank, PerformancesFuelTank, SlipstreamFuelTank
from .tanks.hydrogen_gas_tank import (
    SizingHydrogenGasTank,
    PerformancesHydrogenGasTank,
    SlipstreamHydrogenGasTank,
)
