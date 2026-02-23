export { WORLD_HEIGHT, WORLD_WIDTH } from "./simulation/core";
export { DEFAULT_PLANET, DEFAULT_SIMULATION, DEFAULT_TECTONICS } from "./simulation/defaults";
export {
  BIOME_COLORS,
  BIOME_NAMES,
  LAYER_GRAPH,
  type EventKind,
  type HeightExportPayload,
  type LayerId,
  type LayerSpec,
  type MeteoriteEvent,
  type PlanetInputs,
  type RecomputeTrigger,
  type RegionEvent,
  type SimulationConfig,
  type SimulationResult,
  type SimulationStats,
  type TectonicInputs,
  type WorldDisplayLayer,
  type WorldEvent,
  type WorldEventRecord,
} from "./simulation/types";

import { clamp, minMax } from "./simulation/core";
import { computePlates } from "./simulation/plates";
import { computeRelief } from "./simulation/relief";
import { applyEvents, ensureEventEnergy } from "./simulation/events";
import { computeSlope, computeHydrology } from "./simulation/hydrology";
import { computeClimate } from "./simulation/climate";
import { computeBiomes, computeSettlement } from "./simulation/biomes";
import { evaluateRecompute } from "./simulation/recompute";
import {
  LAYER_GRAPH,
  type HeightExportPayload,
  type RecomputeTrigger,
  type SimulationConfig,
  type SimulationResult,
  type WorldEventRecord,
} from "./simulation/types";
import { WORLD_HEIGHT, WORLD_WIDTH } from "./simulation/core";

export function runSimulation(config: SimulationConfig, reason: RecomputeTrigger = "global"): SimulationResult {
  const planet = { ...config.planet };
  const tectonics = { ...config.tectonics };
  const events = config.events.map((event) => ensureEventEnergy(event as WorldEventRecord));

  const layerOrder = evaluateRecompute(reason);
  const platesLayer = computePlates(planet, tectonics, config.seed);
  const reliefRaw = computeRelief(planet, tectonics, platesLayer, config.seed);
  const eventRelief = applyEvents(planet, reliefRaw.relief, events);
  const { slope, minSlope, maxSlope } = computeSlope(eventRelief.relief);
  const hydro = computeHydrology(eventRelief.relief, slope);
  const climate = computeClimate(
    planet,
    eventRelief.relief,
    slope,
    hydro.rivers,
    hydro.flowAccumulation,
    eventRelief.aerosol,
  );
  const biomes = computeBiomes(climate.temperature, climate.precipitation, eventRelief.relief);
  const settlement = computeSettlement(biomes, eventRelief.relief, climate.temperature, climate.precipitation);
  const heightRange = minMax(eventRelief.relief);

  return {
    width: WORLD_WIDTH,
    height: WORLD_HEIGHT,
    seed: config.seed,
    specs: LAYER_GRAPH.map((s) => ({
      ...s,
      dependsOn: [...s.dependsOn],
    })),
    recomputedLayers: layerOrder,
    planet: {
      seaLevel: reliefRaw.seaLevel,
      radiusKm: planet.radiusKm,
      oceanPercent: planet.oceanPercent,
    },
    plates: platesLayer.plateField,
    boundaryTypes: platesLayer.boundaryTypes,
    heightMap: eventRelief.relief,
    slopeMap: slope,
    riverMap: hydro.rivers,
    lakeMap: hydro.lakes,
    flowDirection: hydro.flowDirection,
    flowAccumulation: hydro.flowAccumulation,
    temperatureMap: climate.temperature,
    precipitationMap: climate.precipitation,
    biomeMap: biomes,
    settlementMap: settlement,
    eventHistory: events,
    stats: {
      minHeight: heightRange.min,
      maxHeight: heightRange.max,
      minTemperature: climate.minTemp,
      maxTemperature: climate.maxTemp,
      minPrecipitation: climate.minPrec,
      maxPrecipitation: climate.maxPrec,
      minSlope,
      maxSlope,
    },
  };
}

export function toHeightExportPayload(
  result: SimulationResult,
  mode: "float32" | "int16",
): HeightExportPayload {
  if (mode === "int16") {
    const scale = 1;
    const values = Array.from(result.heightMap, (v) =>
      Math.round(clamp(v / scale, -32768, 32767)),
    );
    return {
      format: "int16",
      width: result.width,
      height: result.height,
      radiusKm: result.planet.radiusKm,
      scaleFactor: scale,
      values,
    };
  }

  return {
    format: "float32",
    width: result.width,
    height: result.height,
    radiusKm: result.planet.radiusKm,
    values: Array.from(result.heightMap),
  };
}
