export { WORLD_HEIGHT, WORLD_WIDTH } from "./simulation/core";
export { DEFAULT_PLANET, DEFAULT_SIMULATION, DEFAULT_TECTONICS } from "./simulation/defaults";
export {
  BIOME_COLORS,
  BIOME_NAMES,
  LAYER_GRAPH,
  type EventKind,
  type GenerationScope,
  type IslandType,
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

import { clamp } from "./simulation/core";
import type { HeightExportPayload, SimulationResult } from "./simulation/types";

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
