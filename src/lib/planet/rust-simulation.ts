import { run_simulation, run_simulation_with_progress } from "@/lib/planet-rs/pkg/planet_engine";
import {
  LAYER_GRAPH,
  type LayerId,
  type RecomputeTrigger,
  type SimulationConfig,
  type SimulationResult,
  type WorldEventRecord,
} from "@/lib/planet/simulation";

function ensureEventEnergy(event: WorldEventRecord): WorldEventRecord {
  if (event.kind === "meteorite" && event.energyJoule === undefined) {
    const r = (event.diameterKm * 1000) / 2;
    const mass = (4 / 3) * Math.PI * Math.pow(r, 3) * event.densityKgM3;
    return {
      ...event,
      energyJoule: 0.5 * mass * Math.pow(event.speedKms * 1000, 2),
    };
  }
  return event;
}

export function runSimulationRust(
  config: SimulationConfig,
  reason: RecomputeTrigger = "global",
  onProgress?: (percent: number) => void,
): SimulationResult {
  const events = config.events.map((event) => ensureEventEnergy(event as WorldEventRecord));
  const wasm = onProgress
    ? run_simulation_with_progress({ ...config, events }, reason, (value: number) => {
        onProgress(Math.max(0, Math.min(100, value)));
      })
    : run_simulation({ ...config, events }, reason);

  const result: SimulationResult = {
    width: wasm.width,
    height: wasm.height,
    seed: wasm.seed,
    specs: LAYER_GRAPH.map((s) => ({
      ...s,
      dependsOn: [...s.dependsOn],
    })),
    recomputedLayers: Array.from(wasm.recomputedLayers()) as LayerId[],
    planet: {
      seaLevel: wasm.seaLevel(),
      radiusKm: wasm.radiusKm(),
      oceanPercent: wasm.oceanPercent(),
    },
    plates: new Int16Array(wasm.plates()),
    boundaryTypes: new Int8Array(wasm.boundaryTypes()),
    heightMap: new Float32Array(wasm.heightMap()),
    slopeMap: new Float32Array(wasm.slopeMap()),
    riverMap: new Float32Array(wasm.riverMap()),
    lakeMap: new Uint8Array(wasm.lakeMap()),
    flowDirection: new Int32Array(wasm.flowDirection()),
    flowAccumulation: new Float32Array(wasm.flowAccumulation()),
    temperatureMap: new Float32Array(wasm.temperatureMap()),
    precipitationMap: new Float32Array(wasm.precipitationMap()),
    biomeMap: new Uint8Array(wasm.biomeMap()),
    settlementMap: new Float32Array(wasm.settlementMap()),
    eventHistory: events.map((e) => ({ ...e })),
    stats: {
      minHeight: wasm.minHeight(),
      maxHeight: wasm.maxHeight(),
      minTemperature: wasm.minTemperature(),
      maxTemperature: wasm.maxTemperature(),
      minPrecipitation: wasm.minPrecipitation(),
      maxPrecipitation: wasm.maxPrecipitation(),
      minSlope: wasm.minSlope(),
      maxSlope: wasm.maxSlope(),
    },
  };

  wasm.free();
  return result;
}
