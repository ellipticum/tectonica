import type { PlanetInputs, SimulationConfig, TectonicInputs } from "./types";

export const DEFAULT_PLANET: PlanetInputs = {
  radiusKm: 6371,
  gravity: 9.81,
  density: 5510,
  rotationHours: 24,
  axialTiltDeg: 23.5,
  eccentricity: 0.016,
  atmosphereBar: 1,
  oceanPercent: 67,
};

export const DEFAULT_TECTONICS: TectonicInputs = {
  plateCount: 11,
  plateSpeedCmPerYear: 5,
  mantleHeat: 55,
};

export const DEFAULT_SIMULATION: SimulationConfig = {
  seed: 2026,
  planet: DEFAULT_PLANET,
  tectonics: DEFAULT_TECTONICS,
  events: [],
  generationPreset: "balanced",
  scope: "planet",
};
