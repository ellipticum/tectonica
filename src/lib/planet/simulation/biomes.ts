import { WORLD_SIZE } from "./core";

function classifyBiome(temp: number, precip: number, height: number): number {
  if (height < 0) return 0;
  if (temp < -5) return 1;
  if (precip > 1600 && temp >= 5 && temp < 22) return 2;
  if (temp > 12 && precip > 900) return 3;
  if (temp > 24 && precip > 500 && precip <= 1400) return 4;
  if (precip < 400 && temp > 18) return 5;
  if (precip > 700 && temp > 5 && temp < 22) return 6;
  if (height > 1800) return 7;
  return 8;
}

export function computeBiomes(temperature: Float32Array, precipitation: Float32Array, heights: Float32Array) {
  const biomes = new Uint8Array(WORLD_SIZE);
  for (let i = 0; i < WORLD_SIZE; i++) {
    biomes[i] = classifyBiome(temperature[i], precipitation[i], heights[i]);
  }
  return biomes;
}

export function computeSettlement(
  biomes: Uint8Array,
  heights: Float32Array,
  temperature: Float32Array,
  precipitation: Float32Array,
) {
  const settlement = new Float32Array(WORLD_SIZE);
  for (let i = 0; i < WORLD_SIZE; i++) {
    if (biomes[i] === 0) {
      settlement[i] = 0;
      continue;
    }
    const comfortT = 1 - Math.abs(temperature[i] - 18) / 45;
    const comfortP = 1 - Math.abs(precipitation[i] - 1400) / 2200;
    const elevationPenalty = Math.max(0, heights[i] - 1200) / 2600;
    const value = Math.max(0, Math.min(1, (comfortT + comfortP) / 2 - elevationPenalty));
    settlement[i] = value;
  }
  return settlement;
}
