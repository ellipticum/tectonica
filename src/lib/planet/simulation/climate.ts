import { RADIANS, WORLD_SIZE, clamp, latByY } from "./core";
import type { PlanetInputs } from "./types";

export function computeClimate(
  planet: PlanetInputs,
  heights: Float32Array,
  slope: Float32Array,
  rivers: Float32Array,
  flow: Float32Array,
  aerosol: number,
) {
  const temperature = new Float32Array(WORLD_SIZE);
  const precipitation = new Float32Array(WORLD_SIZE);
  const tiltRad = planet.axialTiltDeg * RADIANS;
  const rotationFactor = 20 / Math.max(1, planet.rotationHours);

  let minTemp = Number.POSITIVE_INFINITY;
  let maxTemp = -Number.POSITIVE_INFINITY;
  let minPrec = Number.POSITIVE_INFINITY;
  let maxPrec = -Number.POSITIVE_INFINITY;

  for (let i = 0; i < WORLD_SIZE; i++) {
    const lat = latByY[i] * RADIANS;
    const baseInsolation = Math.max(0, Math.cos(lat - tiltRad)) * (1 + planet.eccentricity * 0.35);
    const elevationForTemperature = Math.max(0, heights[i]);
    const baseTemp = 55 * baseInsolation - 0.0065 * elevationForTemperature - Math.abs(lat) * 0.4;
    const pressureTerm = 12 * Math.log1p(Math.max(0, planet.atmosphereBar));
    const oceanTerm = heights[i] < 0 ? 6 : 0;
    const seasonal = 1 + 0.08 * Math.sin((lat + 0.2 * Math.PI * (tiltRad)) * 2);
    const temp = baseTemp + pressureTerm + oceanTerm + 0.12 * rotationFactor * 100 * seasonal - aerosol * 8;
    const slopeOrographic = 1 + Math.min(1.4, Math.abs(slope[i]) / 400);
    const precipRaw =
      600 + 1800 * baseInsolation * seasonal + 1200 * Math.min(1, rivers[i]) * slopeOrographic + flow[i] * 0.1;
    const precip = Math.max(0, precipRaw - aerosol * 250);

    temperature[i] = clamp(temp, -90, 70);
    precipitation[i] = precip;
    if (temperature[i] < minTemp) minTemp = temperature[i];
    if (temperature[i] > maxTemp) maxTemp = temperature[i];
    if (precip < minPrec) minPrec = precip;
    if (precip > maxPrec) maxPrec = precip;
  }

  return { temperature, precipitation, minTemp, maxTemp, minPrec, maxPrec };
}
