import {
  INDEX,
  KILOMETERS_PER_DEGREE,
  RADIANS,
  WORLD_HEIGHT,
  WORLD_SIZE,
  WORLD_WIDTH,
  clamp,
  latByY,
  lonByX,
} from "./core";
import type { PlanetInputs, WorldEventRecord } from "./types";

export function applyEvents(
  planet: PlanetInputs,
  relief: Float32Array,
  events: WorldEventRecord[],
): { relief: Float32Array; aerosol: number } {
  const updated = new Float32Array(relief);
  let aerosolIndex = 0;

  const nearestCell = (lat: number, lon: number) => {
    const y = clamp(Math.round((90 - lat) / (180 / WORLD_HEIGHT)), 0, WORLD_HEIGHT - 1);
    const x =
      ((Math.round((lon + 180) / (360 / WORLD_WIDTH)) % WORLD_WIDTH + WORLD_WIDTH) % WORLD_WIDTH);
    return INDEX(x, y);
  };

  const kmPerCellLat = KILOMETERS_PER_DEGREE * planet.radiusKm / 6371;
  const kmPerCellLon = (lat: number, lon: number) =>
    Math.max(1, KILOMETERS_PER_DEGREE * Math.cos(lat * RADIANS) * Math.cos(lon * RADIANS) * planet.radiusKm / 6371);

  for (const event of events) {
    if (event.kind === "meteorite") {
      const radiusM = ((event.diameterKm * 1000) / 2) || 1;
      const mass = (4 / 3) * Math.PI * Math.pow(radiusM, 3) * event.densityKgM3;
      const energy = 0.5 * mass * Math.pow(event.speedKms * 1000, 2);
      const craterRadiusKm = Math.max(8, Math.pow(energy, 1 / 5) / 2500);
      const craterDepth = Math.min(9000, 800 + (Math.log10(energy + 1) - 10) * 250);
      const centerIndex = nearestCell(event.latitude, event.longitude);
      const cx = centerIndex % WORLD_WIDTH;
      const cy = Math.floor(centerIndex / WORLD_WIDTH);
      const latSpan = craterRadiusKm / kmPerCellLat;
      const lonSpan = craterRadiusKm / kmPerCellLon(event.latitude, event.longitude);

      const minX = Math.floor(cx - lonSpan - 1);
      const maxX = Math.ceil(cx + lonSpan + 1);
      const minY = clamp(Math.floor(cy - latSpan - 1), 0, WORLD_HEIGHT - 1);
      const maxY = clamp(Math.ceil(cy + latSpan + 1), 0, WORLD_HEIGHT - 1);

      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const wrappedX = (x % WORLD_WIDTH + WORLD_WIDTH) % WORLD_WIDTH;
          const target = INDEX(wrappedX, y);
          const dLat = latByY[target] - event.latitude;
          const dLon =
            (((((lonByX[target] - event.longitude + 180) % 360) + 360) % 360) - 180);
          const dKm = Math.sqrt(
            Math.pow(dLat * kmPerCellLat, 2) +
              Math.pow(dLon * KILOMETERS_PER_DEGREE * Math.cos(latByY[target] * RADIANS) * planet.radiusKm / 6371, 2),
          );
          if (dKm <= craterRadiusKm) {
            const falloff = 1 - Math.pow(dKm / craterRadiusKm, 2);
            updated[target] -= craterDepth * Math.max(0, falloff) * 0.5;
            updated[target] = Math.max(-planet.radiusKm * 10, updated[target]);
          }
        }
      }
      aerosolIndex += Math.min(0.45, Math.log10(energy + 1) / 18);
    } else if (event.kind === "oceanShift") {
      const idx = nearestCell(event.latitude, event.longitude);
      updated[idx] += event.magnitude * 0.5;
      for (let i = 0; i < WORLD_SIZE; i++) {
        updated[i] += event.magnitude * 0.15;
      }
    } else {
      const centerIndex = nearestCell(event.latitude, event.longitude);
      const cx = centerIndex % WORLD_WIDTH;
      const cy = Math.floor(centerIndex / WORLD_WIDTH);
      const sign = event.kind === "rift" ? -1 : 1;
      const magnitude = event.magnitude * 40 * sign;
      const radiusCells = Math.max(
        1,
        Math.round(event.radiusKm / (KILOMETERS_PER_DEGREE * planet.radiusKm / 6371))
      );
      for (let y = cy - radiusCells; y <= cy + radiusCells; y++) {
        if (y < 0 || y >= WORLD_HEIGHT) continue;
        for (let x = cx - radiusCells; x <= cx + radiusCells; x++) {
          const wrappedX = (x % WORLD_WIDTH + WORLD_WIDTH) % WORLD_WIDTH;
          const target = INDEX(wrappedX, y);
          const dx = x - cx;
          const dy = y - cy;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist <= radiusCells) {
            const falloff = Math.exp(-dist / Math.max(1, radiusCells));
            if (event.kind === "uplift") {
              updated[target] += magnitude * falloff;
            } else {
              updated[target] += magnitude * falloff;
            }
          }
        }
      }
    }
  }

  return { relief: updated, aerosol: clamp(aerosolIndex, 0, 1) };
}

export function ensureEventEnergy(event: WorldEventRecord): WorldEventRecord {
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
