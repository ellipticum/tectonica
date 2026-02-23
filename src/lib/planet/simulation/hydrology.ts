import {
  INDEX,
  INDEX_SPHERICAL,
  WORLD_HEIGHT,
  WORLD_SIZE,
  WORLD_WIDTH,
} from "./core";

export function computeSlope(heights: Float32Array) {
  const slope = new Float32Array(WORLD_SIZE);
  let minSlope = Number.POSITIVE_INFINITY;
  let maxSlope = 0;

  for (let y = 0; y < WORLD_HEIGHT; y++) {
    for (let x = 0; x < WORLD_WIDTH; x++) {
      const i = INDEX(x, y);
      const h = heights[i];
      let maxDrop = 0;
      for (let oy = -1; oy <= 1; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          if (ox === 0 && oy === 0) continue;
          const j = INDEX_SPHERICAL(x + ox, y + oy);
          const drop = Math.max(0, h - heights[j]);
          if (drop > maxDrop) {
            maxDrop = drop;
          }
        }
      }
      slope[i] = maxDrop;
      if (maxDrop < minSlope) minSlope = maxDrop;
      if (maxDrop > maxSlope) maxSlope = maxDrop;
    }
  }

  return { slope, minSlope, maxSlope };
}

export function computeHydrology(heights: Float32Array, slope: Float32Array) {
  const flowDirection = new Int32Array(WORLD_SIZE);
  flowDirection.fill(-1);
  const flowAccumulation = new Float32Array(WORLD_SIZE);

  for (let i = 0; i < WORLD_SIZE; i++) {
    flowAccumulation[i] = 1;
  }

  for (let y = 0; y < WORLD_HEIGHT; y++) {
    for (let x = 0; x < WORLD_WIDTH; x++) {
      const i = INDEX(x, y);
      const h = heights[i];
      let bestDrop = 0;
      let best = -1;
      for (let oy = -1; oy <= 1; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          if (ox === 0 && oy === 0) continue;
          const j = INDEX_SPHERICAL(x + ox, y + oy);
          const drop = h - heights[j];
          if (drop > bestDrop) {
            bestDrop = drop;
            best = j;
          }
        }
      }
      flowDirection[i] = best;
    }
  }

  const order = Array.from({ length: WORLD_SIZE }, (_, i) => i).sort((a, b) => heights[b] - heights[a]);
  for (const i of order) {
    const to = flowDirection[i];
    if (to >= 0) {
      flowAccumulation[to] += flowAccumulation[i] * (1 + slope[i] / 1000);
    }
  }

  const sorted = Array.from(flowAccumulation).sort((a, b) => b - a);
  const threshold = sorted[Math.max(0, Math.floor(sorted.length * 0.985))] ?? 1;

  const rivers = new Float32Array(WORLD_SIZE);
  const lakes = new Uint8Array(WORLD_SIZE);
  for (let i = 0; i < WORLD_SIZE; i++) {
    rivers[i] = flowAccumulation[i] / threshold;
    if (flowDirection[i] < 0 && heights[i] < 0 && slope[i] < 40) {
      lakes[i] = 1;
    }
  }

  return { flowDirection, flowAccumulation, rivers, lakes };
}
