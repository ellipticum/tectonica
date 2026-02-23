import {
  INDEX,
  INDEX_SPHERICAL,
  WORLD_HEIGHT,
  WORLD_SIZE,
  WORLD_WIDTH,
  clamp,
  latByY,
  lonByX,
  makeRng,
  quantile,
  randomRange,
} from "./core";
import type { PlanetInputs, TectonicInputs } from "./types";
import type { PlatesComputation } from "./plates";

function normalizeHeightRange(
  relief: Float32Array,
  planet: PlanetInputs,
  tectonics: TectonicInputs,
) {
  const landHeights: number[] = [];
  const oceanDepths: number[] = [];

  for (let i = 0; i < WORLD_SIZE; i++) {
    const h = relief[i];
    if (h > 0) {
      landHeights.push(h);
    } else if (h < 0) {
      oceanDepths.push(-h);
    }
  }

  if (landHeights.length < 8 || oceanDepths.length < 8) {
    return;
  }

  landHeights.sort((a, b) => a - b);
  oceanDepths.sort((a, b) => a - b);

  const landRef = Math.max(1, quantile(landHeights, 0.995));
  const oceanRef = Math.max(1, quantile(oceanDepths, 0.995));

  const speedFactor = clamp(tectonics.plateSpeedCmPerYear / 5, 0.35, 2.8);
  const heatFactor = clamp(tectonics.mantleHeat / 55, 0.45, 2.2);
  const gravityFactor = clamp(9.81 / Math.max(1, planet.gravity), 0.45, 2.4);
  const oceanFactor = clamp(planet.oceanPercent / 67, 0.45, 1.9);

  const targetLandMax = clamp(
    6000 * Math.pow(speedFactor, 0.55) * Math.pow(heatFactor, 0.45) * Math.pow(gravityFactor, 0.7),
    1800,
    9000,
  );
  const targetOceanDepth = clamp(
    7500 * Math.pow(speedFactor, 0.5) * Math.pow(heatFactor, 0.35) * Math.pow(oceanFactor, 0.55),
    1400,
    12000,
  );

  const landScale = targetLandMax / landRef;
  const oceanScale = targetOceanDepth / oceanRef;

  for (let i = 0; i < WORLD_SIZE; i++) {
    const h = relief[i];
    if (h > 0) {
      relief[i] = clamp(h * landScale, -12000, 9000);
    } else if (h < 0) {
      relief[i] = clamp(-(-h * oceanScale), -12000, 9000);
    }
  }
}

function reshapeOceanBoundaries(
  relief: Float32Array,
  boundaryTypes: Int8Array,
  boundaryStrength: Float32Array,
) {
  let deepest = 0;
  for (let i = 0; i < WORLD_SIZE; i++) {
    if (relief[i] < 0) {
      deepest = Math.max(deepest, -relief[i]);
    }
  }
  const depthScale = Math.max(1, deepest);

  for (let i = 0; i < WORLD_SIZE; i++) {
    if (relief[i] >= 0) continue;
    const type = boundaryTypes[i];
    if (type === 0) continue;

    const depthT = clamp(-relief[i] / depthScale, 0, 1);
    const strength = clamp(boundaryStrength[i], 0, 1);

    if (type === 2) {
      // Divergence below sea level should bias toward mid-ocean ridges instead of deep trenches.
      const uplift = (520 + 2200 * strength) * (0.7 + 0.3 * (1 - depthT));
      relief[i] = Math.min(-15, relief[i] + uplift);
    } else if (type === 3) {
      const uplift = (120 + 480 * strength) * (0.65 + 0.35 * (1 - depthT));
      relief[i] += uplift;
    }
  }

  const oceanSmoothed = new Float32Array(relief);
  for (let y = 0; y < WORLD_HEIGHT; y++) {
    for (let x = 0; x < WORLD_WIDTH; x++) {
      const i = INDEX(x, y);
      if (relief[i] >= 0) continue;
      let sum = relief[i] * 3.4;
      let wsum = 3.4;
      for (let oy = -1; oy <= 1; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          if (ox === 0 && oy === 0) continue;
          const j = INDEX_SPHERICAL(x + ox, y + oy);
          if (relief[j] >= 0) continue;
          const w = ox === 0 || oy === 0 ? 0.6 : 0.45;
          sum += relief[j] * w;
          wsum += w;
        }
      }
      oceanSmoothed[i] = sum / Math.max(1e-6, wsum);
    }
  }
  relief.set(oceanSmoothed);
}

function applyCoastalDetail(relief: Float32Array, seed: number) {
  for (let i = 0; i < WORLD_SIZE; i++) {
    const h = relief[i];
    const nearSea = Math.exp(-Math.abs(h) / 900);
    if (nearSea < 0.035) continue;

    const lat = latByY[i];
    const lon = lonByX[i];
    const warpLat = lat + 2.4 * Math.sin(lon * 0.065 + seed * 0.0013);
    const warpLon = lon + 3.6 * Math.cos(lat * 0.058 - seed * 0.0011);
    const n =
      Math.sin(warpLat * 0.19 + warpLon * 0.23 + seed * 0.0022) * 0.62 +
      Math.cos(warpLat * 0.31 - warpLon * 0.17 - seed * 0.0016) * 0.38;
    const coastWeight = Math.pow(nearSea, 1.18);
    relief[i] += coastWeight * n * 92;
  }
}

function cleanupCoastalSpeckles(relief: Float32Array) {
  const next = new Float32Array(relief);

  for (let y = 0; y < WORLD_HEIGHT; y++) {
    for (let x = 0; x < WORLD_WIDTH; x++) {
      const i = INDEX(x, y);
      const h = relief[i];
      if (Math.abs(h) > 260) {
        continue;
      }

      let land = 0;
      let water = 0;
      for (let oy = -1; oy <= 1; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          if (ox === 0 && oy === 0) continue;
          const j = INDEX_SPHERICAL(x + ox, y + oy);
          if (relief[j] >= 0) {
            land++;
          } else {
            water++;
          }
        }
      }

      if (h >= 0 && water >= 7) {
        next[i] = Math.min(next[i], -18 - Math.abs(h) * 0.15);
      } else if (h < 0 && land >= 7) {
        next[i] = Math.max(next[i], 18 + Math.abs(h) * 0.15);
      }
    }
  }

  relief.set(next);
}

export function computeRelief(
  planet: PlanetInputs,
  tectonics: TectonicInputs,
  plates: PlatesComputation,
  seed: number,
) {
  const {
    plateField,
    boundaryTypes,
    boundaryScores,
    boundaryNormalX,
    boundaryNormalY,
    boundaryStrength,
    plateVectors,
  } = plates;
  const relief = new Float32Array(WORLD_SIZE);
  const randomSeed = makeRng(
    (((planet.radiusKm * 1000) | 0) ^
      ((tectonics.plateSpeedCmPerYear * 1000) | 0) ^
      ((tectonics.mantleHeat * 1000) | 0) ^
      ((seed * 2654435761) | 0)) >>> 0,
  );
  const macroA = randomRange(randomSeed, 0.012, 0.03);
  const macroB = randomRange(randomSeed, 0.008, 0.02);
  const macroC = randomRange(randomSeed, 0.006, 0.016);
  const phaseA = randomRange(randomSeed, -Math.PI, Math.PI);
  const phaseB = randomRange(randomSeed, -Math.PI, Math.PI);
  const phaseC = randomRange(randomSeed, -Math.PI, Math.PI);
  const kRelief = 560;
  const maxBoundaryScore = Math.max(1, tectonics.plateSpeedCmPerYear * 2.5);
  const baseKernelRadius = 3;

  for (let i = 0; i < WORLD_SIZE; i++) {
    const plateId = plateField[i];
    const plateSpeed = plateVectors[plateId].speed;
    const heat = plateVectors[plateId].heat;
    const plateBuoyancy = plateVectors[plateId].buoyancy;
    const x = i % WORLD_WIDTH;
    const y = Math.floor(i / WORLD_WIDTH);

    let convInfluence = 0;
    let divInfluence = 0;
    let transformInfluence = 0;

    const localKernel = Math.min(
      6,
      baseKernelRadius + Math.floor(plateSpeed / 1.8) + Math.floor((heat / 30) * 2),
    );

    for (let oy = -localKernel; oy <= localKernel; oy++) {
      for (let ox = -localKernel; ox <= localKernel; ox++) {
        const j = INDEX_SPHERICAL(x + ox, y + oy);
        const t = boundaryTypes[j];
        if (t === 0) continue;

        const sourcePlate = plateField[j];
        const sourceHeat = plateVectors[sourcePlate].heat;
        const ridgeNoise = 0.65 + 0.35 * Math.sin((latByY[j] * 0.11 + lonByX[j] * 0.17 + seed * 0.0027));
        const sourceHeatNorm = sourceHeat / Math.max(1, tectonics.mantleHeat);
        const heatWidth = 0.75 + sourceHeatNorm * 1.6 + (sourceHeatNorm - 0.55) * 0.9;
        const localWidth = clamp(0.65 + heatWidth * ridgeNoise, 0.5, 4.8);
        const s = clamp(Math.abs(boundaryScores[j]) / maxBoundaryScore, 0, 1);
        const nNormalX = boundaryNormalX[j] || 0;
        const nNormalY = boundaryNormalY[j] || 0;
        const normalLen = Math.hypot(nNormalX, nNormalY) || 1;
        const normalX = nNormalX / normalLen;
        const normalY = nNormalY / normalLen;
        const tangentX = -normalY;
        const tangentY = normalX;
        const across = ox * normalX + oy * normalY;
        const along = ox * tangentX + oy * tangentY;
        const boundaryBias = clamp(boundaryStrength[j] * 1.25, 0, 1.4);
        const sigmaAcross = Math.max(0.55, localWidth * (0.85 + 1.1 * s) + 0.05 * Math.sin(ox * 3.1 + oy * 2.3));
        const sigmaAlong = Math.max(
          1.6,
          localWidth * (3.0 + 3.2 * boundaryBias + 1.6 * s) + 1.0 * Math.cos(oy * 2.1 - ox * 1.9),
        );
        const anisotropy =
          (across * across) / (sigmaAcross * sigmaAcross) + (along * along) / (sigmaAlong * sigmaAlong);
        const w = Math.exp(-anisotropy);
        const widthTuning = 0.32 + 0.95 * s + 0.28 * boundaryBias;
        const chainSegment =
          0.35 + 0.65 * (0.5 + 0.5 * Math.sin(latByY[j] * 0.27 + lonByX[j] * 0.34 + phaseA));
        const chainCluster =
          0.45 + 0.55 * (0.5 + 0.5 * Math.cos(latByY[j] * 0.22 - lonByX[j] * 0.19 + phaseB));
        let intensity = w * widthTuning;

        if (t === 1) {
          intensity *= chainSegment * chainCluster;
          convInfluence += intensity;
        } else if (t === 2) {
          intensity *= 0.55 + 0.45 * chainSegment;
          divInfluence += intensity;
        } else {
          transformInfluence += intensity;
        }
      }
    }

    const lat = latByY[i];
    const lon = lonByX[i];
    const warpLat = lat + 11 * Math.sin(lon * 0.045 + phaseA) + 6 * Math.cos(lat * 0.062 + phaseB);
    const warpLon = lon + 14 * Math.cos(lat * 0.037 + phaseC) - 7 * Math.sin(lon * 0.053 + phaseA);
    const continentalSignal =
      Math.sin(warpLat * 0.018 + phaseA) * 0.95 +
      Math.cos(warpLon * 0.013 + phaseB) * 0.8 +
      Math.sin((warpLat + warpLon) * 0.0105 + phaseC) * 0.55 +
      Math.cos(warpLat * 0.023 - warpLon * 0.019 + phaseA * 0.65) * 0.4;
    const regionalSignal = Math.sin((lat + lon) * 0.072 + phaseB) * Math.cos((lat - lon) * 0.059 + phaseC);
    const crustMaskRaw = continentalSignal * 0.95 + regionalSignal * 0.45 + plateBuoyancy * 1.2;
    const crustMask = 1 / (1 + Math.exp(-crustMaskRaw));

    let base = 0;
    if (convInfluence > 0.03) {
      const convShape = Math.pow(convInfluence, 0.78);
      base += ((kRelief * plateSpeed) / Math.max(1, planet.gravity)) * (0.04 + 0.12 * Math.atan(convShape));
      base += 105 * Math.log1p(convShape);
    }
    if (divInfluence > 0.03) {
      const divShape = Math.pow(divInfluence, 0.84);
      base -= (95 + 6 * plateSpeed) * (0.06 + 0.18 * Math.atan(divShape));
    }
    if (transformInfluence > 0.03) {
      base += 18 * (plateSpeed - 1) * Math.atan(transformInfluence);
    }
    const tectonicWeight = 0.16 + 0.84 * crustMask;
    base *= tectonicWeight;
    const intraplateSignal =
      Math.sin((lat * 0.083 + lon * 0.064) + plateId * 1.77 + phaseA * 0.25) *
      Math.cos((lat * 0.059 - lon * 0.051) + plateId * 0.91 + phaseB * 0.3);
    const macroNoise =
      Math.sin(lat * macroA + lon * macroB + phaseA) * 140 +
      Math.sin(lat * (macroA * 0.55) - lon * macroC + phaseB) * 90 +
      Math.cos(lon * (macroB * 1.6) + phaseC) * 70;
    const continentalBase = (crustMask - 0.5) * 3600;
    const macroBase = continentalBase + intraplateSignal * 260 + plateBuoyancy * 420 + regionalSignal * 110;
    const noise = (randomSeed.next() - 0.5) * 78 + macroNoise * 0.45;
    const heatTerm = heat * 1.8;
    relief[i] = base + macroBase + noise + heatTerm;
  }

  const macroBlend = new Float32Array(WORLD_SIZE);
  const macroRadius = 1;
  for (let y = 0; y < WORLD_HEIGHT; y++) {
    for (let x = 0; x < WORLD_WIDTH; x++) {
      const i = INDEX(x, y);
      let sum = 0;
      let wsum = 0;
      for (let oy = -macroRadius; oy <= macroRadius; oy++) {
        for (let ox = -macroRadius; ox <= macroRadius; ox++) {
          const j = INDEX_SPHERICAL(x + ox, y + oy);
          const dist2 = ox * ox + oy * oy;
          const w = Math.exp(-dist2 / 1.9);
          sum += relief[j] * w;
          wsum += w;
        }
      }
      macroBlend[i] = sum / Math.max(1e-6, wsum);
    }
  }
  for (let i = 0; i < WORLD_SIZE; i++) {
    relief[i] = relief[i] * 0.82 + macroBlend[i] * 0.18;
  }

  const erosionRounds = 2;
  const smoothed = new Float32Array(relief);
  const scratch = new Float32Array(WORLD_SIZE);

  for (let round = 0; round < erosionRounds; round++) {
    for (let y = 0; y < WORLD_HEIGHT; y++) {
      for (let x = 0; x < WORLD_WIDTH; x++) {
        const i = INDEX(x, y);
        let sum = 0;
        let count = 0;
        for (let oy = -1; oy <= 1; oy++) {
          for (let ox = -1; ox <= 1; ox++) {
            const j = INDEX_SPHERICAL(x + ox, y + oy);
            sum += smoothed[j];
            count++;
          }
        }
        const avg = sum / count;
        scratch[i] = smoothed[i] * 0.92 + avg * 0.08;
      }
    }

    for (let i = 0; i < WORLD_SIZE; i++) {
      const drop = Math.max(0, smoothed[i] - scratch[i]);
      const heightLoss = Math.min(drop * 0.18, 22);
      smoothed[i] = scratch[i] - heightLoss;
    }
  }

  relief.set(smoothed);

  // Ocean percent normalization.
  const sorted = Array.from(relief).sort((a, b) => a - b);
  const oceanCut = Math.max(
    0,
    Math.min(WORLD_SIZE - 1, Math.floor((planet.oceanPercent / 100) * WORLD_SIZE))
  );
  const seaLevel = sorted[oceanCut] ?? 0;
  for (let i = 0; i < WORLD_SIZE; i++) {
    relief[i] -= seaLevel;
  }

  applyCoastalDetail(relief, seed);
  cleanupCoastalSpeckles(relief);
  const sortedAfterCoast = Array.from(relief).sort((a, b) => a - b);
  const coastRecenter = sortedAfterCoast[oceanCut] ?? 0;
  for (let i = 0; i < WORLD_SIZE; i++) {
    relief[i] -= coastRecenter;
  }

  normalizeHeightRange(relief, planet, tectonics);
  reshapeOceanBoundaries(relief, boundaryTypes, boundaryStrength);
  const sortedFinal = Array.from(relief).sort((a, b) => a - b);
  const finalRecenter = sortedFinal[oceanCut] ?? 0;
  for (let i = 0; i < WORLD_SIZE; i++) {
    relief[i] -= finalRecenter;
  }

  return { relief, seaLevel };
}
