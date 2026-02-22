"use client";

import { useEffect, useMemo, useRef, type MutableRefObject } from "react";
import { BIOME_COLORS, type WorldDisplayLayer, type SimulationResult } from "@/lib/planet/simulation";

export type ProjectionMode = "equirectangular" | "mercator" | "orthographic";

interface PlanetCanvasProps {
  result: SimulationResult;
  layer: WorldDisplayLayer;
  previewScale?: number;
  projection?: ProjectionMode;
  centerLongitudeDeg?: number;
  centerLatitudeDeg?: number;
  externalCanvasRef?: MutableRefObject<HTMLCanvasElement | null>;
}

function hslToRgb(h: number, s: number, l: number) {
  const sat = s / 100;
  const light = l / 100;
  const hue = h / 360;
  const q =
    light < 0.5
      ? light * (1 + sat)
      : light + sat - light * sat;
  const p = 2 * light - q;

  const hueToRgb = (t0: number) => {
    let t = t0;
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };

  const r = hueToRgb(hue + 1 / 3);
  const g = hueToRgb(hue);
  const b = hueToRgb(hue - 1 / 3);
  return [r * 255, g * 255, b * 255];
}

function clamp(value: number, min: number, max: number) {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

function lerpColor(a: [number, number, number], b: [number, number, number], t: number) {
  const k = clamp(t, 0, 1);
  return [
    Math.round(a[0] + (b[0] - a[0]) * k),
    Math.round(a[1] + (b[1] - a[1]) * k),
    Math.round(a[2] + (b[2] - a[2]) * k),
  ] as [number, number, number];
}

type ColorStop = {
  t: number;
  color: [number, number, number];
};

const OCEAN_STOPS: ColorStop[] = [
  { t: 0, color: [176, 206, 232] },
  { t: 0.18, color: [146, 184, 220] },
  { t: 0.45, color: [95, 145, 198] },
  { t: 0.72, color: [48, 93, 155] },
  { t: 1, color: [15, 40, 95] },
];

const LAND_STOPS: ColorStop[] = [
  { t: 0, color: [101, 196, 108] },
  { t: 0.2, color: [168, 214, 112] },
  { t: 0.42, color: [230, 213, 119] },
  { t: 0.62, color: [216, 164, 95] },
  { t: 0.8, color: [190, 116, 90] },
  { t: 0.9, color: [210, 161, 166] },
  { t: 1, color: [247, 245, 242] },
];

function sampleStops(stops: ColorStop[], t: number): [number, number, number] {
  const k = clamp(t, 0, 1);
  for (let i = 1; i < stops.length; i++) {
    const a = stops[i - 1];
    const b = stops[i];
    if (k <= b.t) {
      const local = (k - a.t) / Math.max(1e-6, b.t - a.t);
      return lerpColor(a.color, b.color, local);
    }
  }
  return stops[stops.length - 1]?.color ?? [255, 255, 255];
}

const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;
const MERCATOR_MAX_LAT = 85;
const MERCATOR_MAX_Y = Math.log(Math.tan(Math.PI / 4 + (MERCATOR_MAX_LAT * DEG_TO_RAD) / 2));

function wrapLongitude(lon: number) {
  let value = lon;
  while (value < -180) value += 360;
  while (value > 180) value -= 360;
  return value;
}

function latLonToSourceXY(lat: number, lon: number, width: number, height: number) {
  const clampedLat = clamp(lat, -90, 90);
  const wrappedLon = wrapLongitude(lon);
  const x = ((wrappedLon + 180) / 360) * width - 0.5;
  const y = ((90 - clampedLat) / 180) * height - 0.5;
  return { x, y };
}

function sampleBilinear(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  x: number,
  y: number,
): [number, number, number, number] {
  const wrappedX = ((x % width) + width) % width;
  const clampedY = clamp(y, 0, height - 1);

  const x0 = Math.floor(wrappedX);
  const y0 = Math.floor(clampedY);
  const x1 = (x0 + 1) % width;
  const y1 = Math.min(height - 1, y0 + 1);

  const tx = wrappedX - x0;
  const ty = clampedY - y0;

  const i00 = (y0 * width + x0) * 4;
  const i10 = (y0 * width + x1) * 4;
  const i01 = (y1 * width + x0) * 4;
  const i11 = (y1 * width + x1) * 4;

  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  const bilerp = (a00: number, a10: number, a01: number, a11: number) => {
    const top = lerp(a00, a10, tx);
    const bottom = lerp(a01, a11, tx);
    return Math.round(lerp(top, bottom, ty));
  };

  return [
    bilerp(data[i00], data[i10], data[i01], data[i11]),
    bilerp(data[i00 + 1], data[i10 + 1], data[i01 + 1], data[i11 + 1]),
    bilerp(data[i00 + 2], data[i10 + 2], data[i01 + 2], data[i11 + 2]),
    bilerp(data[i00 + 3], data[i10 + 3], data[i01 + 3], data[i11 + 3]),
  ];
}

function heightToRgb(height: number, minHeight: number, maxHeight: number): [number, number, number] {
  if (height < 0) {
    const depthT = Math.pow(clamp(-height / Math.max(1, -minHeight), 0, 1), 0.68);
    return sampleStops(OCEAN_STOPS, depthT);
  }

  if (maxHeight <= 0) return OCEAN_STOPS[0]?.color ?? [168, 221, 255];

  const landT = clamp(height / Math.max(1, maxHeight), 0, 1);
  return sampleStops(LAND_STOPS, landT);
}

export function PlanetLayerCanvas({
  result,
  layer,
  previewScale = 1,
  projection = "equirectangular",
  centerLongitudeDeg = 0,
  centerLatitudeDeg = 0,
  externalCanvasRef,
}: PlanetCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { width, height } = result;
  const plateCount = new Set(result.plates).size;
  const minHeight = result.stats.minHeight;
  const maxHeight = result.stats.maxHeight;
  const minTemp = result.stats.minTemperature;
  const maxTemp = result.stats.maxTemperature;
  const minPrec = result.stats.minPrecipitation;
  const maxPrec = result.stats.maxPrecipitation;
  const smoothRadius = previewScale >= 1 ? 0 : previewScale >= 0.5 ? 1 : 2;

  const basePixels = useMemo(() => {
    const pixels = new Uint8ClampedArray(width * height * 4);

    for (let i = 0; i < width * height; i++) {
      let r = 0;
      let g = 0;
      let b = 0;

      if (layer === "plates") {
        const p = result.plates[i];
        const hue = ((p / Math.max(1, plateCount)) * 360 + 20) % 360;
        [r, g, b] = hslToRgb(hue, 60, 42);
      }

      if (layer === "height") {
        const x = i % width;
        const y = Math.floor(i / width);
        let heightValue = result.heightMap[i];
        const useRadius = smoothRadius;

        if (useRadius > 0) {
          let sum = 0;
          let count = 0;
          for (let oy = -useRadius; oy <= useRadius; oy++) {
            const ny = clamp(y + oy, 0, height - 1);
            for (let ox = -useRadius; ox <= useRadius; ox++) {
              const nx = (x + ox + width) % width;
              sum += result.heightMap[ny * width + nx];
              count++;
            }
          }
          heightValue = sum / Math.max(1, count);
        }

        [r, g, b] = heightToRgb(heightValue, minHeight, maxHeight);
      }

      if (layer === "slope") {
        const t = clamp(result.slopeMap[i] / Math.max(1, result.stats.maxSlope), 0, 1);
        [r, g, b] = hslToRgb(25 + t * 120, 80, 18 + t * 50);
      }

      if (layer === "rivers") {
        const v = clamp(result.riverMap[i], 0, 1);
        [r, g, b] = [18, 38, 84];
        if (v > 0.7) {
          const alpha = clamp(v, 0.4, 1);
          r = 25 + 220 * alpha;
          g = 150 + 60 * alpha;
          b = 240;
        } else if (result.lakeMap[i]) {
          r = 44;
          g = 108;
          b = 140;
        } else {
          const t = clamp(result.heightMap[i] / Math.max(1, maxHeight), 0, 1);
          [r, g, b] = hslToRgb(210, 40, 20 + 40 * t);
        }
      }

      if (layer === "precipitation") {
        const t = clamp((result.precipitationMap[i] - minPrec) / Math.max(1, maxPrec - minPrec), 0, 1);
        [r, g, b] = hslToRgb(210 - t * 180, 85, 20 + t * 55);
      }

      if (layer === "temperature") {
        const t = clamp((result.temperatureMap[i] - minTemp) / Math.max(1, maxTemp - minTemp), 0, 1);
        [r, g, b] = hslToRgb(240 - t * 260, 75, 20 + t * 55);
      }

      if (layer === "biomes") {
        [r, g, b] = BIOME_COLORS[result.biomeMap[i]] ?? [0, 0, 0];
      }

      if (layer === "settlement") {
        const t = clamp(result.settlementMap[i], 0, 1);
        const glow = 16 + 220 * t;
        r = glow;
        g = glow;
        b = glow;
      }

      if (layer === "events") {
        [r, g, b] = [14, 14, 14];
        if (result.flowDirection[i] >= 0) {
          [r, g, b] = [70, 80, 130];
        }
        if (result.riverMap[i] > 0.8) {
          r = 0;
          g = 190;
          b = 255;
        }
      }

      const o = i * 4;
      pixels[o] = r;
      pixels[o + 1] = g;
      pixels[o + 2] = b;
      pixels[o + 3] = layer === "height" || layer === "rivers" || layer === "events" ? 255 : 220;
      if (layer === "rivers" && result.riverMap[i] < 0.05 && !result.lakeMap[i]) {
        pixels[o + 3] = 80;
      }
    }

    return pixels;
  }, [
    height,
    layer,
    maxHeight,
    maxPrec,
    maxTemp,
    minHeight,
    minPrec,
    minTemp,
    plateCount,
    result,
    smoothRadius,
    width,
  ]);

  useEffect(() => {
    if (!externalCanvasRef) {
      return;
    }
    externalCanvasRef.current = canvasRef.current;
    return () => {
      externalCanvasRef.current = null;
    };
  }, [externalCanvasRef]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    const sourceWidth = width;
    const sourceHeight = height;
    const scale = Math.max(0.125, previewScale);
    const supersampledWidth = Math.max(1, Math.round(sourceWidth * scale));
    const supersampledHeight = Math.max(1, Math.round(sourceHeight * scale));

    const sourceCanvas = document.createElement("canvas");
    sourceCanvas.width = sourceWidth;
    sourceCanvas.height = sourceHeight;
    const sourceCtx = sourceCanvas.getContext("2d");
    if (!sourceCtx) {
      return;
    }

    const out = sourceCtx.createImageData(sourceWidth, sourceHeight);
    out.data.set(basePixels);
    sourceCtx.putImageData(out, 0, 0);

    let renderCanvas: HTMLCanvasElement = sourceCanvas;
    const normalizedLon = wrapLongitude(centerLongitudeDeg);
    const normalizedLat = clamp(centerLatitudeDeg, -89.5, 89.5);
    const needsProjection =
      projection !== "equirectangular" || Math.abs(normalizedLon) > 0.001 || Math.abs(normalizedLat) > 0.001;

    if (needsProjection) {
      const projectedCanvas = document.createElement("canvas");
      projectedCanvas.width = sourceWidth;
      projectedCanvas.height = sourceHeight;
      const projectedCtx = projectedCanvas.getContext("2d");
      if (projectedCtx) {
        const projected = projectedCtx.createImageData(sourceWidth, sourceHeight);
        const projectedPx = projected.data;

        if (projection === "orthographic") {
          const radius = Math.min(sourceWidth, sourceHeight) * 0.48;
          const cx = sourceWidth / 2;
          const cy = sourceHeight / 2;

          const lat0 = normalizedLat * DEG_TO_RAD;
          const lon0 = normalizedLon * DEG_TO_RAD;
          const sinLat0 = Math.sin(lat0);
          const cosLat0 = Math.cos(lat0);
          const sinLon0 = Math.sin(lon0);
          const cosLon0 = Math.cos(lon0);

          const eastX = -sinLon0;
          const eastY = cosLon0;
          const eastZ = 0;
          const northX = -sinLat0 * cosLon0;
          const northY = -sinLat0 * sinLon0;
          const northZ = cosLat0;
          const forwardX = cosLat0 * cosLon0;
          const forwardY = cosLat0 * sinLon0;
          const forwardZ = sinLat0;

          for (let py = 0; py < sourceHeight; py++) {
            for (let px = 0; px < sourceWidth; px++) {
              const dx = (px + 0.5 - cx) / radius;
              const dy = (py + 0.5 - cy) / radius;
              const r2 = dx * dx + dy * dy;
              const outOffset = (py * sourceWidth + px) * 4;

              if (r2 > 1) {
                projectedPx[outOffset] = 0;
                projectedPx[outOffset + 1] = 0;
                projectedPx[outOffset + 2] = 0;
                projectedPx[outOffset + 3] = 0;
                continue;
              }

              const z = Math.sqrt(Math.max(0, 1 - r2));
              const vx = dx;
              const vy = -dy;
              const vz = z;

              const wx = vx * eastX + vy * northX + vz * forwardX;
              const wy = vx * eastY + vy * northY + vz * forwardY;
              const wz = vx * eastZ + vy * northZ + vz * forwardZ;

              const lat = Math.asin(clamp(wz, -1, 1)) * RAD_TO_DEG;
              const lon = Math.atan2(wy, wx) * RAD_TO_DEG;
              const sample = latLonToSourceXY(lat, lon, sourceWidth, sourceHeight);
              const [r, g, b, a] = sampleBilinear(basePixels, sourceWidth, sourceHeight, sample.x, sample.y);

              const shading = clamp(0.74 + 0.26 * (0.75 * vz + 0.25 * vx), 0.55, 1.05);
              projectedPx[outOffset] = Math.round(clamp(r * shading, 0, 255));
              projectedPx[outOffset + 1] = Math.round(clamp(g * shading, 0, 255));
              projectedPx[outOffset + 2] = Math.round(clamp(b * shading, 0, 255));
              projectedPx[outOffset + 3] = a;
            }
          }
        } else {
          for (let py = 0; py < sourceHeight; py++) {
            for (let px = 0; px < sourceWidth; px++) {
              const u = (px + 0.5) / sourceWidth;
              const v = (py + 0.5) / sourceHeight;
              const lon = wrapLongitude(u * 360 - 180 + normalizedLon);

              let lat = 90 - v * 180;
              if (projection === "mercator") {
                const yMerc = (1 - 2 * v) * MERCATOR_MAX_Y;
                lat = (2 * Math.atan(Math.exp(yMerc)) - Math.PI / 2) * RAD_TO_DEG;
              }

              const sample = latLonToSourceXY(lat, lon, sourceWidth, sourceHeight);
              const [r, g, b, a] = sampleBilinear(basePixels, sourceWidth, sourceHeight, sample.x, sample.y);
              const outOffset = (py * sourceWidth + px) * 4;
              projectedPx[outOffset] = r;
              projectedPx[outOffset + 1] = g;
              projectedPx[outOffset + 2] = b;
              projectedPx[outOffset + 3] = a;
            }
          }
        }

        projectedCtx.putImageData(projected, 0, 0);
        renderCanvas = projectedCanvas;
      }
    }

    canvas.width = supersampledWidth;
    canvas.height = supersampledHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const crispHeight = layer === "plates";
    ctx.imageSmoothingEnabled = !crispHeight;
    ctx.imageSmoothingQuality = scale >= 1 ? "high" : "low";
    ctx.drawImage(
      renderCanvas,
      0,
      0,
      sourceWidth,
      sourceHeight,
      0,
      0,
      canvas.width,
      canvas.height,
    );
  }, [
    basePixels,
    centerLatitudeDeg,
    centerLongitudeDeg,
    height,
    layer,
    projection,
    previewScale,
    width,
  ]);

  return (
    <div className="relative w-full overflow-auto rounded-2xl border border-border bg-black/35 shadow-[0_10px_40px_-24px_rgba(34,211,238,0.6)]">
      <canvas
        ref={canvasRef}
        className="relative block rounded-2xl"
        style={{
          width: "100%",
          height: "auto",
          imageRendering: previewScale < 0.75 ? "pixelated" : "auto",
        }}
      />
    </div>
  );
}
