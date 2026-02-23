export type RgbColor = [number, number, number];

export type ColorStop = {
  t: number;
  color: RgbColor;
};

export const OCEAN_STOPS: ColorStop[] = [
  { t: 0, color: [198, 218, 230] },
  { t: 0.16, color: [166, 196, 216] },
  { t: 0.36, color: [125, 164, 197] },
  { t: 0.58, color: [86, 127, 168] },
  { t: 0.78, color: [52, 91, 135] },
  { t: 1, color: [18, 43, 83] },
];

export const LAND_STOPS: ColorStop[] = [
  { t: 0, color: [202, 208, 161] },
  { t: 0.14, color: [182, 194, 140] },
  { t: 0.3, color: [156, 177, 118] },
  { t: 0.48, color: [169, 166, 113] },
  { t: 0.66, color: [158, 139, 95] },
  { t: 0.8, color: [132, 108, 78] },
  { t: 0.92, color: [96, 75, 56] },
  { t: 1, color: [60, 45, 36] },
];

export const clamp = (value: number, min: number, max: number) => {
  if (value < min) return min;
  if (value > max) return max;
  return value;
};

export const rgbToCss = (rgb: RgbColor) => `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;

export const sampleColorStops = (stops: ColorStop[], t: number): RgbColor => {
  const k = clamp(t, 0, 1);
  for (let i = 1; i < stops.length; i++) {
    const a = stops[i - 1];
    const b = stops[i];
    if (k <= b.t) {
      const local = (k - a.t) / Math.max(1e-6, b.t - a.t);
      return [
        Math.round(a.color[0] + (b.color[0] - a.color[0]) * local),
        Math.round(a.color[1] + (b.color[1] - a.color[1]) * local),
        Math.round(a.color[2] + (b.color[2] - a.color[2]) * local),
      ];
    }
  }

  return stops[stops.length - 1]?.color ?? [255, 255, 255];
};
