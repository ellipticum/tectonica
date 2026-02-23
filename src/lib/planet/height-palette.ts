export type RgbColor = [number, number, number];

export type ColorStop = {
  t: number;
  color: RgbColor;
};

export const OCEAN_STOPS: ColorStop[] = [
  { t: 0, color: [220, 236, 250] },
  { t: 0.14, color: [188, 216, 243] },
  { t: 0.3, color: [150, 190, 232] },
  { t: 0.5, color: [108, 156, 218] },
  { t: 0.72, color: [70, 120, 195] },
  { t: 0.88, color: [46, 87, 169] },
  { t: 1, color: [30, 61, 143] },
];

export const LAND_STOPS: ColorStop[] = [
  { t: 0, color: [4, 104, 64] },       // 0 m
  { t: 0.118, color: [36, 129, 53] },  // 200 m
  { t: 0.294, color: [215, 179, 95] }, // 500 m
  { t: 0.471, color: [147, 51, 10] },  // 800 m
  { t: 0.706, color: [99, 96, 94] },   // 1200 m
  { t: 0.824, color: [219, 218, 218] }, // 1400 m
  { t: 0.882, color: [253, 253, 251] }, // 1500 m
  { t: 1, color: [247, 246, 244] },    // 1700 m
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
