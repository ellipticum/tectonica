import { BIOME_COLORS, BIOME_NAMES, type SimulationResult, type WorldDisplayLayer } from "@/lib/planet/simulation";
import { LAND_STOPS, OCEAN_STOPS, clamp, rgbToCss, sampleColorStops } from "@/lib/planet/height-palette";

export type LegendItem = { color: string; label: string };

export const formatNum = (value: number) =>
  value.toLocaleString("ru-RU", { maximumFractionDigits: 2 });

const hslToCss = (hue: number, saturation: number, lightness: number) =>
  `hsl(${hue.toFixed(0)} ${Math.round(saturation)}% ${Math.round(lightness)}%)`;

const heightColor = (value: number, min: number, max: number) => {
  if (value < 0 && max > 0) {
    const t = Math.pow(clamp(-value / Math.max(1, -min), 0, 1), 0.68);
    return rgbToCss(sampleColorStops(OCEAN_STOPS, t));
  }

  if (max <= 0) {
    return rgbToCss(sampleColorStops(OCEAN_STOPS, 1));
  }

  const t = Math.pow(clamp(value / Math.max(1, max), 0, 1), 0.72);
  return rgbToCss(sampleColorStops(LAND_STOPS, t));
};

const slopeLegendColor = (slope: number, maxSlope: number) => {
  const t = clamp(slope, 0, maxSlope) / Math.max(1, maxSlope);
  return hslToCss(25 + t * 120, 80, 18 + t * 50);
};

const climateLegendColor = (value: number, min: number, max: number, temp = false) => {
  const t = (value - min) / Math.max(1, max - min);
  if (temp) {
    return hslToCss(240 - t * 260, 75, 20 + t * 55);
  }
  return hslToCss(210 - t * 180, 85, 20 + t * 55);
};

export const buildHeightGradient = (minHeight: number, maxHeight: number) => {
  if (maxHeight <= 0) {
    const oceanOnlyStops = OCEAN_STOPS.map((s) => {
      const color = sampleColorStops(OCEAN_STOPS, 1 - s.t);
      return `${rgbToCss(color)} ${(s.t * 100).toFixed(2)}%`;
    });
    return `linear-gradient(90deg, ${oceanOnlyStops.join(", ")})`;
  }

  if (minHeight >= 0) {
    return `linear-gradient(90deg, ${LAND_STOPS.map((s) => `${rgbToCss(s.color)} ${(s.t * 100).toFixed(2)}%`).join(", ")})`;
  }

  const seaSplit = clamp((0 - minHeight) / Math.max(1, maxHeight - minHeight), 0.08, 0.92);
  const oceanStops = OCEAN_STOPS.map((s) => {
    const color = sampleColorStops(OCEAN_STOPS, 1 - s.t);
    return `${rgbToCss(color)} ${(s.t * seaSplit * 100).toFixed(2)}%`;
  });
  const landStops = LAND_STOPS.map((s) =>
    `${rgbToCss(s.color)} ${(seaSplit * 100 + s.t * (1 - seaSplit) * 100).toFixed(2)}%`,
  );
  return `linear-gradient(90deg, ${[...oceanStops, ...landStops].join(", ")})`;
};

export const buildLayerLegend = (layer: WorldDisplayLayer, result: SimulationResult): LegendItem[] => {
  const minHeight = result.stats.minHeight;
  const maxHeight = result.stats.maxHeight;
  const maxSlope = result.stats.maxSlope;
  const minTemp = result.stats.minTemperature;
  const maxTemp = result.stats.maxTemperature;
  const minPrec = result.stats.minPrecipitation;
  const maxPrec = result.stats.maxPrecipitation;

  if (layer === "biomes") {
    return BIOME_NAMES.map((name, index) => {
      const color = BIOME_COLORS[index] ?? [255, 255, 255];
      return {
        color: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
        label: `${index}. ${name}`,
      };
    });
  }

  if (layer === "plates") {
    const plateSet = new Set(result.plates);
    const plateIds = Array.from(plateSet).sort((a, b) => a - b);
    const limit = Math.min(plateIds.length, 12);

    const entries: LegendItem[] = plateIds.slice(0, limit).map((plateId) => {
      const hue = ((plateId / Math.max(1, plateIds.length)) * 360 + 20) % 360;
      return {
        color: hslToCss(hue, 60, 42),
        label: `Плита ${plateId + 1}`,
      };
    });

    if (plateIds.length > limit) {
      entries.push({
        color: "rgba(255,255,255,0.25)",
        label: `...и еще ${plateIds.length - limit} шт.`,
      });
    }

    return entries;
  }

  if (layer === "height") {
    const vMin = minHeight;
    const vMax = maxHeight;
    return [
      { color: heightColor(vMin, vMin, vMax), label: `Глубокий океан (min ${formatNum(vMin)} м)` },
      { color: heightColor(vMin * 0.35, vMin, vMax), label: "Мелкая вода (пастельно-голубая)" },
      { color: heightColor(vMin * 0.75, vMin, vMax), label: "Глубокая вода (кобальтово-синяя)" },
      { color: heightColor(0, vMin, vMax), label: "Уровень моря: 0 м (насыщенный зелёный)" },
      { color: heightColor(vMax * 0.2, vMin, vMax), label: "Низины суши (зелёный)" },
      { color: heightColor(vMax * 0.55, vMin, vMax), label: "Возвышенности (жёлто-охристый)" },
      { color: heightColor(vMax * 0.85, vMin, vMax), label: "Высокогорья (серо-светлые)" },
      { color: heightColor(vMax, vMin, vMax), label: `Пики: ~${formatNum(vMax)} м (почти белый)` },
    ];
  }

  if (layer === "slope") {
    const max = Math.max(1, maxSlope);
    return [
      { color: slopeLegendColor(0, maxSlope), label: "Пологое / слабый уклон" },
      { color: slopeLegendColor(0.2 * max, maxSlope), label: "Лёгкий уклон" },
      { color: slopeLegendColor(0.45 * max, maxSlope), label: "Умеренный уклон" },
      { color: slopeLegendColor(0.7 * max, maxSlope), label: "Крутой" },
      { color: slopeLegendColor(max, maxSlope), label: "Очень крутой" },
    ];
  }

  if (layer === "rivers") {
    return [
      { color: "rgb(18, 38, 84)", label: "Низкая/низкоуровневая вода" },
      { color: "rgb(44, 108, 140)", label: "Озера (впадины)" },
      { color: "rgb(128, 174, 255)", label: "Река, средняя сила потока" },
      { color: "rgb(245, 170, 255)", label: "Сильные русла" },
      { color: "rgb(14, 14, 14)", label: "Остальные области (подложка)" },
    ];
  }

  if (layer === "precipitation") {
    const max = maxPrec;
    const min = minPrec;
    return [
      { color: climateLegendColor(minPrec, min, max), label: "Сухо" },
      { color: climateLegendColor(minPrec + (max - min) * 0.25, min, max), label: "Низкие осадки" },
      { color: climateLegendColor(minPrec + (max - min) * 0.5, min, max), label: "Умеренные осадки" },
      { color: climateLegendColor(minPrec + (max - min) * 0.75, min, max), label: "Высокие осадки" },
      { color: climateLegendColor(max, min, max), label: "Экстремально влажные" },
    ];
  }

  if (layer === "temperature") {
    return [
      { color: climateLegendColor(minTemp, minTemp, maxTemp, true), label: "Самый холодный" },
      { color: climateLegendColor(minTemp + (maxTemp - minTemp) * 0.25, minTemp, maxTemp, true), label: "Холодный" },
      { color: climateLegendColor(minTemp + (maxTemp - minTemp) * 0.5, minTemp, maxTemp, true), label: "Умеренный" },
      { color: climateLegendColor(minTemp + (maxTemp - minTemp) * 0.75, minTemp, maxTemp, true), label: "Тёплый" },
      { color: climateLegendColor(maxTemp, minTemp, maxTemp, true), label: "Очень тёплый" },
    ];
  }

  if (layer === "settlement") {
    return [
      { color: "rgb(16, 16, 16)", label: "Нет поселений" },
      { color: "rgb(90, 90, 90)", label: "Слабая активность" },
      { color: "rgb(145, 145, 145)", label: "Умеренная активность" },
      { color: "rgb(190, 190, 190)", label: "Развитая колония" },
      { color: "rgb(236, 236, 236)", label: "Максимальная плотность" },
    ];
  }

  return [
    { color: "rgb(14, 14, 14)", label: "Подложка" },
    { color: "rgb(70, 80, 130)", label: "Признаки гидро- и рельефной динамики" },
    { color: "rgb(0, 190, 255)", label: "Сильные потоки после пересчета событий" },
  ];
};
