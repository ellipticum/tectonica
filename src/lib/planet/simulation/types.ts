export type LayerId =
  | "planet"
  | "plates"
  | "relief"
  | "hydrology"
  | "climate"
  | "biomes"
  | "settlement";

export type RecomputeTrigger = "global" | "tectonics" | "events";
export type GenerationScope = "planet" | "island";
export type IslandType = "continental" | "arc" | "hotspot" | "rift";

export interface LayerSpec {
  id: LayerId;
  name: string;
  formula: string;
  inputs: string[];
  dependsOn: LayerId[];
}

export const LAYER_GRAPH: LayerSpec[] = [
  {
    id: "planet",
    name: "РџР»Р°РЅРµС‚Р°СЂРЅР°СЏ РіРµРѕРјРµС‚СЂРёСЏ",
    formula: "РЎС„РµСЂР° РІ РєРј, РіР»РѕР±Р°Р»СЊРЅС‹Рµ РєРѕРЅСЃС‚Р°РЅС‚С‹ (R, g, РїР»РѕС‚РЅРѕСЃС‚СЊ, Р°С‚РјРѕСЃС„РµСЂР°, РЅР°РєР»РѕРЅ, СЌРєСЃС†РµРЅС‚СЂРёСЃРёС‚РµС‚).",
    inputs: [
      "Р Р°РґРёСѓСЃ",
      "g",
      "РџР»РѕС‚РЅРѕСЃС‚СЊ",
      "РџРµСЂРёРѕРґ РІСЂР°С‰РµРЅРёСЏ",
      "РќР°РєР»РѕРЅ РѕСЃРё",
      "Р­РєСЃС†РµРЅС‚СЂРёСЃРёС‚РµС‚",
      "РђС‚Рј. РґР°РІР»РµРЅРёРµ",
      "РћРєРµР°РЅ%",
    ],
    dependsOn: [],
  },
  {
    id: "plates",
    name: "РўРµРєС‚РѕРЅРёРєР°",
    formula: "РўРёРї РіСЂР°РЅРёС†С‹ РѕРїСЂРµРґРµР»С‘РЅ РїРѕ РІРµРєС‚РѕСЂРЅРѕР№ СЂР°Р·РЅРёС†Рµ СЃРєРѕСЂРѕСЃС‚РµР№ РїР»РёС‚. Р“РѕСЂРЅР°СЏ РІС‹СЃРѕС‚Р° H = k Г— v_convergence / g.",
    inputs: ["РљРѕР»-РІРѕ РїР»РёС‚", "РЎРєРѕСЂРѕСЃС‚СЊ РїР»РёС‚", "РўРµРїР»РѕРІРѕР№ РїРѕС‚РѕРє", "Р“Р»РѕР±Р°Р»СЊРЅР°СЏ РіРµРѕРјРµС‚СЂРёСЏ"],
    dependsOn: ["planet"],
  },
  {
    id: "relief",
    name: "Р РµР»СЊРµС„",
    formula:
      "HeightMap = Р±Р°Р·РѕРІР°СЏ С‚РµРєС‚РѕРЅРёРєР° + РѕРіСЂР°РЅРёС‡РµРЅРЅС‹Р№ СЃС‚РѕС…Р°СЃС‚РёС‡РµСЃРєРёР№ С€СѓРј + РёС‚РµСЂР°С‚РёРІРЅР°СЏ СЌСЂРѕР·РёСЏ + СЃРѕР±С‹С‚РёСЏ.",
    inputs: ["РўРµРєС‚РѕРЅРёРєР°", "РўРµРїР»РѕРІРѕР№ РїРѕС‚РѕРє", "РЎРѕР±С‹С‚РёСЏ", "РћРєРµР°РЅСЃРєРёР№ СѓСЂРѕРІРµРЅСЊ"],
    dependsOn: ["plates", "planet"],
  },
  {
    id: "hydrology",
    name: "Р“РёРґСЂРѕР»РѕРіРёСЏ",
    formula:
      "РЎС…РµРјР°: СѓРєР»РѕРЅ -> РЅР°РїСЂР°РІР»РµРЅРёРµ РїРѕС‚РѕРєР° -> РЅР°РєРѕРїР»РµРЅРёРµ РІРѕРґС‹ -> СЂРµР»СЊРµС„РЅС‹Р№ СЂР°Р·СЂРµР· Рё РѕР·С‘СЂР° РІ Р»РѕРєР°Р»СЊРЅС‹С… РјРёРЅРёРјСѓРјР°С….",
    inputs: ["Р РµР»СЊРµС„", "РЎРѕР±С‹С‚РёСЏ", "РћРєРµР°РЅ%"],
    dependsOn: ["relief"],
  },
  {
    id: "climate",
    name: "РљР»РёРјР°С‚",
    formula:
      "T, P РїРѕ С€РёСЂРѕС‚Рµ (РёРЅСЃРѕР»СЏС†РёСЏ), РІРµС‚СЂСѓ Рё РІС‹СЃРѕС‚Рµ; Р°СЌСЂРѕР·РѕР»Рё РѕС‚ РјРµС‚РµРѕСЂРёС‚РѕРІ СѓРјРµРЅСЊС€Р°СЋС‚ С‚РµРјРїРµСЂР°С‚СѓСЂСѓ.",
    inputs: ["РџР»Р°РЅРµС‚Р°СЂРЅР°СЏ РіРµРѕРјРµС‚СЂРёСЏ", "Р РµР»СЊРµС„", "Р“РёРґСЂРѕР»РѕРіРёСЏ", "РЎРѕР±С‹С‚РёСЏ"],
    dependsOn: ["planet", "relief", "hydrology"],
  },
  {
    id: "biomes",
    name: "Р‘РёРѕРјС‹",
    formula: "Biome = f(Temperature, Precipitation, Elevation).",
    inputs: ["РўРµРјРїРµСЂР°С‚СѓСЂР°", "РћСЃР°РґРєРё", "Р’С‹СЃРѕС‚Р°"],
    dependsOn: ["climate", "relief"],
  },
  {
    id: "settlement",
    name: "Р Р°СЃСЃРµР»РµРЅРёРµ",
    formula:
      "РџРѕС‚РµРЅС†РёР°Р» Р·Р°СЃРµР»РµРЅРёСЏ = РјСЏРіРєР°СЏ С„СѓРЅРєС†РёСЏ РєРѕРјС„РѕСЂС‚Р° РїРѕ T, P Рё СѓРєР»РѕРЅСѓ + С€С‚СЂР°С„ Р·Р° РІС‹СЃРѕС‚Сѓ Рё РїРѕРґРІРѕРґРЅС‹Рµ РєР»РµС‚РєРё.",
    inputs: ["Р‘РёРѕРјС‹", "РўРѕРїРѕРіСЂР°С„РёСЏ", "РљР»РёРјР°С‚"],
    dependsOn: ["biomes", "climate", "relief"],
  },
];

// Whittaker biome classification — 12 types matching Rust classify_biome_whittaker()
export const BIOME_NAMES = [
  "Океан",           // 0 — Ocean
  "Тундра",          // 1 — Tundra/Ice
  "Тайга",           // 2 — Boreal/Taiga
  "Лес",             // 3 — Temperate Forest
  "Луга",            // 4 — Temperate Grassland
  "Средиземноморье", // 5 — Mediterranean
  "Тропический лес", // 6 — Tropical Rainforest
  "Саванна",         // 7 — Tropical Savanna
  "Пустыня",         // 8 — Desert
  "Субтропический лес", // 9 — Subtropical Forest
  "Высокогорье",     // 10 — Alpine
  "Степь",           // 11 — Steppe
];

// Naturalistic palette — Olson/MODIS/Whittaker textbook synthesis,
// maximising perceptual distance between neighbours on the map.
export const BIOME_COLORS: Record<number, [number, number, number]> = {
  0:  [15,  40,  80],   // Ocean — deep indigo
  1:  [170, 190, 210],  // Tundra — cold grey-blue
  2:  [40,  80,  50],   // Boreal/Taiga — dark conifer
  3:  [50,  130, 50],   // Temperate Forest — saturated green
  4:  [160, 190, 60],   // Temperate Grassland — fresh grass
  5:  [175, 120, 60],   // Mediterranean — terracotta
  6:  [0,   70,  35],   // Tropical Rainforest — deep emerald
  7:  [185, 170, 65],   // Tropical Savanna — dry grass
  8:  [210, 180, 110],  // Desert — sand
  9:  [25,  120, 70],   // Subtropical Forest — warm teal
  10: [145, 135, 150],  // Alpine — cool stone lavender
  11: [195, 175, 100],  // Steppe — faded yellow
};

export type WorldDisplayLayer =
  | "plates"
  | "height"
  | "slope"
  | "rivers"
  | "precipitation"
  | "temperature"
  | "biomes"
  | "settlement"
  | "events";

export interface PlanetInputs {
  radiusKm: number;
  gravity: number;
  density: number;
  rotationHours: number;
  axialTiltDeg: number;
  eccentricity: number;
  atmosphereBar: number;
  oceanPercent: number;
}

export interface TectonicInputs {
  plateCount: number;
  plateSpeedCmPerYear: number;
  mantleHeat: number;
}

export type EventKind = "meteorite" | "rift" | "subduction" | "uplift" | "oceanShift";

export interface MeteoriteEvent {
  kind: "meteorite";
  latitude: number;
  longitude: number;
  diameterKm: number;
  speedKms: number;
  angleDeg: number;
  densityKgM3: number;
}

export interface RegionEvent {
  kind: "rift" | "subduction" | "uplift" | "oceanShift";
  latitude: number;
  longitude: number;
  radiusKm: number;
  magnitude: number;
}

export type WorldEvent = MeteoriteEvent | RegionEvent;

export type WorldEventRecord = WorldEvent & {
  id: string;
  createdAt: string;
  summary: string;
  energyJoule?: number;
};

export interface SimulationConfig {
  seed: number;
  planet: PlanetInputs;
  tectonics: TectonicInputs;
  events: WorldEventRecord[];
  generationPreset?: "ultra" | "fast" | "balanced" | "detailed";
  scope?: GenerationScope;
  /** Island tectonic type (only used when scope === "island") */
  islandType?: IslandType;
  /** Physical width of the island grid in km (50–2000). Default: 400. */
  islandScaleKm?: number;
}

export interface SimulationStats {
  minHeight: number;
  maxHeight: number;
  minTemperature: number;
  maxTemperature: number;
  minPrecipitation: number;
  maxPrecipitation: number;
  minSlope: number;
  maxSlope: number;
}

export interface SimulationResult {
  width: number;
  height: number;
  seed: number;
  specs: LayerSpec[];
  recomputedLayers: LayerId[];
  planet: {
    seaLevel: number;
    radiusKm: number;
    oceanPercent: number;
  };
  plates: Int16Array;
  boundaryTypes: Int8Array;
  heightMap: Float32Array;
  slopeMap: Float32Array;
  riverMap: Float32Array;
  lakeMap: Uint8Array;
  flowDirection: Int32Array;
  flowAccumulation: Float32Array;
  temperatureMap: Float32Array;
  precipitationMap: Float32Array;
  biomeMap: Uint8Array;
  settlementMap: Float32Array;
  eventHistory: WorldEventRecord[];
  stats: SimulationStats;
}

export interface HeightExportPayload {
  format: "float32" | "int16";
  width: number;
  height: number;
  radiusKm: number;
  scaleFactor?: number;
  values: number[];
}
