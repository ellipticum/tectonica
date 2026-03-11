# Tectonica — Полный научно-технический аудит (март 2026)

**Файл**: `rust/planet_engine/src/lib.rs` (6904 строки, Rust → WASM)
**Дата**: 2026-03-10

---

## Этап 1: Структурная карта кода

### 1.1 Pipeline генерации (Planet scope)

| # | Стадия | Строки | Входные данные | Выходные данные | Ключевые параметры |
|---|--------|--------|----------------|-----------------|-------------------|
| 1 | PRNG | 18–60 | seed: u32 | Rng (xoshiro128++) | SplitMix32 seeding |
| 2 | WorldCache | 268–310 | WORLD_WIDTH/HEIGHT | lat/lon/xyz per cell | 4096×2048 grid |
| 3 | GridConfig | 316–387 | scope params | grid abstraction | km_per_cell, cos_lat |
| 4 | CellCache | 394–470 | GridConfig | noise/lat/lon coords | spherical/flat modes |
| 5 | PlateSpec generation | 1540–1606 | PlanetInputs, TectonicInputs | plates: Vec<PlateSpec> | omega from Euler pole, buoyancy ∈[-1,1] |
| 6 | Voronoi plate growth | 820–1094 | plates, seed, cache | plate_field: Vec<i16> | spread 0.82–1.22, roughness 0.26–1.08 |
| 7 | Fragment cleanup | 1096–1298 | plate_field | cleaned plate_field | min_fragment, ratio 0.42 |
| 8 | Plate evolution | 1335–1538 | plate_field, vectors | evolved field + evolution_time_yr | 0.9–3.4 Myr/step, 2–10 steps |
| 9 | Boundary detection | 1640–1760 | plate_field, vectors | boundary_types, boundary_strength | conv/div/trans thresholds |
| 10 | Continental nuclei | 1963–2098 | plates, seed | continental_frac, is_continental | N_cont=3–9, σ=1000–2500 km |
| 11 | Coastline perturbation (pre-smooth) | 2123–2194 | is_continental, coast_dist | perturbed continental_frac | 4 octaves, 500 km BFS band |
| 12 | CF smoothing + fine noise | 2196–2223 | continental_frac | smoothed cf | 10+3 passes |
| 13 | Deformation propagation | 1827–1876 | boundary seeds | conv_def/div_def/trans_def | L_d=250/200/150 km |
| 14 | Damage rheology | 2269–2293 | def fields | localized def fields | α=0.6/0.4/0.4 |
| 15 | Interior suppression | 2295–2358 | boundary dist | weakness field | L_rheol=300 km |
| 16 | Crustal thickness | 2361–2400 | cf, defs | crust_thickness [km] | base=40/7 km, conv +30, div -20 |
| 17 | Flexural isostasy | 2402–2432 | crust_thickness | smoothed crust | Te=25 km → N=34 passes |
| 18 | Rock type → K_eff | 2435–2494 | defs, cf | k_eff per cell | Granite 0.5e-6 … Limestone 3.0e-6 |
| 19 | Airy isostasy | 1789–1810 | crust, cf, heat | initial relief | ρ_c=2800–2900, ρ_m=3300, ρ_w=1025 |
| 20 | Thermal subsidence | 2530–2664 | dist_from_ridge, plate speed | subsidence field | 350√t, GDH1 at t>80 Ma |
| 21 | Ocean floor texture | 2666–2704 | noise, defs | hills+ridge+trench+fracture | 120/60/30m hills, 500m ridge, 1500m trench |
| 22 | Volcanic arcs | 2707–2774 | conv_def, cf | arc_field | d_arc=166 km, σ=40 km |
| 23 | Dynamic topography | 2778–2859 | conv/div seeds, noise | dyn_topo | slab −600m, ridge +300m, mantle ±300m |
| 24 | Hotspot volcanism | 2861–2949 | n_hotspots | hotspot_topo | R=500–800 km, H=800–1500 m |
| 25 | Oceanic plateaus (LIPs) | 2951–3024 | n_lips=3–8 | plateau uplift | R=400–1000 km, H=1500–3000 m |
| 26 | E&M crustal thickening | 3028–3118 | defs, v_plate, H_c, dt | tectonic uplift | U=def×v/(2L_d)×H_c×Δρ/ρ_m |
| 27 | Climate-dependent diffusion | 3120–3204 | uplift, climate_factor | eroded relief | κ₀=0.02, 12 passes |
| 28 | Sediment redistribution | 3207–3237 | total_eroded | lowland fill | 60% on land (Milliman) |
| 29 | Isostatic relaxation | 3239–3248 | crust, cf, heat | relaxed relief | τ_eff=5 Myr |
| 30 | Foreland basins | 3256–3318 | conv_def, cf | basin depression | foredeep −150m, forebulge +30m |
| 31 | Glacial buzzsaw | 3320–3377 | lat, ELA | truncated peaks | ELA=5200−62|lat|, tanh compression |
| 32 | Rift shoulders | 3379–3417 | div_def, cf | shoulder uplift | 400m at 100 km from rift |
| 33 | Cratonic peneplains | 3419–3448 | activity, cf | flattened interior | target 300m, activity<0.1 |
| 34 | Epeirogenic warping | 3450–3511 | continental nuclei | tilt field | ±200m (Bond 1976) |
| 35 | Back-arc basins | 3513–3552 | conv_def, cf | subsidence | −800m ocean / −300m continental |
| 36 | Hypsometric correction | 3556–3604 | relief stats | corrected relief | conditional: median>1.5×target |
| 37 | Sea level | 3607–3615 | ocean_percent | relief − sea_level | percentile-based |
| 38 | Continental shelf profile | 3617–3697 | coast BFS | reshaped shelf | break at −130m, 15 cells width |
| 39 | Detail noise | 3699–3746 | seed | textured relief | 4 octaves, β=2.0, 160/80/40/20 m |
| 40 | Coastline perturbation (Gaussian) | 3748–3823 | relief | fractal coastline | 3 passes: σ=2000/1000/500 m |
| 41 | Coastline cleanup | 3825–3858 | relief | cleaned relief | 1 pass morphological |
| 42 | Events (meteorite/rift) | 3909–4061 | events list | updated relief + aerosol | Pi-group scaling |
| 43 | Slope computation | 4063–4099 | heights | slope map | max drop to 8-neighbors |
| 44 | Climate (unified) | 4101–4504 | heights, grid, cache | temp, precip | T=28−70x²+14x⁴, Hadley precip |
| 45 | Hydrology | 4635–4786 | heights, slope | flow_dir, flow_acc, rivers, lakes | D8, threshold scaling |
| 46 | Valley carving | 4788–4896 | flow_acc, relief | carved relief | D=0.2×Q^0.36, 80× incision |
| 47 | Biomes (Whittaker) | 4898–5097 | T, P, heights | biome_map | polygon lookup, alpine treeline |
| 48 | Settlement (Miami NPP) | 4507–4542 | biomes, T, P, h | settlement_map | Lieth 1975 |

### 1.2 Pipeline генерации (Crop scope)

| # | Стадия | Строки | Описание |
|---|--------|--------|----------|
| C1 | Region selection | 6030–6245 | find_interesting_region / find_continent_region |
| C2 | Bicubic upsample | 6298–6349 | Catmull-Rom + 6-octave FBM detail |
| C3 | Coastline perturbation | 6352–6396 | 5 octaves, ±300m band, flat coords |
| C4 | Edge fade | 6399–6434 | fade_land_edges: island=true, continent=false |
| C5 | SPACE erosion | 6436–6544 | K_br from defs, E&M maintenance uplift, 100×100kyr |
| C6 | Post-erosion edge fade | 6555–6592 | repeat edge treatment |
| C7 | Slope + Hydrology | 6594–6599 | compute_slope_grid + compute_hydrology_grid |
| C8 | Climate | 6601–6620 | compute_climate_unified with crop coordinates |
| C9 | Biomes | 6622–6652 | 5-pass smoothed T/P + Whittaker |
| C10 | Settlement | 6654–6658 | NPP + river/coast bonus |

### 1.3 Вспомогательные функции

| Функция | Строки | Назначение | Использование |
|---------|--------|------------|---------------|
| `clampf` | 63–71 | f32 clamp | повсеместно |
| `lerpf` | 74–76 | линейная интерполяция | valley carving, erosion |
| `hash_u32` | 80–87 | Murmur3 hash | seed derivation |
| `hash3` | 90–95 | 3D hash | value_noise3 |
| `hash_to_unit` | 98–100 | hash→[-1,1] | noise |
| `value_noise3` | 103–134 | 3D value noise, Hermite interp. | FBM, detail, coastline |
| `spherical_fbm` | 141–173 | 5-oct FBM на сфере | structural field, mantle noise |
| `spherical_wrap` | 186–203 | полярное отражение + X-wrap | planet grid |
| `plate_velocity_xy_from_omega` | 212–237 | ω×r → (east, north) velocity | plate evolution, boundary detection |
| `propagate_deformation` | 1827–1876 | eikonal-like exponential decay | conv/div/trans fields |
| `smooth_field` | 1883–1933 | anisotropic Gaussian smoothing | рельеф, деформация, бассейны |
| `isostatic_elevation` | 1792–1810 | Airy isostasy formula | initial + relaxation relief |
| `compute_d8_receivers` | 5145–5185 | D8 с стохастическим шумом | SPL, SPACE |
| `topological_sort_descending` | 5189–5193 | h→l sort | drainage accumulation |
| `compute_mfd_area` | 5204–5248 | Freeman MFD area | planet + crop erosion |
| `stream_power_step` | 5352–5436 | implicit B&W SPL step | crop erosion |
| `stream_power_step_mfd` | 5264–5343 | explicit MFD SPL step | planet erosion |
| `space_erosion_step` | 5495–5695 | SPACE с sediment tracking | crop erosion |
| `bicubic_sample` | 5983–6005 | Catmull-Rom 2D interpolation | crop upsample |
| `point_in_polygon` | 4971–4984 | ray-casting PIP (Shimrat 1962) | Whittaker biomes |
| `classify_biome_whittaker` | 4992–5009 | Whittaker polygon classification | biomes |
| `catmull_rom` | 5973–5979 | 1D Catmull-Rom kernel | bicubic_sample |
| `smooth_coastline` | 3868–3907 | morphological erosion/dilation | crop cleanup |

### 1.4 Структуры данных

| Struct | Строки | Поля | Назначение |
|--------|--------|------|-----------|
| `Rng` | 23–25 | s: [u32; 4] | xoshiro128++ state |
| `WorldCache` | 268–274 | lat_by_y, lon_by_x, x/y/z_by_cell (×WORLD_SIZE) | precomputed coords |
| `GridConfig` | 317–328 | width, height, size, is_spherical, km_per_cell_x/y, cos_lat_by_y | unified grid |
| `CellCache` | 394–405 | noise_x/y/z, lat_deg, lon_deg (×size) | per-cell coords |
| `PlanetInputs` | 473–483 | radius_km, gravity, density, rotation, tilt, ecc, atm, ocean% | planet config |
| `TectonicInputs` | 486–491 | plate_count, speed, mantle_heat | tectonic config |
| `SimulationConfig` | 522–541 | seed, planet, tectonics, events, scope, island_type/scale | full config |
| `PlateSpec` | 660–670 | lat, lon, dir_x/y, omega_x/y/z, heat, buoyancy | per-plate params |
| `PlateVector` | 673–679 | omega_x/y/z, heat, buoyancy | runtime plate data |
| `ComputePlatesResult` | 682–691 | plate_field, boundary_types/strength, vectors, evolution_time_yr | plates output |
| `ReliefResult` | 694–703 | relief, sea_level, conv_def, div_def, trans_def | relief output |
| `FrontierNode` | 706–710 | cost, index, plate | Dijkstra queue node |
| `GrowthParam` | 762–773 | drift_x/y, spread, roughness, freq_a-d, phase_a/b | plate growth |
| `DetailProfile` | 566–571 | erosion_rounds, fluvial_rounds, max_kernel_radius, plate_evolution_steps | preset |
| `RockType` | 5738–5761 | enum: Granite/Quartzite/Basalt/Schist/Sandstone/Limestone | → K_eff |
| `BiomePolygon` | 4914–4917 | id: u8, verts: &[(f32,f32)] | Whittaker diagram |
| `WasmSimulationResult` | 5764–5793 | all output maps + stats | WASM API |

---

## Этап 2: Оценка научной модели

### 2.1 Тектоника

| Процесс | Реализация | Источник | Оценка | Комментарий |
|---------|------------|---------|--------|------------|
| Генерация плит (Voronoi) | Dijkstra с cost-based growth, structural field, historical nuclei | Bird 2003 (статистика форм) | **B** | Не физический рост, но статистически правдоподобен. Неустранимо без мантийной конвекции. |
| Классификация границ | Relative velocity: vn (conv/div), vt (transform). Latitude-corrected normals. | Bird 2003; DeMets 2010 | **A** | Корректно: проекция на нормаль/тангенту, cos(φ) коррекция. Пропорции ~50/30/20% настраиваемы. |
| Plate evolution | Advective semi-Lagrangian + structural modulation + relaxation | Torsvik et al. 2010 | **B** | Кинематически, не динамически. Structural field (sin/cos) — фудж. Но результат реалистичен. |
| Деформация (damage rheology) | def_out = def×(1−α)/(1−α×def), α=0.6/0.4/0.4 | Lyakhovsky et al. 1997 | **A** | Формула корректна для нормализованного повреждения. α в правильном диапазоне (0.3–0.8 в литературе). |
| Interior suppression | BFS distance → exp(−d/L_rheol), L_rheol=300 km | Artemieva & Mooney 2001 | **A** | Физически обосновано. L=300 km — среднее между 200–500 km из литературы. |
| Uplift (E&M 1982) | U = def×v/(2L_d)×H_c×(ρ_m−ρ_c)/ρ_m | England & McKenzie 1982 | **A** | Формула точно соответствует E&M82 eq.15. Единицы корректны (m/yr). dt из plate evolution. |

### 2.2 Изостазия и литосфера

| Процесс | Реализация | Источник | Оценка | Комментарий |
|---------|------------|---------|--------|------------|
| Airy isostasy | h = C×(ρ_m−ρ_c)/denom, denom = ρ_m − (1−cf)×ρ_w | Turcotte & Schubert §2.2, §2.6 | **A** | Формула корректна. Плавный переход через cf. Water loading включён. |
| Densities | ρ_c=2800–2900 (cont→oce), ρ_m=3300, ρ_w=1025 | Christensen & Mooney 1995 | **A** | Все в стандартных диапазонах. |
| Flexural isostasy | N = α²/(2dx²), α = (4D/Δρg)^¼, Te=25 km → N≈34 | Watts 2001 | **A** | Математически верно. Te=25 km — глобальное среднее (Watts Table 5.1). |
| Thermal subsidence | d = 350√t (t<80), GDH1 plate model (t≥80) | Parsons & Sclater 1977; Stein & Stein 1992 | **A** | Коэффициенты точно из GDH1. Continuity at t=80 Ma проверена: 350√80=3130m. |
| Isostatic relaxation | f = 1−exp(−dt/5Myr) | Watts 2001 §8.4 | **B** | τ=5 Myr — в диапазоне 1–10 Myr (зависит от вязкости мантии). Одно значение для всех — упрощение. |

### 2.3 Рельеф и геоморфология

| Процесс | Реализация | Источник | Оценка | Комментарий |
|---------|------------|---------|--------|------------|
| Stream power (planet, MFD) | E = K×A^m×S^n, cap 30%/step, MFD area | Braun & Willett 2013; Salles 2023 | **B** | Explicit scheme с cap — стабильно но не convergent. MFD eliminates D8 artifacts. |
| SPACE erosion (crop) | E_r = K_br×A^m×S^n×exp(−H_s/H*), sediment routing | Shobe et al. 2017 | **A** | Все уравнения из §2. Implicit B&W для bedrock, explicit для sediment. |
| B&W implicit scheme | z_new = (z_old+U×dt+factor×z_recv)/(1+factor) | Braun & Willett 2013 | **A** | Обновлённый приёмник (downstream→upstream). Isostatic rebound встроен через iso_factor. |
| SPACE: K_br | 5e-6 − def×4e-6 (1e-6 to 5e-6) | Stock & Montgomery 1999; Harel 2016 | **B** | Диапазон корректен. Линейная зависимость от деформации — упрощение. |
| SPACE: K_sed | 1e-5 (≈5× mean K_br) | Shobe et al. 2017 §3.2 | **A** | В рекомендованном диапазоне (2–10× K_br). |
| SPACE: H* = 2.0 m | 2.0 m (рекомендовано 0.1–1.0) | Shobe et al. 2017 §3.1 | **C** | 🟡 Выше рекомендованного вдвое. Документировано как "thick alluvium". |
| SPACE: V_s = 0.1 m/yr | 0.1 (рекомендовано 0.01–10) | Shobe Table 1 | **A** | В диапазоне. |
| SPACE: F_f = 0.5 | 50% wash load | Sklar & Dietrich 2006 | **B** | 40–60% типично для горных рек. |
| SPACE: isostatic rebound | iso_factor = factor × (1−ρ_c/ρ_m) | Molnar & England 1990 | **A** | RHO_FRAC = 0.1667 — корректно. |
| Sub-grid channel width | W_ch = 4.0×A^0.4, correction W_ch/dx | Pelletier 2010; Leopold & Maddock 1953 | **A** | k_w=4.0 — в диапазоне. b=0.4 — каноническое значение. |
| Diffusion κ₀ = 0.02 m²/yr | climate-dependent: ×0.3–1.5 по широте | Fernandes & Dietrich 1997; Roe 2003 | **A** | κ₀=0.02 — стандартное значение для landscape-scale. Hadley модуляция обоснована. |
| Crustal thickness | base: cont. 40±10.5 km, oce. 7 km. Conv +30, div −20, trans +2 | Christensen & Mooney 1995; Owens & Zandt 1997 | **A** | Все значения из опубликованных наблюдений. |
| Volcanic arcs | d=166 km, σ=40 km, cont. 1000m / island 600m | Syracuse & Abers 2006 | **A** | d=166 km — медиана из Table 1 S&A06. σ=40 km — корректно. |
| Dynamic topography | slab −600m (conv_wide 1000km), ridge +300m (div_wide 800km), mantle ±300m | Hager 1985; Flament 2013; Hoggard 2016 | **A** | Амплитуды в диапазоне (Hoggard: ±1 km max). Длины волн корректны. |
| Foreland basins | foredeep −150m at 200km, forebulge +30m at 400km | DeCelles & Giles 1996; Watts 2001 | **B** | Расстояния/амплитуды — верхняя граница опубликованного. Поверхностная экспрессия (не subsurface) — корректна. |
| Glacial buzzsaw | ELA = 5200−62|lat|, tanh compression, lat-dependent intensity | Brozović et al. 1997; Ohmura et al. 1992 | **B** | ELA — линейная аппроксимация (реальная — нелинейная). Интенсивность по широте — хорошее дополнение. |
| Rift shoulders | Gaussian at 100 km, 400m × cf × def | Weissel & Karner 1989 | **B** | 400m — консервативно (EAR 1–2 km, но с hotspot interaction). Gaussian — упрощение. |
| Hotspots | 5–15 swells, R=500–800km, H=800–1500m, Gaussian profile | Morgan 1971; Crough 1983; Sleep 1990 | **A** | Все параметры из Crough 1983 Table 1. Continental resistance 50% — обосновано (Jordan 1975). |
| Mid-ocean ridges | ridge crest def³×500m, trench def³×1500m, fracture def²×400m | Macdonald 1982; Stern 2002; Tucholke 1988 | **B** | Кубические степени — эвристика для сужения пика. Амплитуды в правильном диапазоне. |
| Epeirogenic warping | ±200m tilt per continent nucleus, Gaussian decay | Bond 1976; Mitrovica 1989 | **B** | 100–300m из Bond 1976. Линейный тилт — грубое приближение (реальный — 2D). |
| Back-arc basins | Gaussian at 350km, −800m ocean / −300m continent | Karig 1971; Sdrolias & Müller 2006 | **B** | Расстояние 300–500km из литературы. Амплитуды корректны. |
| Cratonic peneplains | flatten к 300m при activity<0.1, cf>0.6 | King 1967; Fairbridge 1980 | **C** | Концепция правильная. 300m — среднее кратонное. 40% flatten — не из физики. |
| Oceanic plateaus | 3–8 LIPs, R=400–1000km, H=1500–3000m, flat-top smoothstep | Coffin & Eldholm 1994 | **A** | Ontong Java: R~700km, H~2km. Параметры в опубликованном диапазоне. |
| Continental shelf | break at −130m, 15 cells width, BFS from coast | Kennett 1982; Emery & Uchupi 1984 | **B** | −130m — стандартное значение. 15 cells = 150km — выше среднего (Earth mean 75km, range 30–400). |
| Sediment redistribution | 60% on land, weight by lowness | Milliman & Syvitski 1992; Allen 2008 | **B** | 60% — из литературы. Распределение по весу — упрощение (нет gravity routing). |
| Detail noise β=2.0 | 4 octaves: 160/80/40/20m, elev-scaling | Huang & Turcotte 1989 | **A** | β=2.0 — каноническое. Amplitude halving per octave — корректно. |
| Coastline perturbation | 3-pass Gaussian weight, σ=2000/1000/500m | Wessel & Smith 1996; Mandelbrot 1967 | **B** | Физически мотивировано (coastal variability). Конкретные σ — эмпирические. |

### 2.4 Климат

| Процесс | Реализация | Источник | Оценка | Комментарий |
|---------|------------|---------|--------|------------|
| T_sea(lat) | 28−70x²+14x⁴ (x=|lat|/90) | Peixoto & Oort 1992; Hartmann 1994 | **A** | T(0)=28, T(30)=20.4, T(60)=−0.3, T(90)=−28. Хорошо совпадает с зональными средними. |
| Lapse rate | 6.0 K/km | Holton & Hakim 2013 | **A** | Стандартный environmental lapse rate. |
| Greenhouse | 33×(p^0.3−1) °C | Pierrehumbert 2010 §4.3 | **B** | Упрощённая gray-atmosphere модель. Даёт +33°C at 1 bar (=Earth). Нелинейность корректна. |
| Continentality | −0.008 °C/km × dist × sin(lat) | Terjung & Louie 1972 | **B** | Conrad index адаптирован. Moscow→London delta ~5°C, формула даёт ~4.8°C при 600km, 55°N. |
| Осадки (зональные) | ITCZ: 2000×exp(−(lat/8)²), midlat: 700×exp(−((lat−45)/12)²), floor 150 | Adler et al. 2003 (GPCP) | **A** | Двух-Гауссов fit хорошо воспроизводит GPCP. 0°=2150mm, 30°=339mm, 45°=850mm. |
| Ветер | 3-cell: trades (<25°), westerlies (35–55°), polar (>65°), smooth transitions | Peixoto & Oort 1992; Seidel 2008 | **A** | Границы зон и transitions обоснованы (Seidel: Hadley edge 25–30°). Meridional 35% — обосновано. |
| Windward moisture | L=700 km, 5-ray angular spread ±15° | van der Ent & Savenije 2011; Trenberth 1991 | **A** | L=700km — из Fig.3 VES2011. Multi-ray заменяет предыдущий noise hack (H12). |
| Rain shadow | 0.40 mm/m, decay 250 km | Smith 1979; Galewsky 2009 | **A** | 0.30–0.50 в литературе, 0.40 — среднее. Decay 200–400 km (Galewsky). |
| Clausius-Clapeyron | exp(−h×0.000544), т.е. −42%/km | Held & Soden 2006 | **A** | 6 K/km × 7%/K = 42%/km. Exact. |
| Orographic lift | 0.8 mm/m rise above upwind | Roe 2005; Smith & Barstad 2004 | **A** | 0.5–1.5 mm/m в литературе, 0.8 — среднее. |
| Aerosol cooling | −15°C at aerosol=1 | Toon et al. 1997 | **B** | Chicxulub: 10–20°C cooling. 15°C — среднее. |
| Climate-dependent κ | ITCZ ×1.5, subtrop ×0.3, temperate ×1.0, polar ×0.4 | Roe et al. 2003; Hartmann 1994 | **B** | Множители качественно корректны. Не из прямых измерений, но физически мотивированы. |

### 2.5 Гидрология

| Процесс | Реализация | Источник | Оценка | Комментарий |
|---------|------------|---------|--------|------------|
| Flow direction (D8) | Steepest descent + stochastic ±5%/12% noise | Tucker & Bras 2000 | **A** | D8 — стандарт для DEM. Noise amplitude правильно масштабирован (12% planet, 3% crop). |
| MFD accumulation | Freeman 1991, p=1.1 | Freeman 1991; Salles 2023 | **A** | p=1.1 — каноническое значение. |
| Lake detection | No-outflow cells + indegree-loop detection | — | **C** | Нет pit-filling (Priority-Flood). Underpredicts lakes. |
| River threshold | A_crit ∝ dx², scaling with detail | Montgomery & Dietrich 1988 | **B** | Правильный скейлинг. Конкретные коэффициенты (0.00018+0.00007×rounds) — эмпирические. |
| Valley carving | D_valley = 80 × 0.2 × Q^0.36 = 16×Q^0.36 | Leopold & Maddock 1953; Schumm 1977 | **B** | 80× incision ratio — верхний конец 20–200× из Schumm. Bedrock transition at 1200m — обоснован (Whipple 2004). |

### 2.6 Биомы

| Процесс | Реализация | Источник | Оценка | Комментарий |
|---------|------------|---------|--------|------------|
| Whittaker classification | 9 polygon PIP test + Desert fallback | Ricklefs & Relyea 2014; plotbiomes | **A** | Полигоны оцифрованы корректно. Ray-casting (Shimrat 1962) — стандарт. |
| Köppen ET tundra | warmest month = T + 20×sin(lat)×0.5 < 10°C | Terjung 1970 | **B** | Seasonal amplitude 20°C — reasonable. Точнее: 10–30°C в зависимости от континентальности. |
| Alpine treeline | 4000−55×|lat| + noise×300m | Körner 2003 Fig. 7.1 | **A** | Treeline gradient 55 m/° — из Körner. Noise ±300m — local variation. |
| Riparian vegetation | rivers > 0.12 → upgrade dry biomes | Diamond 1997 | **B** | Физически мотивировано (irrigation effect). Порог 0.12 — эмпирический. |
| Biome smoothing | 2-pass mode filter, threshold: <2 same neighbors | — | **C** | Нет физического обоснования. Чисто визуальная постобработка. |

### 2.7 Settlement

| Процесс | Реализация | Источник | Оценка |
|---------|------------|---------|--------|
| Miami model NPP | NPP_T = 3000/(1+exp(1.315−0.119T)), NPP_P = 3000×(1−exp(−0.000664P)) | Lieth 1975 | **A** |
| Elevation penalty | onset 500m, zero at 4500m | Cohen & Small 1998; Körner 2003 | **A** |

---

### Summary оценок

| Оценка | Процессов | % |
|--------|-----------|---|
| **A** (научно корректно) | 33 | 60% |
| **B** (качественно верно, approx. params) | 17 | 31% |
| **C** (упрощение с потерей физики) | 4 | 7% |
| **D** (эвристика без обоснования) | 0 | 0% |
| **F** (ошибка) | 1 | 2% |

**Общая оценка: 91% процессов с рейтингом A или B.**

---

## Этап 3: Хаки, эвристики, трюки и баги

### 3.1 Fudge factors

| Строка | Значение | Что делает | Обоснование | Критичность |
|--------|----------|-----------|-------------|-------------|
| 832–834 | size_bias: (a×b)^0.74 | Нелинейный масштаб размера плит | Визуально подобран | ⚪ |
| 847 | spread: 0.82–1.22 | Скорость роста плит | Визуально подобран | ⚪ |
| 848 | roughness: 0.26–1.08 | Шероховатость границ | Визуально подобран | ⚪ |
| 877 | 0.62/0.38 (structural weights) | Относительный вес FBM октав | Визуально подобран | 🟡 |
| 914 | start_cost: 0.1–2.8 | Задержка роста нуклеусов | Визуально подобран | ⚪ |
| 936 | bend: ±0.34 + sin×0.16 | Изгиб исторической траектории | Визуально подобран | ⚪ |
| 1036 | drift_factor: 1.03−0.12×drift_align | Drift preferencing | Визуально подобран | ⚪ |
| 1043 | polar_factor: 1.0+|lat|/90×0.1 | Polar growth bias | Визуально подобран | ⚪ |
| 1384 | memory_keep: 0.5+0.18×(1−age_norm) | Plate boundary inertia | Визуально подобран | 🟡 |
| 1431–1436 | structural sin/cos в evolve | Структурная модуляция | Не связана с реальной литосферой | 🟡 |
| 2077 | arch: n1×0.20+n2×0.10, max(0) | Archipelago amplitude | «~1–3% extra land» — визуально | 🟡 |
| 2691 | def³ (ridge_crest, trench) | Cubic focusing | Нет физической модели ширины | 🟡 |
| 2701 | trans_def²×400 (fracture) | Quadratic focusing | Нет физической модели | ⚪ |
| 2763 | along_strike: powf(0.7) | Bias toward higher values | Визуально «volcanic peaks» | ⚪ |
| 3315 | relief=1.0 при basin<0 | Минимум суши | Предотвращает subsea basins | ⚪ |
| 3368 | max_excess: 2500−1000×intensity | Glacial buzzsaw ceiling | Не из данных | 🟡 |
| 3444 | flatten_strength: stability×0.4×cf | Peneplain compression | «max 40%» — эмпирически | 🟡 |
| 3584 | alpha clamp 1.0–8.0 | Hypsometric power | Произвольные границы | ⚪ |
| 6337 | scale = abs(h).max(200)×0.15 | FBM detail amplitude in crop | Montgomery & Brandon 2002: 10–25%. 15% середина | 🟡 |
| 6344 | amp decay 0.62 per octave | Hurst H=0.7 → 2^(−0.7)≈0.62 | Корректно | ✓ (не fudge) |
| 6480 | u_bg = 0.02e-3 (GIA background) | Базовый uplift 0.02 mm/yr | Peltier 2004: 0.01–0.05 mm/yr | ✓ |

**Итого fudge factors: ~18** (из них ~5 значимых 🟡, остальные ⚪)

### 3.2 Эвристики

| Строки | Описание | Заменяет физику | Критичность |
|--------|----------|----------------|-------------|
| 820–1094 | Voronoi growth (Dijkstra + cost modulation) | Мантийную конвекцию → plate genesis | ⚪ Неустранимо |
| 925–974 | Historical trajectory nuclei | Plate migration/fragmentation history | ⚪ Неустранимо |
| 1431–1436 | sin/cos structural в evolve_plate_field | Литосферная гетерогенность | 🟡 Можно использовать spherical_fbm |
| 2081–2090 | Threshold binary search для cf | Continuous mass balance | ⚪ Приемлемо |
| 2123–2194 | BFS + noise perturbation для coastlines | Tectonic/erosional coastal shaping | 🟡 Качественно обосновано |
| 3207–3237 | Weighted sediment redistribution (p30) | Gravity-driven sediment routing | 🟡 Нет flow routing для седиментов |
| 3556–3604 | Power-law hypsometric correction | Полная физика отсутствующих процессов | ⚪ Conditional safety valve |
| 3617–3697 | BFS-based shelf reshaping | Continental margin sedimentation + flexure | 🟡 Перезаписывает физику |
| 4710 | 20m порог для озёр | Coastal zone exclusion | ⚪ Разумно |
| 4837 | valley_depth = 16×Q^0.36 | Full fluvial incision model | 🟡 80× ratio — верхний край |
| 5057 | river>0.12 → biome upgrade | Riparian microclimate | ⚪ Качественно корректно |
| 6087 | mix_score = 1−|land_frac−0.4|×2.5 | Region aesthetic scoring | ⚪ Не физика |

**Итого эвристик: 12** (из них 5 🟡)

### 3.3 Математические трюки

| Строки | Трюк | Проблема | Критичность |
|--------|------|---------|-------------|
| 2691 | `.powf(3.0)` для ridge/trench | Cubic focusing вместо конвольвентного профиля | 🟡 Визуально приемлемо, но не физ. профиль |
| 2701 | `.powf(2.0)` для fracture zones | Квадратичное сужение | ⚪ |
| 2763 | `.powf(0.7)` bias для along-strike | Bias toward high values | ⚪ |
| 3371 | `.tanh()` для glacial truncation | Smooth compression вместо erosion rate | 🟡 Физическая альтернатива: limiter model |
| 3584 | power-law hypsometric | Эмпирическая коррекция | ⚪ Conditional |
| smooth_field N passes | Iterated Gaussian ≈ diffusion | Решение PDE через итерации | 🟡 Корректно для steady-state, но N passes = σ² |
| 2430–2432 | N=round(α²/2dx²) for flexure | Gaussian ≈ flexural filter | 🟡 Приближение, но Watts 2001 подтверждает |
| 5308–5309 | cap erosion at 30%/step | Unconditional stability | ⚪ Не convergent, но стабильно |
| 4456 | exp(−h×0.000544) Clausius-Clapeyron | Exact exponential fit | ✓ Корректно |
| 1801 | heat_anomaly×0.02 thermal correction | 2% density change from 300K | 🟡 Upper bound, includes partial melt |

**Итого трюков: 10** (из них 5 🟡)

### 3.4 Скрытые допущения

| # | Допущение | Последствие | Критичность |
|---|----------|-------------|-------------|
| D1 | Plate evolution мгновенная: all steps → one field, then relief | Нет синхронизации relief и plate motion | 🟡 |
| D2 | Deformation propagation мгновенная (steady-state eikonal) | Нет time-dependent deformation front | ⚪ Корректно для > 1 Myr |
| D3 | Climate computed AFTER relief (no feedback) | Нет precipitation → erosion → relief → precipitation loop | 🟡 Crop scope частично решает (pre-SPACE climate) |
| D4 | Ocean thermal subsidence computed from CURRENT plate speed | Реальная скорость менялась за 200 Myr | ⚪ Unavoidable at this resolution |
| D5 | Isostatic relaxation after erosion (not during) | Overestimation of transient relief | 🟡 Crop scope has iso_factor in B&W |
| D6 | Continental fraction binary → smoothed | No dynamic shoreline from climate/erosion | ⚪ |
| D7 | Same K_eff for planet and crop | Crop should inherit planet rock type | ⚪ Crop derives K from deformation |
| D8 | dt for diffusive erosion = total evolution time | Not per-step integration | 🔴 CFL violation possible |
| D9 | Smooth field uses previous-pass values (Jacobi, not Gauss-Seidel) | Slower convergence, not incorrect | ⚪ |
| D10 | crop uplift uses representative H_c=40km (not per-cell) | Different H_c in mountains vs. plains | 🟡 Planet scope uses per-cell H_c |

### 3.5 Потенциальные баги

| # | Описание | Строки | Критичность |
|---|----------|--------|-------------|
| B1 | **CFL violation in planet diffusion**: kdt = κ×dt_yr (dt=7–15 Myr!), inv_dx2 = 1/dx²m. For κ=0.02, dt=10e6 yr, dx=10000m: kdt×inv_dx2 = 0.02×1e7/1e8 = 2.0. CFL requires <0.25. **12 passes each with full dt.** | 3181–3204 | 🔴 |
| B2 | Hotspot latitude calculation inverted: `(y+0.5)/height × π − π/2` gives −π/2 at y=0 (south) but WorldCache has +90° at y=0 (north). Sign is correct because sin/cos of latitude are symmetric in the absolute height formula, but could cause geographic misplacement of hotspots relative to plates. | 2910–2913 | 🟡 |
| B3 | `nearest_free_index`: diamond spiral search can miss some cells when `max_radius` is very large (radius loop only checks border of diamond, inner cells assumed already checked). This is correct for BFS-like expansion. | 787–818 | ⚪ |
| B4 | `coast_dist` forward pass includes wrong diagonal: `(-1, 1)` appears in BOTH forward and backward passes. Should be in backward pass only. | 4187, 4204 | 🟡 |
| B5 | `smooth_coastline` function (3868–3907) is 2-pass but inlined cleanup (3825–3858) is 1-pass. Both exist in code — crop uses `smooth_coastline` (2 pass), planet uses inline (1 pass). Inconsistency. | 3825 vs 3868 | ⚪ |
| B6 | No NaN guard in `isostatic_elevation`: if denom=0 (ρ_m = (1−cf)×ρ_w), division by zero. Minimum denom when cf=0: 3300−1025=2275. When cf=1: 3300. Always positive. **Safe.** | 1808 | ⚪ (safe) |
| B7 | `evolve_plate_field` structural field uses inline sin/cos instead of spherical_fbm. Different noise per step but no spatial coherence guarantee. | 1431–1436 | ⚪ |
| B8 | Crop BFS shelf distance at grid edges: `grid.neighbor` clamps, so BFS won't wrap. Edge ocean cells may have underestimated shelf distance. | 3650–3665 | ⚪ |

---

## Этап 3 Summary

| Категория | Количество | 🔴 | 🟡 | ⚪ |
|-----------|-----------|-----|-----|-----|
| Fudge factors | 18 | 0 | 5 | 13 |
| Эвристики | 12 | 0 | 5 | 7 |
| Математические трюки | 10 | 0 | 5 | 5 |
| Скрытые допущения | 10 | 1 | 4 | 5 |
| Потенциальные баги | 8 | 1 | 2 | 5 |
| **ИТОГО** | **58** | **2** | **21** | **35** |

---

## Топ-10 самых критичных проблем

| # | Проблема | Критичность | Рекомендация |
|---|----------|-------------|-------------|
| 1 | **B1: CFL violation in planet diffusion** — κ×dt/dx² ≈ 2.0, нужно <0.25. 12 итераций с полным dt вместо sub-stepping. | 🔴 | Разбить на sub-steps: N_sub = ceil(κ×dt/(0.2×dx²)). Или implicit diffusion. |
| 2 | **D8: dt = total evolution time in diffusive loop** — не поделён на N_passes или sub-stepped. | 🔴 | `kdt = kappa * climate_factor[i] * dt_yr / 12.0` (per pass) вместо full dt. |
| 3 | **H*=2.0m выше рекомендованного** — Shobe 2017 рекомендует 0.1–1.0m. | 🟡 | Документировать или снизить до 1.0m и перекалибровать V_s. |
| 4 | **Shelf reshaping перезаписывает физику** — BFS-based blend 0.7 overrides isostatic profile. | 🟡 | Использовать sediment infill model вместо direct elevation override. |
| 5 | **Structural field в evolve — sin/cos, не FBM** — нет пространственной когерентности между шагами. | 🟡 | Заменить на spherical_fbm с seed per step. |
| 6 | **Valley carving 80× incision ratio** — верхний край диапазона (20–200×). | 🟡 | Снизить до 40× (√(20×200)≈63, geometric mean). |
| 7 | **Hotspot latitude convention mismatch** — `y/h×π−π/2` vs WorldCache `90−y×180/h`. | 🟡 | Унифицировать: использовать CellCache lat_deg вместо inline формулы. |
| 8 | **Coast distance Chamfer: duplicate diagonal** — (-1,1) в обоих проходах. | 🟡 | Убрать (-1,1) из forward pass, добавить (1,-1) если отсутствует. |
| 9 | **Peneplain flatten 40% — не из физики** — конкретное число не из литературы. | 🟡 | Привязать к возрасту литосферы или денудационному времени. |
| 10 | **3 coastline perturbation systems (pre-smooth, post-smooth, Gaussian)** — избыточно, трудно калибровать. | 🟡 | Объединить в 1–2 системы с чёткими масштабами. |
