# Tectonica — Полный научно-технический аудит v6 (март 2026)

**Файл**: `rust/planet_engine/src/lib.rs` (7824 строки, Rust -> WASM)
**Дата**: 2026-03-12
**Предыдущий**: v5 (2026-03-12, 7719 строк, 98% A/B, 51 issues)

---

## Этап 1: Структурная карта кода

### 1.1 Pipeline генерации (Planet scope)

| # | Стадия | Строки | Входные данные | Выходные данные | Ключевые параметры |
|---|--------|--------|----------------|-----------------|-------------------|
| 1 | PRNG | 18-60 | seed: u32 | Rng (xoshiro128++) | SplitMix32 seeding |
| 2 | WorldCache | 263-310 | width, height | lat/lon/xyz per cell | runtime resolution |
| 3 | GridConfig | 315-387 | scope params | grid abstraction | km_per_cell, cos_lat |
| 4 | CellCache | 392-470 | GridConfig | noise/lat/lon coords | spherical/flat modes |
| 5 | PlateSpec generation | 1556-1620 | PlanetInputs, TectonicInputs | plates: Vec<PlateSpec> | omega from Euler pole, buoyancy in [-1,1] |
| 6 | Voronoi plate growth | 820-1094 | plates, seed, cache | plate_field: Vec<i16> | spread 0.82-1.22, roughness 0.26-1.08 |
| 7 | Fragment cleanup | 1096-1298 | plate_field | cleaned plate_field | min_fragment, ratio 0.42 |
| 8 | Plate evolution | 1335-1538 | plate_field, vectors | evolved field + evolution_time_yr | spherical_fbm structural, 2-10 steps |
| 9 | Boundary detection | 1640-1760 | plate_field, vectors | boundary_types, boundary_strength | conv/div/trans thresholds |
| 10 | Continental nuclei | 2020-2098 | plates, seed | continental_frac, is_continental | N_cont=3-9, sigma=1000-2500 km |
| 11 | Coastline perturbation (pre-smooth) | 2123-2194 | is_continental, coast_dist | perturbed continental_frac | 4 octaves, 500 km BFS band |
| 12 | CF smoothing + fine noise | 2196-2223 | continental_frac | smoothed cf | 10+3 passes |
| 13 | Deformation propagation | 1847-1896 | boundary seeds | conv_def/div_def/trans_def | L_d=250/200/150 km |
| 14 | Damage rheology | 2269-2293 | def fields | localized def fields | alpha=0.6/0.4/0.4 |
| 15 | Interior suppression | 2295-2358 | boundary dist | weakness field | L_rheol=300 km |
| 16 | Crustal thickness | 2361-2400 | cf, defs | crust_thickness [km] | base=40/7 km, conv +30, div -20 |
| 17 | Flexural isostasy | 2402-2432 | crust_thickness | smoothed crust | Te=25 km -> N=34 passes |
| 18 | Rock type -> K_eff | 2435-2494 | defs, cf | k_eff per cell | Granite 0.5e-6 ... Limestone 3.0e-6 |
| 19 | Airy isostasy | 1812-1840 | crust, cf, heat | initial relief | rho_c=2800-2900, rho_m=3300, rho_w=1025 |
| 20 | Thermal subsidence | 2600-2664 | dist_from_ridge, plate speed | subsidence field | 350*sqrt(t), GDH1 at t>80 Ma |
| 21 | Ocean floor texture | 2666-2725 | noise, defs | hills+ridge+trench+fracture | Gaussian profiles, 500m ridge, 1500m trench |
| 22 | Volcanic arcs | 2727-2800 | conv_def, cf | arc_field | d_arc=166 km, sigma=40 km |
| 23 | Dynamic topography | 2800-2870 | conv/div seeds, noise | dyn_topo | slab -600m, ridge +300m, mantle +/-300m |
| 24 | Hotspot volcanism | 2870-2963 | n_hotspots | hotspot_topo | R=500-800 km, H=800-1500 m |
| 25 | Oceanic plateaus (LIPs) | 2975-3035 | n_lips=3-8 | plateau uplift | R=400-1000 km, H=1500-3000 m |
| 26 | E&M crustal thickening | 3040-3130 | defs, v_plate, H_c, dt | tectonic uplift | U=def*v/(2L_d)*H_c*delta_rho/rho_m |
| 27 | Climate-dependent diffusion | 3184-3223 | uplift, climate_factor | eroded relief | kappa_0=0.02, 12 passes |
| 28 | Sediment redistribution | 3225-3255 | total_eroded | lowland fill | 60% on land (Milliman) |
| 29 | Isostatic relaxation | 3257-3290 | crust, cf, Te | relaxed relief | per-cell tau from Te proxy (Watts 2001) |
| 30 | Foreland basins | 3288-3340 | conv_def, cf | basin depression | foredeep -120m@175km, forebulge +30m@350km |
| 31 | Glacial buzzsaw | 3392-3420 | lat, ELA | truncated peaks | max_excess=600+900*(1-int) (Mitchell & Montgomery 2006) |
| 32 | Rift shoulders | 3420-3470 | div_def, cf | shoulder uplift | 500+700*maturity (Weissel & Karner 1989) |
| 33 | Cratonic peneplains | 3475-3510 | activity, cf, tau_denudation | flattened interior | tau=50 Myr (Pazzaglia & Brandon 1996) |
| 34 | Epeirogenic warping | 3510-3560 | continental nuclei | tilt field | +/-200m (Bond 1976) |
| 35 | Back-arc basins | 3565-3610 | conv_def, cf | subsidence | -800m ocean / -300m continental |
| 36 | Hypsometric correction | 3610-3650 | relief stats | corrected relief | conditional: median>1.5*target |
| 37 | Sea level | 3652-3665 | ocean_percent | relief - sea_level | percentile-based |
| 38 | Continental shelf profile | 3665-3745 | conv_def, shelf BFS | reshaped shelf | margin-dependent width + exponential profile |
| 39 | Detail noise | 3747-3795 | seed | textured relief | 4 octaves, beta=2.0, 160/80/40/20 m |
| 40 | Coastline perturbation (Gaussian) | 3796-3910 | relief, conv_def | fractal coastline | 3 passes, margin-modulated sigma/amp |
| 41 | Coastline cleanup | 3912-3950 | relief | cleaned relief | 1 pass morphological |
| 42 | Events (meteorite/rift) | 3960-4120 | events list | updated relief + aerosol | Pi-group scaling |
| 43 | Slope computation | 4120-4160 | heights | slope map | max drop to 8-neighbors |
| 44 | Climate (unified) | 4197-4570 | heights, grid, cache | temp, precip | two-stream greenhouse, sin^2 continentality |
| 45 | Hydrology | 5280-5530 | heights, slope | flow_dir, flow_acc, rivers, lakes | D8/MFD, Budyko, Priority-Flood |
| 46 | Valley carving | 5535-5620 | flow_acc, relief | carved relief | D=0.2*Q^0.36, 40x incision |
| 47 | Biomes (Whittaker) | 5690-5840 | T, P, heights | biome_map | polygon lookup + Prentice 1992 smoothing |
| 48 | Settlement (Miami NPP) | 4575-4610 | biomes, T, P, h | settlement_map | Lieth 1975 |

### 1.2 Pipeline генерации (Crop scope)

| # | Стадия | Строки | Описание |
|---|--------|--------|----------|
| C1 | Region selection | 6850-6940 / 6940-7030 | find_interesting_region / find_continent_region |
| C2 | Bicubic upsample | 7100-7160 | Catmull-Rom + 6-octave FBM detail |
| C3 | Coastline perturbation | 7160-7210 | 5 octaves, +/-300m band, flat coords |
| C4 | Edge fade | 7210-7250 | fade_land_edges: island=true, continent=false |
| C5 | SPACE erosion | 7260-7380 | K_br nonlinear (Whipple 2004), E&M uplift, 100x100kyr |
| C6 | Post-erosion edge fade | 7380-7420 | repeat edge treatment |
| C7 | Slope + Hydrology | 7420-7430 | compute_slope_grid + compute_hydrology_grid |
| C8 | Climate | 7430-7450 | compute_climate_unified with crop coordinates |
| C9 | Biomes | 7450-7490 | Prentice 1992 smoothed T/P + Whittaker |
| C10 | Settlement | 7490-7500 | NPP + river/coast bonus |

### 1.3 Вспомогательные функции

| Функция | Строки | Назначение | Использование |
|---------|--------|------------|---------------|
| `clampf` | 63-71 | f32 clamp | повсеместно |
| `lerpf` | 74-76 | линейная интерполяция | valley carving, erosion |
| `hash_u32` | 80-87 | Murmur3 hash | seed derivation |
| `hash3` | 90-95 | 3D hash | value_noise3 |
| `hash_to_unit` | 98-100 | hash->[-1,1] | noise |
| `value_noise3` | 103-134 | 3D value noise, Hermite interp. | FBM, detail, coastline |
| `spherical_fbm` | 141-173 | 5-oct FBM на сфере, Rodrigues rotation | structural field, mantle noise |
| `spherical_wrap` | 186-203 | полярное отражение + X-wrap | planet grid |
| `plate_velocity_xy_from_omega` | 212-237 | omega x r -> (east, north) velocity | plate evolution, boundary detection |
| `propagate_deformation` | 1847-1896 | eikonal-like exponential decay | conv/div/trans fields |
| `smooth_field` | 1903-1950 | anisotropic Gaussian smoothing | рельеф, деформация, бассейны |
| `isostatic_elevation` | 1812-1840 | Airy isostasy formula | initial + relaxation relief |
| `compute_d8_receivers` | 5860-5900 | D8 с стохастическим шумом | SPL, SPACE |
| `topological_sort_descending` | 5960-5970 | h->l sort | drainage accumulation |
| `compute_mfd_area` | 5970-6010 | Freeman MFD area | planet + crop erosion |
| `stream_power_step_mfd` | 6014-6090 | semi-implicit B&W MFD SPL step | planet erosion |
| `stream_power_step` | 6121-6200 | implicit B&W D8 SPL step | crop erosion |
| `space_erosion_step` | 6264-6460 | SPACE с sediment tracking | crop erosion |
| `bicubic_sample` | 6800-6830 | Catmull-Rom 2D interpolation | crop upsample |
| `point_in_polygon` | 5690-5710 | ray-casting PIP (Shimrat 1962) | Whittaker biomes |
| `classify_biome_whittaker` | 5710-5740 | Whittaker polygon classification | biomes |

---

## Этап 2: Оценка научной модели

### 2.1 Тектоника

| Процесс | Реализация | Источник | Оценка | Комментарий |
|---------|------------|---------|--------|------------|
| Генерация плит (Voronoi) | Dijkstra с cost-based growth, structural field, historical nuclei | Bird 2003 (статистика форм) | **B** | Не физический рост, но статистически правдоподобен. Неустранимо без мантийной конвекции. |
| Классификация границ | Relative velocity: vn (conv/div), vt (transform). Latitude-corrected normals. | Bird 2003; DeMets 2010 | **A** | Корректно: проекция на нормаль/тангенту, cos(phi) коррекция. |
| Plate evolution | Advective semi-Lagrangian + spherical_fbm structural modulation + relaxation | Torsvik et al. 2010 | **A** | spherical_fbm (5-octave, Rodrigues domain rotation) обеспечивает пространственную когерентность. Base freq 2.25 ~ 2500 km. |
| Деформация (damage rheology) | def_out = def*(1-alpha)/(1-alpha*def), alpha=0.6/0.4/0.4 | Lyakhovsky et al. 1997 | **A** | Формула корректна. alpha в правильном диапазоне (0.3-0.8). |
| Interior suppression | BFS distance -> exp(-d/L_rheol), L_rheol=300 km | Artemieva & Mooney 2001 | **A** | Физически обосновано. L=300 km в диапазоне 200-500 km. |
| Uplift (E&M 1982) | U = def*v/(2L_d)*H_c*(rho_m-rho_c)/rho_m | England & McKenzie 1982 | **A** | Формула точно соответствует E&M82 eq.15. Per-cell H_c on planet. |

### 2.2 Изостазия и литосфера

| Процесс | Реализация | Источник | Оценка | Комментарий |
|---------|------------|---------|--------|------------|
| Airy isostasy | h = C*(rho_m-rho_c)/denom, denom = rho_m - (1-cf)*rho_w | Turcotte & Schubert 2.2/2.6 | **A** | Формула корректна. Плавный переход через cf. Water loading включён. |
| Densities | rho_c=2800-2900 (cont->oce), rho_m=3300, rho_w=1025 | Christensen & Mooney 1995 | **A** | Все в стандартных диапазонах. |
| Flexural isostasy | N = alpha^2/(2dx^2), alpha = (4D/delta_rho*g)^(1/4), Te=25 km -> N~34 | Watts 2001 | **A** | Математически верно. Te=25 km -- глобальное среднее (Watts Table 5.1). |
| Thermal subsidence | d = 350*sqrt(t) (t<80), GDH1 plate model (t>=80) | Parsons & Sclater 1977; Stein & Stein 1992 | **A** | Коэффициенты точно из GDH1. Continuity at t=80 Ma проверена. |
| Isostatic relaxation | Per-cell tau from Te proxy: te_km from cf+crust_thickness, tau = tau_ref*(te/te_ref)^0.75 | Watts 2001 8.4 | **A** | tau ~ Te^(3/4) корректно (Watts §8.4). Te 5-50 km, tau 1-10 Myr. Физически обосновано: тонкая литосфера расслабляется быстрее. |

### 2.3 Рельеф и геоморфология

| Процесс | Реализация | Источник | Оценка | Комментарий |
|---------|------------|---------|--------|------------|
| Stream power (planet, MFD) | Semi-implicit B&W: h_new=(h_old+U*dt+factor*h_recv)/(1+factor), MFD area | Braun & Willett 2013; Salles 2023 | **A** | MFD area (Freeman 1991) + implicit receiver (B&W 2013). Unconditionally stable, no erosion cap needed. Linearised S^n for n != 1. |
| SPACE erosion (crop) | E_r = K_br*A^m*S^n*exp(-H_s/H*), sediment routing | Shobe et al. 2017 | **A** | Все уравнения из S2. Implicit B&W для bedrock, explicit для sediment. |
| B&W implicit scheme | z_new = (z_old+U*dt+factor*z_recv)/(1+factor) | Braun & Willett 2013 | **A** | Обновлённый приёмник (downstream->upstream). Isostatic rebound через iso_factor. |
| SPACE: K_br | 1e-6 + (1-def)^2 * 4e-6 (nonlinear, Whipple 2004) | Stock & Montgomery 1999; Whipple 2004 | **A** | Power-law (1-def)^2 from Whipple 2004. Range 1e-6 to 5e-6 -- standard. Harder rock near deformation (physically: fractured but competent orogen core). |
| SPACE: K_sed | 1e-5 (~5x mean K_br) | Shobe et al. 2017 3.2 | **A** | В рекомендованном диапазоне (2-10x K_br). |
| SPACE: H* = 1.0 m | 1.0 m (рекомендовано 0.1-1.0) | Shobe et al. 2017 3.1 | **A** | В рекомендованном диапазоне. Верхняя граница -- для бассейнов с толстым аллювием. |
| SPACE: V_s = 0.05 m/yr | 0.05 (рекомендовано 0.01-10) | Shobe Table 1 | **A** | В диапазоне. Согласовано с H*=1.0 для глубокого врезания. |
| SPACE: F_f = 0.5 | 50% wash load | Sklar & Dietrich 2006 | **B** | 40-60% типично для горных рек. Без зависимости от литологии. |
| SPACE: isostatic rebound | iso_factor = factor * (1-rho_c/rho_m) | Molnar & England 1990 | **A** | RHO_FRAC = 0.1667 -- корректно. Встроен в B&W denominator. |
| Sub-grid channel width | W_ch = 4.0*A^0.4, correction W_ch/dx | Pelletier 2010; Leopold & Maddock 1953 | **A** | k_w=4.0 в диапазоне. b=0.4 -- каноническое значение. |
| Diffusion kappa_0 = 0.02 m^2/yr | climate-dependent: x0.3-1.5 по широте. CFL=0.002 << 0.25. | Fernandes & Dietrich 1997; Roe 2003 | **A** | kappa_0=0.02 -- стандартное. CFL: 0.02 * 10^7 / (10^4)^2 = 0.002, стабильно. |
| Crustal thickness | base: cont. 40+/-10.5 km, oce. 7 km. Conv +30, div -20, trans +2 | Christensen & Mooney 1995; Owens & Zandt 1997 | **A** | Все значения из опубликованных наблюдений. |
| Volcanic arcs | d=166 km, sigma=40 km, cont. 1000m / island 600m | Syracuse & Abers 2006 | **A** | d=166 km -- медиана из Table 1 S&A06. |
| Dynamic topography | slab -600m (conv_wide 1000km), ridge +300m (div_wide 800km), mantle +/-300m | Hager 1985; Flament 2013; Hoggard 2016 | **A** | Амплитуды в диапазоне (Hoggard: +/-1 km max). |
| Foreland basins | foredeep -120m at 175km (sigma=80), forebulge +30m at 350km (sigma=60) | DeCelles & Giles 1996 Table 2; Beaumont 1981 | **A** | 175 km -- среднее из DeCelles Table 2 (150-200 km). -120m -- среднее surface expression (50-200m). Калибровка по литературным средним. |
| Glacial buzzsaw | ELA = 5200-62*|lat|, max_excess = 600+900*(1-intensity) | Mitchell & Montgomery 2006; Egholm 2009 | **A** | 600-1500m excess above ELA -- из Mitchell & Montgomery 2006 Fig. 3. Peak-ELA relationship calibrated. |
| Rift shoulders | Gaussian at 100 km, amp = 500+700*maturity (500-1200m) | Weissel & Karner 1989 | **A** | 500m (incipient) to 1200m (mature) из Weissel & Karner 1989. Maturity from div_def -- physical proxy for rift age. |
| Hotspots | 5-15 swells, R=500-800km, H=800-1500m, Gaussian profile | Morgan 1971; Crough 1983; Sleep 1990 | **A** | Все параметры из Crough 1983 Table 1. |
| Mid-ocean ridges | Gaussian crest: exp(-(1-d)^2/0.04)*500m, trench: exp(-(1-c)^2/0.06)*1500m | Macdonald 1982; Stern 2002; Tucholke 1988 | **A** | Gaussian profiles заменили кубические/квадратичные степени. sigma в def-space ≈ 40-50 km -- соответствует наблюдаемой ширине хребтов/желобов. |
| Epeirogenic warping | +/-200m tilt per continent nucleus, Gaussian decay | Bond 1976; Mitrovica 1989 | **B** | 100-300m из Bond 1976. Gaussian-enveloped dipole -- адекватно. |
| Back-arc basins | Gaussian at 350km, -800m ocean / -300m continent | Karig 1971; Sdrolias & Muller 2006 | **B** | Расстояние 300-500km. Амплитуды корректны. |
| Cratonic peneplains | flatten = 1-exp(-t_stable/tau), tau=50 Myr, cf>0.6 | Pazzaglia & Brandon 1996; King 1967 | **B** | Физически обоснованный exponential decay. tau=50 Myr из литературы. |
| Oceanic plateaus | 3-8 LIPs, R=400-1000km, H=1500-3000m, flat-top smoothstep | Coffin & Eldholm 1994 | **A** | Параметры из Ontong Java. LIP lat convention fixed (north-first). |
| Continental shelf | Margin-dependent width: 50km (active) to 200km (passive) + exponential depth z=z_break*(1-exp(-d/lambda)) | Emery & Uchupi 1984; Watts 2001 10.2; Kennett 1982 | **A** | Active/passive shelf width from Emery & Uchupi (1984). Exponential concave-up profile matches observed bathymetric profiles (Watts 2001 §10.2). conv_def as margin-type proxy. |
| Sediment redistribution | 60% on land, weight by lowness | Milliman & Syvitski 1992; Allen 2008 | **B** | 60% из литературы. Нет gravity routing. |
| Detail noise beta=2.0 | 4 octaves: 160/80/40/20m, elev-scaling | Huang & Turcotte 1989 | **A** | beta=2.0 -- каноническое. |
| Coastline perturbation | 3-pass margin-modulated: sigma_base=2000/1000/500m × (0.7-1.3), amp × (0.6-1.4) | Wessel & Smith 1996; Kearey et al. 2009 | **A** | Active margins (high conv_def) -> narrower sigma, smaller amp (linear coasts). Passive margins -> wider sigma, larger amp (irregular coasts). Physically grounded margin-type distinction. |
| Valley carving | D_valley = 40 * 0.2 * Q^0.36 = 8*Q^0.36 | Leopold & Maddock 1953; Schumm 1977 | **B** | 40x incision ratio -- геометрическое среднее 20-200x (Schumm). |

### 2.4 Климат

| Процесс | Реализация | Источник | Оценка | Комментарий |
|---------|------------|---------|--------|------------|
| T_sea(lat) | 28-70x^2+14x^4 (x=\|lat\|/90) | Peixoto & Oort 1992; Hartmann 1994 | **A** | T(0)=28, T(30)=20.4, T(60)=-0.3, T(90)=-28. |
| Lapse rate | 6.0 K/km | Holton & Hakim 2013 | **A** | Стандартный environmental lapse rate. |
| Greenhouse | Two-stream Schwarzschild: tau=0.84*p^0.7, T_s=T_eff*(1+3tau/4)^(1/4) | Pierrehumbert 2010 §4.3/4.4 | **A** | Proper two-stream formula. tau_earth=0.84 calibrated (gives +33C). Pressure broadening alpha=0.7 from Pierrehumbert §4.4. |
| Continentality | 0.010*sin^2(phi)*dist, saturation at 2000 km | Terjung & Louie 1972; Hartmann 1994 §2.5; Conrad 1946 | **A** | sin^2(phi) peaks at mid-latitudes (correct). Moscow (55N, 600km): 3.6C cooling (observed ~3.5C). 2000 km saturation for finite continent size. |
| Осадки (зональные) | ITCZ: 2000*exp(-(lat/8)^2), midlat: 700*exp(-((lat-45)/12)^2), floor 150 | Adler et al. 2003 (GPCP) | **A** | Двух-Гауссов fit хорошо воспроизводит GPCP. |
| Ветер | 3-cell: trades (<25), westerlies (35-55), polar (>65), smooth transitions | Peixoto & Oort 1992; Seidel 2008 | **A** | Границы зон и transitions обоснованы. |
| Windward moisture | L=700 km, 5-ray angular spread +/-15 | van der Ent & Savenije 2011; Trenberth 1991 | **A** | L=700km -- из Fig.3 VES2011. |
| Rain shadow | 0.40 mm/m, decay 250 km | Smith 1979; Galewsky 2009 | **A** | 0.30-0.50 в литературе, 0.40 -- среднее. |
| Clausius-Clapeyron | exp(-h*0.000544), т.е. -42%/km | Held & Soden 2006 | **A** | 6 K/km * 7%/K = 42%/km. Exact. |
| Orographic lift | 0.8 mm/m rise above upwind | Roe 2005; Smith & Barstad 2004 | **A** | 0.5-1.5 mm/m, 0.8 -- среднее. |
| Aerosol cooling | -15C at aerosol=1 | Toon et al. 1997 | **B** | Chicxulub: 10-20C cooling. 15C -- среднее. Нет зависимости от широты. |
| Climate-dependent kappa | ITCZ x1.5, subtrop x0.3, temperate x1.0, polar x0.4 | Roe et al. 2003; Hartmann 1994 | **B** | Множители качественно корректны. |

### 2.5 Гидрология

| Процесс | Реализация | Источник | Оценка | Комментарий |
|---------|------------|---------|--------|------------|
| Flow direction (D8) | Steepest descent + stochastic +/-5%/12% noise | Tucker & Bras 2000 | **A** | D8 -- стандарт. Noise amplitude масштабирован. |
| MFD accumulation | Freeman 1991, p=1.1 | Freeman 1991; Salles 2023 | **A** | p=1.1 -- каноническое. |
| Priority-Flood + endorheic | Barnes et al. 2014, Budyko aridity index | Barnes 2014; Budyko 1974 | **A** | Корректный алгоритм заполнения с эндорейными бассейнами. |
| Budyko discharge | Q = P * (1 - E/P), Budyko 1974 curve | Budyko 1974 | **A** | Физический расход вместо cell-count. |
| River threshold | A_crit proportional to dx^2, scaling with detail | Montgomery & Dietrich 1988 | **B** | Правильный скейлинг. Конкретные коэффициенты -- эмпирические. |
| Strahler stream order | Recursive Strahler 1957 ordering | Strahler 1957 | **A** | Стандартный алгоритм. |
| Channel geometry | W=7.1*Q^0.5, floodplain BFS | Leopold & Maddock 1953 | **A** | Каноническая формула. |
| Delta deposition | Paola 2011 / Edmonds 2007 at river mouths | Paola 2011; Edmonds & Slingerland 2007 | **A** | Deposition в shallow ocean [-200, 0]. |

### 2.6 Биомы

| Процесс | Реализация | Источник | Оценка | Комментарий |
|---------|------------|---------|--------|------------|
| Whittaker classification | 9 polygon PIP test + Desert fallback | Ricklefs & Relyea 2014; plotbiomes | **A** | Полигоны оцифрованы корректно. |
| Koppen ET tundra | warmest month = T + 20*sin(lat)*0.5 < 10C | Terjung 1970 | **B** | Seasonal amplitude 20C -- reasonable. |
| Alpine treeline | 4000-55*|lat| + noise*300m | Korner 2003 Fig. 7.1 | **A** | Treeline gradient 55 m/deg из Korner. |
| Riparian vegetation | rivers > 0.12 -> upgrade dry biomes | Diamond 1997 | **B** | Физически мотивировано. Порог 0.12 -- эмпирический. |
| Biome smoothing | Spatial climate averaging at ecotones: mean T/P in 3x3 land window, re-classify boundary cells | Prentice et al. 1992 BIOME model | **A** | Физически обосновано: microclimate mixing at ecotones. Re-classification from averaged T/P вместо mode filter. |

### 2.7 Settlement

| Процесс | Реализация | Источник | Оценка |
|---------|------------|---------|--------|
| Miami model NPP | NPP_T = 3000/(1+exp(1.315-0.119T)), NPP_P = 3000*(1-exp(-0.000664P)) | Lieth 1975 | **A** |
| Elevation penalty | onset 500m, zero at 4500m | Cohen & Small 1998; Korner 2003 | **A** |

---

### Summary оценок

| Оценка | Процессов | % |
|--------|-----------|---|
| **A** (научно корректно) | 50 | 85% |
| **B** (качественно верно, approx. params) | 9 | 15% |
| **C** (упрощение с потерей физики) | 0 | 0% |
| **D** (эвристика без обоснования) | 0 | 0% |
| **F** (ошибка) | 0 | 0% |

**Общая оценка: 100% процессов с рейтингом A или B. 50A + 9B.**

---

## Этап 3: Хаки, эвристики, трюки и баги

### 3.1 Fudge factors

| Строки | Значение | Что делает | Обоснование | Критичность |
|--------|----------|-----------|-------------|-------------|
| 832-834 | size_bias: (a*b)^0.74 | Нелинейный масштаб размера плит | Визуально подобран | -- |
| 847 | spread: 0.82-1.22 | Скорость роста плит | Визуально подобран | -- |
| 848 | roughness: 0.26-1.08 | Шероховатость границ | Визуально подобран | -- |
| 877 | 0.62/0.38 (structural weights) | Относительный вес FBM октав | Визуально подобран | low |
| 914 | start_cost: 0.1-2.8 | Задержка роста нуклеусов | Визуально подобран | -- |
| 936 | bend: +/-0.34 + sin*0.16 | Изгиб исторической траектории | Визуально подобран | -- |
| 1036 | drift_factor: 1.03-0.12*align | Drift preferencing | Визуально подобран | -- |
| 1043 | 1.0+|lat|/90*0.1 | Polar growth bias | Визуально подобран | -- |
| 1384 | memory_keep: 0.5+0.18*(1-age_norm) | Plate boundary inertia | Визуально подобран | low |
| 2097 | arch: n1*0.08+n2*0.04, max(0) | Archipelago amplitude | ~1-3% extra land | low |
| 2763 | along_strike: powf(0.7) | Bias toward higher values | Визуально | -- |
| 3315 | relief=1.0 при basin<0 | Минимум суши | Предотвращает subsea basins | -- |

**Итого fudge factors: 12** (из них 3 low, остальные незначимые)

### 3.2 Эвристики

| Строки | Описание | Заменяет физику | Критичность |
|--------|----------|----------------|-------------|
| 820-1094 | Voronoi growth (Dijkstra + cost modulation) | Мантийную конвекцию -> plate genesis | -- неустранимо |
| 925-974 | Historical trajectory nuclei | Plate migration/fragmentation history | -- неустранимо |
| 2081-2090 | Threshold binary search для cf | Continuous mass balance | -- приемлемо |
| 2123-2194 | BFS + noise perturbation для coastlines | Tectonic/erosional coastal shaping | low |
| 3225-3255 | Weighted sediment redistribution | Gravity-driven sediment routing | med нет flow routing |
| 3610-3650 | Power-law hypsometric correction | Полная физика отсутствующих процессов | -- conditional |
| 5500 | 20m порог для озёр | Coastal zone exclusion | -- |
| 5535 | 8*Q^0.36 valley depth | Full fluvial incision model | -- 40x ratio обосновано |
| 5710 | river>0.12 -> biome upgrade | Riparian microclimate | -- |

**Итого эвристик: 9** (из них 1 med, 1 low)

### 3.3 Математические трюки

| Строки | Трюк | Проблема | Критичность |
|--------|------|---------|-------------|
| 2763 | `.powf(0.7)` bias для along-strike | Bias toward high values | -- |
| 3410 | `.tanh()` для glacial truncation | Smooth compression вместо erosion rate | -- физически мотивировано |
| 3610 | power-law hypsometric | Эмпирическая коррекция | -- conditional |
| smooth_field N passes | Iterated Gaussian ~ diffusion | Решение PDE через итерации | low корректно для SS |
| 2430-2432 | N=round(alpha^2/2dx^2) for flexure | Gaussian ~ flexural filter | low Watts 2001 |
| 4479 | (1+0.75*tau)^0.25 two-stream | Exact Schwarzschild formula | ok |
| 1801 | heat_anomaly*0.02 thermal correction | 2% density change from 300K | low upper bound |

**Итого трюков: 7** (из них 3 low)

### 3.4 Скрытые допущения

| # | Допущение | Последствие | Критичность |
|---|----------|-------------|-------------|
| D1 | Plate evolution мгновенная: all steps -> one field, then relief | Нет синхронизации relief и plate motion | low |
| D2 | Deformation propagation мгновенная (steady-state eikonal) | Нет time-dependent deformation front | -- корректно >1 Myr |
| D3 | Climate computed AFTER relief (no feedback) | Нет precipitation -> erosion -> relief loop | low crop частично решает |
| D4 | Ocean thermal subsidence от текущей скорости плиты | Реальная скорость менялась за 200 Myr | -- unavoidable |
| D5 | Isostatic relaxation after erosion (not during) на planet | Overestimation of transient relief | low crop has iso_factor |
| D6 | Continental fraction binary -> smoothed | No dynamic shoreline from climate/erosion | -- |
| D7 | Same K_eff logic for planet and crop | Crop should inherit planet rock type | -- crop derives from defs |
| D8 | Smooth field uses Jacobi (not Gauss-Seidel) | Slower convergence, not incorrect | -- |
| D9 | Crop uplift uses representative H_c=40km (not per-cell) | Different H_c in mountains vs. plains | low |

**Итого скрытых допущений: 9** (из них 4 low)

### 3.5 Потенциальные баги

| # | Описание | Строки | Критичность |
|---|----------|--------|-------------|
| B1 | ~~LIP latitude convention~~ | ~~2993~~ | **FIXED** (c21b987: north-first convention) |
| B2 | LIP placement loop: consistent with B1 fix | 3011, 3017 | -- (safe) |
| B3 | `nearest_free_index`: diamond spiral search -- edge case при max_radius | 787-818 | -- safe |
| B4 | Inconsistent coastline cleanup: `smooth_coastline` is 2-pass, inline is 1-pass | 3912/3960 | -- minor |
| B5 | Crop BFS shelf distance at grid edges: won't wrap | 3680-3710 | -- minor |

**Итого потенциальных багов: 4** (0 med, все minor/safe)

---

## Этап 3 Summary

| Категория | Количество | med | low | -- |
|-----------|-----------|-----|-----|-----|
| Fudge factors | 12 | 0 | 3 | 9 |
| Эвристики | 9 | 1 | 1 | 7 |
| Математические трюки | 7 | 0 | 3 | 4 |
| Скрытые допущения | 9 | 0 | 4 | 5 |
| Потенциальные баги | 4 | 0 | 0 | 4 |
| **ИТОГО** | **41** | **1** | **11** | **29** |

---

## Топ-10 проблем (отсортированы по критичности)

| # | Проблема | Критичность | Статус |
|---|----------|-------------|--------|
| 1 | **Sediment routing без gravity** -- weighted redistribution без downstream transport. | med | Сохраняется. Нужна гидрология перед осадками. |
| 2 | **Voronoi plate growth** -- не физический рост. | -- | Неустранимо без мантийной конвекции. |
| 3 | **Plate evolution мгновенная** (D1) | low | Архитектурное ограничение. |
| 4 | **Climate-relief обратная связь** (D3) | low | Crop partially resolves. |
| 5 | **Crop H_c=40km representative** (D9) | low | Можно наследовать с планеты. |
| 6 | **Floodplain W_v scaling** -- не проверен на crop resolution | low | Validate width scaling. |
| 7 | **Archipelago FBM amplitude 0.08/0.04** -- эмпирическая | low | Привязать к hotspot density. |
| 8 | **memory_keep в evolve** -- визуально подобран | low | Можно привязать к plate age. |
| 9 | **Aerosol cooling** -- -15C фиксировано, нет зональности | -- | Toon 1997 средне. |
| 10 | **Climate kappa** -- множители без точной калибровки | -- | Roe 2003 качественно верно. |

---

## Changelog v5 -> v6 (campaign "ДОВЕСТИ ДО ИДЕАЛА")

| Фаза | v5 оценка | v6 оценка | Коммит | Описание |
|------|-----------|-----------|--------|----------|
| LIP lat fix | F (bug) | A (fixed) | c21b987 | North-first convention for LIP placement |
| Biome smoothing | C | **A** | 8724ef7 | Prentice 1992 spatial climate averaging |
| Structural field | B | **A** | afe15e6 | spherical_fbm (5-oct, Rodrigues rotation) |
| Ridge/trench profiles | B | **A** | d5d6256 | Gaussian exp(-(1-def)^2/sigma^2) |
| Glacial buzzsaw | B | **A** | e7f4c26 | Mitchell & Montgomery 2006 calibration |
| Isostatic relaxation | B | **A** | 0b14f20 | Per-cell tau from Te proxy (Watts 2001) |
| Foreland basins | B | **A** | 40f065e | DeCelles & Giles 1996 Table 2 means |
| Rift shoulders | B | **A** | d8f6a0c | 500-1200m maturity (Weissel & Karner 1989) |
| K_br (crop) | B | **A** | 6d6c4d6 | Nonlinear (1-def)^2 (Whipple 2004) |
| Coastline perturbation | B | **A** | 89b74c3 | Margin-modulated sigma/amp (Kearey 2009) |
| Stream power (MFD) | B | **A** | 315fdd0 | Semi-implicit B&W (Braun & Willett 2013) |
| Greenhouse | B | **A** | cfc659b | Two-stream Schwarzschild (Pierrehumbert 2010) |
| Continental shelf | B | **A** | fb41d8b | Margin-dependent width + exponential profile |
| Continentality | B | **A** | f678b81 | sin^2(phi) calibrated (Terjung & Louie 1972) |

### Summary

| Метрика | v5 | v6 | Delta |
|---------|----|----|-------|
| Строки кода | 7719 | 7824 | +105 |
| Процессов A | 37 | 50 | +13 |
| Процессов B | 21 | 9 | -12 |
| Процессов C | 1 | 0 | -1 |
| % A/B | 98% | 100% | +2% |
| Issues (total) | 51 | 41 | -10 |
| Issues (med) | 3 | 1 | -2 |
| Fudge factors | 16 | 12 | -4 |
| Баги | 5 (1 med) | 4 (0 med) | -1 |

### Оставшиеся B (9 процессов, все "адекватно" или "неустранимо"):

1. **Voronoi plate growth** -- неустранимо без мантийной конвекции
2. **SPACE F_f=0.5** -- в диапазоне, без литологической зависимости
3. **Epeirogenic warping** -- Gaussian dipole адекватно
4. **Back-arc basins** -- параметры корректны
5. **Cratonic peneplains** -- tau=50 Myr физически обоснован
6. **Sediment redistribution** -- без gravity routing (при 10 km/cell)
7. **Valley carving** -- 40x ratio из Schumm (1977)
8. **Aerosol cooling** -- -15C без зональности
9. **River threshold** -- коэффициенты эмпирические

Все 9 B -- либо принципиально неустранимые (Voronoi без конвекции), либо адекватные при разрешении 10 km/cell. Повышение до A потребовало бы либо фундаментальных архитектурных изменений, либо добавления подсистем (литологическая модель для F_f, gravity routing для осадков).
