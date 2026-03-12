# Tectonica — Blueprint (март 2026, аудит v6)

Процедурная генерация планет на основе геофизики.
Движок: `rust/planet_engine/src/lib.rs` (Rust -> WASM, 7824 строки).
Аудит: `docs/tectonics/AUDIT_REPORT.md` (полная карта кода + оценки + 41 issue).

---

## Научная оценка (100% A/B)

| Оценка | Процессов | % | Значение |
|--------|-----------|---|----------|
| **A** | 50 | 85% | Научно корректно, параметры из литературы |
| **B** | 9 | 15% | Качественно верно, приблизительные параметры |
| **C** | 0 | 0% | Упрощение с потерей физики |
| **D** | 0 | 0% | Эвристика без обоснования |
| **F** | 0 | 0% | Ошибка |

---

## Реализованные модули

### Тектоника

| Модуль | Строки | Оценка | Источник | Проблемы |
|--------|--------|--------|---------|----------|
| PRNG (xoshiro128++) | 18-60 | A | Blackman & Vigna 2021 | -- |
| Voronoi plate growth | 820-1094 | B | Bird 2003 (статистика) | Неустранимо без мантийной конвекции |
| Plate evolution (semi-Lagrangian) | 1335-1538 | A | Torsvik et al. 2010 | spherical_fbm structural (5-oct, Rodrigues rotation) |
| Boundary classification | 1640-1760 | A | Bird 2003; DeMets 2010 | -- |
| Damage rheology | 2269-2293 | A | Lyakhovsky et al. 1997 | alpha=0.6/0.4/0.4 в диапазоне |
| Interior suppression | 2295-2358 | A | Artemieva & Mooney 2001 | L_rheol=300km -- среднее |
| Continental nuclei | 2020-2098 | A | Rogers & Santosh 2004 | -- |
| E&M crustal thickening | 3040-3130 | A | England & McKenzie 1982 | -- |

### Изостазия и литосфера

| Модуль | Строки | Оценка | Источник | Проблемы |
|--------|--------|--------|---------|----------|
| Airy isostasy + water loading | 1812-1840 | A | Turcotte & Schubert 2.2/2.6 | -- |
| Flexural isostasy (N=34) | 2402-2432 | A | Watts 2001 | Te=25km -- глобальное среднее |
| Thermal subsidence (GDH1) | 2600-2664 | A | Parsons & Sclater 1977; Stein & Stein 1992 | -- |
| Isostatic relaxation (per-cell tau) | 3257-3290 | A | Watts 2001 8.4 | Per-cell tau ~ Te^(3/4), 5-50 km Te range |
| Crustal thickness | 2361-2400 | A | Christensen & Mooney 1995 | -- |

### Рельеф и геоморфология

| Модуль | Строки | Оценка | Источник | Проблемы |
|--------|--------|--------|---------|----------|
| Stream power (planet, MFD) | 6014-6090 | A | Braun & Willett 2013; Salles 2023 | Semi-implicit B&W + MFD area. Unconditionally stable. |
| SPACE erosion (crop) | 6264-6460 | A | Shobe et al. 2017 | H*=1.0m, V_s=0.05 -- в диапазоне |
| B&W implicit scheme | 6121-6200 | A | Braun & Willett 2013 | -- |
| SPACE K_br (nonlinear) | ~7320 | A | Stock & Montgomery 1999; Whipple 2004 | (1-def)^2 power law |
| Sub-grid channel width | ~6310 | A | Pelletier 2010; Leopold & Maddock 1953 | -- |
| Climate-dependent diffusion | 3184-3223 | A | Fernandes & Dietrich 1997; Roe 2003 | CFL=0.002, стабильно |
| Volcanic arcs | 2727-2800 | A | Syracuse & Abers 2006 | d=166km, sigma=40km -- из Table 1 |
| Dynamic topography | 2800-2870 | A | Hager 1985; Flament 2013; Hoggard 2016 | -- |
| Hotspot volcanism | 2870-2963 | A | Morgan 1971; Crough 1983 | -- |
| Foreland basins | 3288-3340 | A | DeCelles & Giles 1996 Table 2; Beaumont 1981 | 175km foredeep, -120m -- средние из таблицы |
| Glacial buzzsaw | 3392-3420 | A | Mitchell & Montgomery 2006; Egholm 2009 | 600-1500m excess above ELA |
| Rift shoulders | 3420-3470 | A | Weissel & Karner 1989 | 500-1200m maturity-dependent |
| Mid-ocean ridges | 2666-2725 | A | Macdonald 1982; Stern 2002 | Gaussian profiles, sigma ~40-50 km |
| Epeirogenic warping | 3510-3560 | B | Bond 1976; Mitrovica 1989 | Gaussian dipole -- адекватно |
| Back-arc basins | 3565-3610 | B | Karig 1971; Sdrolias & Muller 2006 | -- |
| Cratonic peneplains | 3475-3510 | B | Pazzaglia & Brandon 1996; King 1967 | tau=50 Myr -- физически обосновано |
| Oceanic plateaus (LIPs) | 2975-3035 | A | Coffin & Eldholm 1994 | LIP lat fixed (north-first) |
| Continental shelf | 3665-3745 | A | Emery & Uchupi 1984; Watts 2001 10.2 | Margin-dependent width + exponential profile |
| Sediment redistribution | 3225-3255 | B | Milliman & Syvitski 1992 | Без gravity routing (адекватно для single-pass) |
| Detail noise beta=2.0 | 3747-3795 | A | Huang & Turcotte 1989 | -- |
| Coastline perturbation | 3796-3910 | A | Wessel & Smith 1996; Kearey et al. 2009 | Margin-modulated sigma/amplitude |
| Valley carving | 5535-5620 | B | Leopold & Maddock 1953; Schumm 1977 | 40x ratio -- geom. mean |

### Климат

| Модуль | Строки | Оценка | Источник | Проблемы |
|--------|--------|--------|---------|----------|
| T_sea(lat) polynomial | ~4430 | A | Peixoto & Oort 1992 | -- |
| Lapse rate 6.0 K/km | ~4445 | A | Holton & Hakim 2013 | -- |
| Greenhouse (two-stream) | 4469-4481 | A | Pierrehumbert 2010 §4.3/4.4 | Schwarzschild: tau=0.84*p^0.7, (1+3tau/4)^0.25 |
| Continentality (sin^2 phi) | 4483-4500 | A | Terjung & Louie 1972; Hartmann 1994; Conrad 1946 | 0.010*sin^2*dist, 2000km saturation |
| Zonal precipitation (GPCP) | ~4530 | A | Adler et al. 2003 | -- |
| 3-cell wind circulation | ~4560 | A | Peixoto & Oort 1992; Seidel 2008 | -- |
| Windward moisture (L=700km) | ~4600 | A | van der Ent & Savenije 2011 | -- |
| Rain shadow (0.40 mm/m) | ~4640 | A | Smith 1979; Galewsky 2009 | -- |
| Clausius-Clapeyron | ~4660 | A | Held & Soden 2006 | -- |
| Aerosol cooling | ~4490 | B | Toon et al. 1997 | -15C без зональности |
| Climate-dependent kappa | ~3215 | B | Roe et al. 2003 | Множители качественно верны |

### Гидрология, биомы, settlement

| Модуль | Строки | Оценка | Источник | Проблемы |
|--------|--------|--------|---------|----------|
| D8 flow direction | 5860-5900 | A | Tucker & Bras 2000 | -- |
| MFD accumulation (p=1.1) | 5970-6010 | A | Freeman 1991 | -- |
| Priority-Flood + endorheic | 5280-5400 | A | Barnes 2014; Budyko 1974 | -- |
| Budyko discharge Q | 5400-5460 | A | Budyko 1974 | -- |
| River threshold | ~5350 | B | Montgomery & Dietrich 1988 | Коэффициенты эмпирические |
| Strahler stream order | ~5510 | A | Strahler 1957 | -- |
| Channel geometry W=7.1*Q^0.5 | ~5560 | A | Leopold & Maddock 1953 | -- |
| Delta deposition | ~5600 | A | Paola 2011; Edmonds 2007 | -- |
| Whittaker biomes | 5690-5800 | A | Ricklefs & Relyea 2014 | -- |
| Alpine treeline | ~5760 | A | Korner 2003 | -- |
| Koppen ET tundra | ~5780 | B | Terjung 1970 | Seasonal amp 20C -- reasonable |
| Riparian vegetation | ~5790 | B | Diamond 1997 | Порог 0.12 -- эмпирический |
| Biome smoothing (Prentice 1992) | 5802-5840 | A | Prentice et al. 1992 BIOME model | Spatial T/P averaging at ecotones |
| Miami model NPP | ~4575 | A | Lieth 1975 | -- |

---

## Оставшиеся проблемы (по приоритету)

### med (1 шт.)

| # | Задача | Описание |
|---|--------|----------|
| P1 | Sediment routing без gravity | Weighted redistribution без downstream transport. Адекватно при 10 km/cell. |

### low (3 шт.)

| # | Задача | Описание |
|---|--------|----------|
| P2 | Plate evolution мгновенная | Все шаги -> один field -> relief. Архитектурное ограничение. |
| P3 | Climate-relief feedback | Climate после relief, нет обратной связи. Crop частично решает. |
| P4 | Crop H_c=40km representative | Можно наследовать per-cell с планеты. |

### -- (неустранимо или адекватно)

- Voronoi plate growth -- неустранимо без мантийной конвекции
- Plate growth fudge factors (F1-F8) -- неустранимо
- Shelf BFS -- адекватно при 10 km/cell
- Hypsometric correction -- safety valve, conditional
- 40x valley ratio -- geom. mean Schumm 1977
- Aerosol cooling -15C -- Toon 1997 среднее
- Climate kappa -- Roe 2003 качественно верно

---

## Полный список хаков и эвристик

### Fudge factors (12 шт.)

| # | Строки | Значение | Что делает | Критичность |
|---|--------|----------|-----------|-------------|
| F1 | 832-834 | (a*b)^0.74 | Нелинейный масштаб размера плит | -- |
| F2 | 847 | 0.82-1.22 | Spread -- скорость роста плит | -- |
| F3 | 848 | 0.26-1.08 | Roughness -- шероховатость границ | -- |
| F4 | 877 | 0.62/0.38 | Вес FBM октав structural field | low |
| F5 | 914 | 0.1-2.8 | Start cost нуклеусов | -- |
| F6 | 936 | +/-0.34 + sin*0.16 | Изгиб исторической траектории | -- |
| F7 | 1036 | 1.03-0.12*align | Drift preferencing | -- |
| F8 | 1043 | 1.0+lat/90*0.1 | Polar growth bias | -- |
| F9 | 1384 | 0.5+0.18*(1-age) | Plate boundary inertia | low |
| F10 | 2097 | 0.08/0.04 | Archipelago FBM amplitude | low |
| F11 | 2763 | powf(0.7) | Along-strike bias | -- |
| F12 | 3315 | relief=1.0 | Минимум суши в бассейнах | -- |

### Эвристики (9 шт.)

| # | Строки | Описание | Заменяет | Критичность |
|---|--------|----------|---------|-------------|
| E1 | 820-1094 | Voronoi growth (Dijkstra) | Мантийную конвекцию | -- неустранимо |
| E2 | 925-974 | Historical trajectory | Plate migration history | -- неустранимо |
| E3 | 2081-2090 | Binary search для cf | Continuous mass balance | -- приемлемо |
| E4 | 2123-2194 | BFS + noise для coastlines | Tectonic/erosional shaping | low |
| E5 | 3225-3255 | Weighted sediment redistribution | Gravity-driven routing | med |
| E6 | 3610-3650 | Hypsometric correction | Полная физика | -- safety valve |
| E7 | ~5500 | 20m порог озёр | Coastal exclusion | -- |
| E8 | 5535 | 8*Q^0.36 valley depth | Fluvial incision | -- 40x обосновано |
| E9 | ~5790 | river>0.12 biome upgrade | Riparian microclimate | -- |

### Скрытые допущения (9 шт.)

| # | Описание | Критичность |
|---|----------|-------------|
| D1 | Plate evolution мгновенная (все шаги -> один field -> relief) | low |
| D2 | Deformation propagation мгновенная (steady-state eikonal) | -- корректно >1 Myr |
| D3 | Climate computed AFTER relief (no feedback loop) | low crop частично решает |
| D4 | Ocean thermal subsidence от текущей скорости плиты | -- unavoidable |
| D5 | Isostatic relaxation after erosion (not during) на planet | low crop has iso_factor |
| D6 | Continental fraction binary -> smoothed (no dynamic shoreline) | -- |
| D7 | Same K_eff logic for planet and crop | -- crop derives from defs |
| D8 | Smooth field uses Jacobi (not Gauss-Seidel) | -- slower but correct |
| D9 | Crop uplift uses representative H_c=40km (not per-cell) | low |

---

## Устранённые хаки (история)

| Дата | Хак | Замена |
|------|-----|--------|
| Фев 2026 | LCG PRNG | xoshiro128++ (Blackman & Vigna 2021) |
| Фев 2026 | Нет water loading | Airy + rho_w = 1025 (Turcotte & Schubert 2.6) |
| Фев 2026 | Нет thermal subsidence | Parsons & Sclater 1977 + GDH1 |
| Фев 2026 | 12 проходов (подобрано) | N = alpha^2/(2dx^2) = 34 из Te=25 км |
| Фев 2026 | Noise +/-5 m после эрозии | Stochastic flow +/-5% (Tucker & Bras 2000) |
| Фев 2026 | Decision tree биомы | Polygon lookup Уиттакера (Shimrat 1962) |
| Фев 2026 | Uplift = 4 константы по boundary_type | Деформационные поля conv/div/trans |
| Фев 2026 | Stale receiver в SPACE | Обновлённый приёмник (Braun & Willett 2013) |
| Мар 2026 | V_s=0.5, F_f=0.25 -> заполнение каналов | V_s=0.05, F_f=0.5, H*=1.0 |
| Мар 2026 | Uplift crop = полные тектонические скорости | 33% maintenance (Willett & Brandon 2002) |
| Мар 2026 | Произвольные uplift коэффициенты | England & McKenzie 1982 crustal thickening |
| Мар 2026 | Линейная пропагация деформации | Lyakhovsky 1997 damage rheology |
| Мар 2026 | Шумовая динамическая топография | Slab-pull + ridge upwelling (Hager 1985; Flament 2013) |
| Мар 2026 | Нет вулканических дуг | Syracuse & Abers 2006: arc at 166km |
| Мар 2026 | Нет хотспотов | Morgan 1971 / Crough 1983: 5-15 swells |
| Мар 2026 | Нет форландовых бассейнов | DeCelles & Giles 1996: foredeep + forebulge |
| Мар 2026 | Нет гляциальной эрозии | Brozovic 1997: latitude-dependent ELA buzzsaw |
| Мар 2026 | Нет рифтовых плеч | Weissel & Karner 1989: uplift at 100km |
| Мар 2026 | Равномерный kappa | Roe 2003: Hadley cell climate-dependent kappa |
| Мар 2026 | Плоский шельф | Kennett 1982: shelf break at -130m, BFS profile |
| Мар 2026 | Нет перераспределения осадков | Milliman & Syvitski 1992: 60% на суше |
| Мар 2026 | Нет кратонных пенепленов | Pazzaglia & Brandon 1996: tau=50 Myr exp. decay |
| Мар 2026 | Нет океанических плато | Coffin & Eldholm 1994: 3-8 LIPs |
| Мар 2026 | Плоское океаническое дно | Macdonald 1982: ridge/trench/fracture |
| Мар 2026 | Нет эпейрогенического варпинга | Mitrovica 1989; Bond 1976: +/-200m tilt |
| Мар 2026 | Нет задуговых бассейнов | Karig 1971; Sdrolias & Muller 2006 |
| Мар 2026 | Нет Priority-Flood pit-filling | Barnes 2014 + Budyko endorheic basins |
| Мар 2026 | Cell-count accumulation | Budyko 1974 physical discharge Q [m^3/s] |
| Мар 2026 | H*=2.0m выше рекомендованного | H*=1.0m (Shobe 2017 верхняя граница) |
| Мар 2026 | V_s=0.1 -> недостаточное врезание | V_s=0.05 (согласовано с H*=1.0) |
| Мар 2026 | 80x valley ratio -> верхний край | 40x (geom. mean Schumm 1977) |
| Мар 2026 | Peneplain 40% flatten -> не из физики | tau_denudation=50 Myr (Pazzaglia & Brandon 1996) |
| **Мар 2026** | **LIP lat convention bug** | **North-first: (PI/2-lat)/PI*height** |
| **Мар 2026** | **Biome mode filter (no physics)** | **Prentice 1992 spatial climate averaging** |
| **Мар 2026** | **sin/cos structural field** | **spherical_fbm (5-oct Rodrigues)** |
| **Мар 2026** | **def^3/def^2 ridge/trench** | **Gaussian exp(-(1-def)^2/sigma^2)** |
| **Мар 2026** | **Glacial buzzsaw 2500-1000*int** | **Mitchell & Montgomery 2006: 600+900*(1-int)** |
| **Мар 2026** | **Iso relax one tau=5Myr** | **Per-cell tau from Te^0.75 (Watts 2001)** |
| **Мар 2026** | **Foreland upper-bound amplitudes** | **DeCelles & Giles 1996 Table 2 means** |
| **Мар 2026** | **Rift shoulders 400m fixed** | **500-1200m maturity (Weissel & Karner 1989)** |
| **Мар 2026** | **Linear K_br in SPACE** | **Nonlinear (1-def)^2 (Whipple 2004)** |
| **Мар 2026** | **Coastline sigma empirical** | **Margin-modulated (Kearey et al. 2009)** |
| **Мар 2026** | **MFD explicit + 30% cap** | **Semi-implicit B&W (Braun & Willett 2013)** |
| **Мар 2026** | **Gray atmosphere p^0.3** | **Two-stream Schwarzschild (Pierrehumbert 2010)** |
| **Мар 2026** | **Shelf 15-cell fixed width** | **Margin-dependent + exponential (Emery & Uchupi 1984)** |
| **Мар 2026** | **Continentality 0.008*sin(lat)** | **0.010*sin^2(phi) calibrated (Terjung 1972)** |

---

## Завершённые фазы

| Фаза | Описание | Источник |
|------|----------|---------|
| Level 0 | Замена fudge factors на физику (E&M uplift, Lyakhovsky deformation) | England & McKenzie 1982; Lyakhovsky 1997 |
| Level 1.1 | Вулканические дуги на субдукционных зонах | Syracuse & Abers 2006 |
| Level 1.2 | Структурная динамическая топография | Hager 1985; Flament 2013 |
| Level 1.3 | Хотспотовый вулканизм | Morgan 1971; Crough 1983 |
| Level 2.1 | Форландовые бассейны | DeCelles & Giles 1996 |
| Level 2.2 | Гляциальная пила | Brozovic et al. 1997 |
| Level 2.3 | Рифтовые плечи | Weissel & Karner 1989 |
| Level 3.1 | Климато-зависимая эрозия | Roe et al. 2003 |
| Level 3.2 | Профиль континентального шельфа | Kennett 1982 |
| Level 4.1 | Перераспределение осадков | Milliman & Syvitski 1992 |
| Level 4.2 | Кратонные пенеплены (tau_denudation) | Pazzaglia & Brandon 1996; King 1967 |
| Level 4.3 | Океанические плато / LIPs | Coffin & Eldholm 1994 |
| Level 5.1 | Срединно-океанические хребты | Macdonald 1982; Stern 2002 |
| Level 5.2 | Эпейрогенический варпинг | Mitrovica 1989; Bond 1976 |
| Level 5.3 | Задуговые бассейны | Karig 1971; Sdrolias & Muller 2006 |
| Phase H13 | Climate-coupled runoff в SPACE | Roe 2003 |
| Phase H16 | Изостатическая разгрузка в B&W solver | Molnar & England 1990 |
| River R0-R5 | Priority-Flood, Budyko, Strahler, channel geom, deltas | Barnes 2014; Budyko 1974; Leopold 1953 |
| SPACE cal | H*=1.0m, V_s=0.05 m/yr (in-range params) | Shobe et al. 2017 |
| **IDEAL-01** | **LIP lat convention fix** | **North-first WorldCache convention** |
| **IDEAL-02** | **Biome smoothing: Prentice 1992** | **Spatial climate averaging at ecotones** |
| **IDEAL-03** | **Structural field: spherical_fbm** | **5-octave FBM, Rodrigues domain rotation** |
| **IDEAL-04** | **Ridge/trench Gaussian profiles** | **exp(-(1-def)^2/sigma^2), sigma from lit.** |
| **IDEAL-05** | **Glacial buzzsaw calibration** | **Mitchell & Montgomery 2006 peak-ELA** |
| **IDEAL-06** | **Per-cell isostatic tau** | **tau ~ Te^0.75 (Watts 2001 §8.4)** |
| **IDEAL-07** | **Foreland basin calibration** | **DeCelles & Giles 1996 Table 2 means** |
| **IDEAL-08** | **Rift shoulder maturity** | **500-1200m (Weissel & Karner 1989)** |
| **IDEAL-09** | **Nonlinear K_br** | **Whipple 2004: (1-def)^2 power law** |
| **IDEAL-10** | **Margin-modulated coastline** | **Kearey et al. 2009 active/passive** |
| **IDEAL-11** | **Semi-implicit MFD stream power** | **B&W 2013 implicit + MFD area** |
| **IDEAL-12** | **Two-stream greenhouse** | **Pierrehumbert 2010 Schwarzschild eq.** |
| **IDEAL-13** | **Margin-dependent shelf** | **Emery & Uchupi 1984 + Watts 2001** |
| **IDEAL-14** | **Calibrated continentality** | **sin^2(phi), 2000km sat. (Terjung 1972)** |

---

## Файлы документации

| # | Файл | Тема |
|---|------|------|
| -- | [AUDIT_REPORT](AUDIT_REPORT.md) | Полный научно-технический аудит v6 (март 2026) |
| 01 | [01_ОБЗОР](01_ОБЗОР.md) | Архитектура, pipeline, входы/выходы |
| 02 | [02_СЕТКА](02_СЕТКА.md) | Сетка, координаты, обёртка полюсов |
| 03 | [03_ПЛИТЫ](03_ПЛИТЫ.md) | Генерация и эволюция тектонических плит |
| 04 | [04_ГРАНИЦЫ](04_ГРАНИЦЫ.md) | Классификация границ плит |
| 05 | [05_КОРА](05_КОРА.md) | Мощность коры, типы пород |
| 06 | [06_ИЗОСТАЗИЯ](06_ИЗОСТАЗИЯ.md) | Изостазия Эйри, флексуральное сглаживание |
| 07 | [07_ДЕФОРМАЦИЯ](07_ДЕФОРМАЦИЯ.md) | Пропагация деформации, interior suppression |
| 08 | [08_ЭРОЗИЯ](08_ЭРОЗИЯ.md) | Stream power (planet), SPACE (crop) |
| 09 | [09_ТЕМПЕРАТУРА](09_ТЕМПЕРАТУРА.md) | Температурная модель |
| 10 | [10_ОСАДКИ](10_ОСАДКИ.md) | Осадки, ветер, rain shadow |
| 11 | [11_БИОМЫ](11_БИОМЫ.md) | Классификация Уиттакера, альпийская зона |
| 12 | [12_РЕКИ](12_РЕКИ.md) | Гидрология, долины, озёра |
| 13 | [13_КРОП](13_ОСТРОВ.md) | Crop scope: island + continent pipeline |
| 14 | [14_СОБЫТИЯ](14_СОБЫТИЯ.md) | Метеориты, рифты, ocean shift |
| 15 | [15_ХАКИ](15_ХАКИ.md) | Известные хаки и эвристики |
| 16 | [16_ЛИТЕРАТУРА](16_ЛИТЕРАТУРА.md) | Список источников |

---

## Принцип

Каждый параметр указан с точным значением из кода и источником.
Где физика упрощена или подобрана вручную -- указано явно.
