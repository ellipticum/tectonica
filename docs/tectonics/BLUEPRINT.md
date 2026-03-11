# Tectonica — Blueprint (март 2026, аудит v4)

Процедурная генерация планет на основе геофизики.
Движок: `rust/planet_engine/src/lib.rs` (Rust -> WASM, 6904 строки).
Аудит: `docs/tectonics/AUDIT_REPORT.md` (полная карта кода + оценки + 58 issue).

---

## Научная оценка (91% A/B)

| Оценка | Процессов | % | Значение |
|--------|-----------|---|----------|
| **A** | 33 | 60% | Научно корректно, параметры из литературы |
| **B** | 17 | 31% | Качественно верно, приблизительные параметры |
| **C** | 4 | 7% | Упрощение с потерей физики |
| **D** | 0 | 0% | Эвристика без обоснования |
| **F** | 1 | 2% | Ошибка (CFL violation) |

---

## Реализованные модули

### Тектоника

| Модуль | Строки | Оценка | Источник | Проблемы |
|--------|--------|--------|---------|----------|
| PRNG (xoshiro128++) | 18-60 | A | Blackman & Vigna 2021 | — |
| Voronoi plate growth | 820-1094 | B | Bird 2003 (статистика) | Неустранимо без мантийной конвекции |
| Plate evolution (semi-Lagrangian) | 1335-1538 | B | Torsvik et al. 2010 | sin/cos structural field вместо FBM (🟡) |
| Boundary classification | 1640-1760 | A | Bird 2003; DeMets 2010 | — |
| Damage rheology | 2269-2293 | A | Lyakhovsky et al. 1997 | alpha=0.6/0.4/0.4 в диапазоне |
| Interior suppression | 2295-2358 | A | Artemieva & Mooney 2001 | L_rheol=300km — среднее |
| Continental nuclei | 1963-2098 | A | Rogers & Santosh 2004 | — |
| E&M crustal thickening | 3028-3118 | A | England & McKenzie 1982 | — |

### Изостазия и литосфера

| Модуль | Строки | Оценка | Источник | Проблемы |
|--------|--------|--------|---------|----------|
| Airy isostasy + water loading | 1789-1810 | A | Turcotte & Schubert 2.2/2.6 | — |
| Flexural isostasy (N=34) | 2402-2432 | A | Watts 2001 | Te=25km — глобальное среднее |
| Thermal subsidence (GDH1) | 2530-2664 | A | Parsons & Sclater 1977; Stein & Stein 1992 | — |
| Isostatic relaxation (tau=5Myr) | 3239-3248 | B | Watts 2001 8.4 | Одно tau для всех — упрощение |
| Crustal thickness | 2361-2400 | A | Christensen & Mooney 1995 | — |

### Рельеф и геоморфология

| Модуль | Строки | Оценка | Источник | Проблемы |
|--------|--------|--------|---------|----------|
| Stream power (planet, MFD) | 5264-5343 | B | Braun & Willett 2013 | Explicit + cap 30% — не convergent |
| SPACE erosion (crop) | 5495-5695 | A | Shobe et al. 2017 | H*=2.0m выше рекомендованного (🟡) |
| B&W implicit scheme | 5352-5436 | A | Braun & Willett 2013 | — |
| Sub-grid channel width | 5595-5610 | A | Pelletier 2010; Leopold & Maddock 1953 | — |
| Climate-dependent diffusion | 3120-3204 | A/F | Fernandes & Dietrich 1997; Roe 2003 | **CFL violation** kappa*dt/dx2=2.0 (🔴) |
| Volcanic arcs | 2707-2774 | A | Syracuse & Abers 2006 | d=166km, sigma=40km — из Table 1 |
| Dynamic topography | 2778-2859 | A | Hager 1985; Flament 2013; Hoggard 2016 | — |
| Hotspot volcanism | 2861-2949 | A | Morgan 1971; Crough 1983 | Latitude convention mismatch (🟡) |
| Foreland basins | 3256-3318 | B | DeCelles & Giles 1996 | Амплитуды — верхняя граница |
| Glacial buzzsaw | 3320-3377 | B | Brozovic et al. 1997 | ELA — линейная аппроксимация |
| Rift shoulders | 3379-3417 | B | Weissel & Karner 1989 | 400m — консервативно |
| Mid-ocean ridges | 2666-2704 | B | Macdonald 1982; Stern 2002 | def^3 focusing — эвристика |
| Cratonic peneplains | 3419-3448 | C | King 1967; Fairbridge 1980 | 40% flatten — не из физики (🟡) |
| Epeirogenic warping | 3450-3511 | B | Bond 1976; Mitrovica 1989 | Линейный тилт — грубо |
| Back-arc basins | 3513-3552 | B | Karig 1971; Sdrolias & Muller 2006 | — |
| Oceanic plateaus (LIPs) | 2951-3024 | A | Coffin & Eldholm 1994 | — |
| Continental shelf | 3617-3697 | B | Kennett 1982; Emery & Uchupi 1984 | BFS перезаписывает физику (🟡) |
| Sediment redistribution | 3207-3237 | B | Milliman & Syvitski 1992 | Нет gravity routing |
| Detail noise beta=2.0 | 3699-3746 | A | Huang & Turcotte 1989 | — |
| Coastline perturbation (Gaussian) | 3748-3823 | B | Wessel & Smith 1996 | sigma эмпирические |
| Valley carving | 4788-4896 | B | Leopold & Maddock 1953 | 80x ratio — верхний край (🟡) |

### Климат

| Модуль | Строки | Оценка | Источник | Проблемы |
|--------|--------|--------|---------|----------|
| T_sea(lat) polynomial | 4135-4140 | A | Peixoto & Oort 1992 | — |
| Lapse rate 6.0 K/km | 4143 | A | Holton & Hakim 2013 | — |
| Greenhouse effect | 4148 | B | Pierrehumbert 2010 4.3 | Gray-atmosphere |
| Continentality | 4155 | B | Terjung & Louie 1972 | — |
| Zonal precipitation (GPCP) | 4250-4270 | A | Adler et al. 2003 | — |
| 3-cell wind circulation | 4300-4380 | A | Peixoto & Oort 1992; Seidel 2008 | — |
| Windward moisture (L=700km) | 4390-4430 | A | van der Ent & Savenije 2011 | — |
| Rain shadow (0.40 mm/m) | 4435-4455 | A | Smith 1979; Galewsky 2009 | — |
| Clausius-Clapeyron | 4456 | A | Held & Soden 2006 | — |

### Гидрология, биомы, settlement

| Модуль | Строки | Оценка | Источник | Проблемы |
|--------|--------|--------|---------|----------|
| D8 flow direction | 5145-5185 | A | Tucker & Bras 2000 | — |
| MFD accumulation (p=1.1) | 5204-5248 | A | Freeman 1991 | — |
| Lake detection | 4710-4786 | C | — | Нет pit-filling (Priority-Flood) |
| Whittaker biomes | 4898-5097 | A | Ricklefs & Relyea 2014 | — |
| Alpine treeline | 5040-5050 | A | Korner 2003 | — |
| Biome smoothing (2-pass mode) | 5070-5097 | C | — | Нет физического обоснования |
| Miami model NPP | 4507-4542 | A | Lieth 1975 | — |

---

## Критические проблемы (по приоритету)

### 🔴 P1: CFL violation в планетарной диффузии
**Строки**: 3181-3204
**Проблема**: 12 итераций диффузионной эрозии, каждая с ПОЛНЫМ dt (7-15 Myr).
kappa*dt/dx^2 = 0.02 * 10e6 / 10000^2 = 2.0. CFL требует < 0.25.
**Следствие**: Числовая нестабильность, нефизичные осцилляции рельефа.
**Исправление**: Разбить на sub-steps: `kdt = kappa * dt_yr / N_passes` вместо `kdt = kappa * dt_yr`. Или: N_sub = ceil(kappa*dt/(0.2*dx^2)) ~10 sub-steps per pass. Или: implicit diffusion (трёхдиагональная матрица, ADI).

---

### 🔴 P2: dt = total evolution time (не разделён на passes)
**Строки**: 3181-3204 (то же что P1)
**Проблема**: `kdt = kappa * climate_factor[i] * dt_yr` — dt_yr = total evolution time = 7-15 Myr. Не поделено на 12 (число проходов). Проходы работают как 12 шагов с полным dt каждый.
**Исправление**: `dt_per_pass = dt_yr / n_passes` или sub-stepping.

> Примечание: P1 и P2 — одна и та же проблема. Исправление одного решает оба.

---

### 🟡 P3: H* = 2.0m выше рекомендованного
**Строка**: 5478 (crop SPACE)
**Проблема**: Shobe et al. 2017 рекомендуют 0.1-1.0m. Текущее значение 2.0m — вдвое выше верхней границы.
**Следствие**: Слишком сильное экранирование bedrock осадками, может подавлять врезание каналов.
**Исправление**: Снизить до 1.0m и перекалибровать V_s (возможно увеличить) для сохранения глубины врезания. Или: документировать как "thick alluvium" для равнинных условий.

---

### 🟡 P4: Continental shelf BFS перезаписывает физику
**Строки**: 3617-3697
**Проблема**: BFS-based shelf reshaping с blend коэффициентом 0.7 перезаписывает изостатический профиль. Нет физической модели седиментации.
**Исправление**: Заменить direct override на модель седиментарного заполнения с учётом flexure. Или: использовать как fallback при `abs(h + 130) < threshold` вместо unconditional blend.

---

### 🟡 P5: Structural field в evolve — sin/cos, не FBM
**Строки**: 1431-1436
**Проблема**: Inline sin/cos без пространственной когерентности между шагами эволюции. Каждый шаг создаёт новый random structural pattern.
**Исправление**: Заменить на `spherical_fbm(seed_per_step)` — аналогичная стоимость, лучшая когерентность.

---

### 🟡 P6: Hotspot latitude convention mismatch
**Строки**: 2910-2913
**Проблема**: Inline формула `(y+0.5)/height * pi - pi/2` может давать инвертированную широту относительно WorldCache (`90 - y*180/h`). В текущем коде результат симметричен по абс. значению высоты, но хотспоты могут оказаться в неправильном полушарии.
**Исправление**: Использовать `CellCache::lat_deg` вместо inline формулы.

---

### 🟡 P7: Valley carving 80x incision ratio
**Строки**: 4837
**Проблема**: `valley_depth = 80 * 0.2 * Q^0.36 = 16 * Q^0.36`. 80x — верхний край диапазона 20-200x (Schumm 1977).
**Исправление**: Снизить до 40x (geom. mean ~63). Или: масштабировать по rock type (K_eff).

---

### 🟡 P8: Coast distance Chamfer — дупликат диагонали
**Строки**: 4187, 4204
**Проблема**: Диагональ (-1, 1) в обоих forward и backward pass. По алгоритму Borgefors 1986 должна быть только в backward.
**Следствие**: Систематическая ошибка расстояний в NW-SE направлении. Небольшая (~7%).
**Исправление**: Убрать (-1, 1) из forward pass.

---

### 🟡 P9: Peneplain flatten 40% — не из физики
**Строки**: 3444
**Проблема**: `flatten_strength = stability * 0.4 * cf`. Конкретное число 0.4 не из литературы.
**Исправление**: Привязать к денудационному времени: flatten ~ 1 - exp(-t_stable / tau_denudation), tau ~50 Myr (Pazzaglia & Brandon 1996).

---

### 🟡 P10: 3 системы coastline perturbation
**Строки**: 2123-2194, 2196-2223, 3748-3823
**Проблема**: Pre-smooth perturbation (BFS + 4 octaves), CF smoothing (10+3 passes), Gaussian perturbation (3 passes sigma=2000/1000/500m). Три отдельные системы, трудно калибровать.
**Исправление**: Объединить в 1-2 системы с чёткими пространственными масштабами.

---

## Полный список хаков и эвристик

### Fudge factors (18 шт.)

| # | Строки | Значение | Что делает | Критичность |
|---|--------|----------|-----------|-------------|
| F1 | 832-834 | (a*b)^0.74 | Нелинейный масштаб размера плит | ⚪ |
| F2 | 847 | 0.82-1.22 | Spread — скорость роста плит | ⚪ |
| F3 | 848 | 0.26-1.08 | Roughness — шероховатость границ | ⚪ |
| F4 | 877 | 0.62/0.38 | Вес FBM октав structural field | 🟡 |
| F5 | 914 | 0.1-2.8 | Start cost нуклеусов | ⚪ |
| F6 | 936 | +-0.34 + sin*0.16 | Изгиб исторической траектории | ⚪ |
| F7 | 1036 | 1.03-0.12*align | Drift preferencing | ⚪ |
| F8 | 1043 | 1.0+lat/90*0.1 | Polar growth bias | ⚪ |
| F9 | 1384 | 0.5+0.18*(1-age) | Plate boundary inertia | 🟡 |
| F10 | 1431-1436 | sin/cos structural | Модуляция эволюции плит | 🟡 |
| F11 | 2077 | 0.20/0.10 | Archipelago FBM amplitude | 🟡 |
| F12 | 2691 | def^3 | Cubic focusing ridge/trench | 🟡 |
| F13 | 2701 | def^2 * 400m | Fracture zone amplitude | ⚪ |
| F14 | 2763 | powf(0.7) | Along-strike bias | ⚪ |
| F15 | 3315 | relief=1.0 | Минимум суши в бассейнах | ⚪ |
| F16 | 3368 | 2500-1000*int. | Glacial buzzsaw ceiling | 🟡 |
| F17 | 3444 | 0.4 | Peneplain flatten strength | 🟡 |
| F18 | 3584 | clamp 1.0-8.0 | Hypsometric power | ⚪ |

### Эвристики (12 шт.)

| # | Строки | Описание | Заменяет | Критичность |
|---|--------|----------|---------|-------------|
| E1 | 820-1094 | Voronoi growth (Dijkstra) | Мантийную конвекцию | ⚪ неустранимо |
| E2 | 925-974 | Historical trajectory | Plate migration history | ⚪ неустранимо |
| E3 | 1431-1436 | sin/cos structural evolve | Литосферная гетерогенность | 🟡 заменить на FBM |
| E4 | 2081-2090 | Binary search для cf | Continuous mass balance | ⚪ приемлемо |
| E5 | 2123-2194 | BFS + noise для coastlines | Tectonic/erosional shaping | 🟡 |
| E6 | 3207-3237 | Weighted sediment redistribution | Gravity-driven routing | 🟡 нет flow routing |
| E7 | 3556-3604 | Hypsometric correction | Полная физика | ⚪ safety valve |
| E8 | 3617-3697 | BFS shelf reshaping | Margin sedimentation + flexure | 🟡 перезаписывает |
| E9 | 4710 | 20m порог озёр | Coastal exclusion | ⚪ |
| E10 | 4837 | 16*Q^0.36 valley depth | Fluvial incision | 🟡 80x ratio |
| E11 | 5057 | river>0.12 biome upgrade | Riparian microclimate | ⚪ |
| E12 | 6087 | mix_score scoring | Region selection | ⚪ не физика |

### Скрытые допущения (10 шт.)

| # | Описание | Критичность |
|---|----------|-------------|
| D1 | Plate evolution мгновенная (все шаги -> один field -> relief) | 🟡 |
| D2 | Deformation propagation мгновенная (steady-state eikonal) | ⚪ корректно >1 Myr |
| D3 | Climate computed AFTER relief (no feedback loop) | 🟡 crop частично решает |
| D4 | Ocean thermal subsidence от текущей скорости плиты | ⚪ unavoidable |
| D5 | Isostatic relaxation after erosion (not during) на planet | 🟡 crop has iso_factor |
| D6 | Continental fraction binary -> smoothed (no dynamic shoreline) | ⚪ |
| D7 | Same K_eff logic for planet and crop | ⚪ crop derives from defs |
| D8 | **dt for diffusion = total evolution time (CFL violation)** | 🔴 = P1/P2 |
| D9 | Smooth field uses Jacobi (not Gauss-Seidel) | ⚪ slower but correct |
| D10 | Crop uplift uses representative H_c=40km (not per-cell) | 🟡 |

### Потенциальные баги (8 шт.)

| # | Строки | Описание | Критичность |
|---|--------|----------|-------------|
| B1 | 3181-3204 | **CFL violation**: kappa*dt/dx^2 = 2.0, нужно <0.25 | 🔴 = P1 |
| B2 | 2910-2913 | Hotspot latitude convention mismatch | 🟡 = P6 |
| B3 | 787-818 | nearest_free_index diamond spiral — edge case | ⚪ safe |
| B4 | 4187/4204 | Coast distance: duplicate diagonal (-1,1) | 🟡 = P8 |
| B5 | 3825/3868 | Inconsistent coastline cleanup (1 vs 2 pass) | ⚪ |
| B6 | 1808 | Division by zero in isostatic_elevation | ⚪ safe (denom >= 2275) |
| B7 | 1431-1436 | Structural field no spatial coherence | ⚪ = P5 |
| B8 | 3650-3665 | Crop BFS shelf no grid wrapping | ⚪ |

---

## Отсутствующая физика

| # | Процесс | Последствие пропуска | Статус |
|---|---------|---------------------|--------|
| G1 | Осадочная нагрузка -> прогиб | Шельфы слишком крутые | Частично: профиль шельфа + перераспр. осадков |
| G2 | Климат-эрозия связь | Постоянный сток | ✅ Hadley cell kappa + climate runoff в SPACE |
| G3 | Изостатическая разгрузка от эрозии | Crop горы без rebound | ✅ B&W iso_factor в SPACE |
| G4 | Возраст литосферы -> K_eff | Старые породы не твёрже | Открыто (🟡) |
| G5 | Вулканические острова (hotspot) | Нет гавайского типа | ✅ 5-15 hotspot swells |
| G6 | Эвстазия | Нет изменений уровня моря | Открыто (⚪) |
| G7 | Sub-grid channel width | K_eff не масштабируется с dx | ✅ Pelletier 2010 в SPACE |
| G8 | Pit-filling / Priority-Flood | Underpredicts lakes | Открыто (🟡) — lake detection rated C |
| G9 | Sediment gravity routing | Нет downstream transport | Открыто (🟡) — текущая redistribution = weighted |

---

## Устранённые хаки (история)

| Дата | Хак | Замена |
|------|-----|--------|
| Фев 2026 | LCG PRNG | xoshiro128++ (Blackman & Vigna 2021) |
| Фев 2026 | Нет water loading | Airy + rho_w = 1025 (Turcotte & Schubert 2.6) |
| Фев 2026 | Нет thermal subsidence | Parsons & Sclater 1977 + GDH1 |
| Фев 2026 | 12 проходов (подобрано) | N = alpha^2/(2dx^2) = 34 из Te=25 км |
| Фев 2026 | Noise +/-5 m после эрозии | Stochastic flow +/-5% (Tucker & Bras 2000) |
| Фев 2026 | Decision tree биомы | Polygon lookup Уиттекера (Shimrat 1962) |
| Фев 2026 | Uplift = 4 константы по boundary_type | Деформационные поля conv/div/trans |
| Фев 2026 | Stale receiver в SPACE | Обновлённый приёмник (Braun & Willett 2013) |
| Мар 2026 | V_s=0.5, F_f=0.25 -> заполнение каналов | V_s=0.1, F_f=0.5, H*=2.0 |
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
| Мар 2026 | Нет кратонных пенепленов | King 1967: выравнивание при low activity |
| Мар 2026 | Нет океанических плато | Coffin & Eldholm 1994: 3-8 LIPs |
| Мар 2026 | Плоское океаническое дно | Macdonald 1982: ridge/trench/fracture |
| Мар 2026 | Нет эпейрогенического варпинга | Mitrovica 1989; Bond 1976: +/-200m tilt |
| Мар 2026 | Нет задуговых бассейнов | Karig 1971; Sdrolias & Muller 2006 |

---

## Приоритеты улучшений

### 🔴 Критические (ошибки — исправить немедленно)

| # | Задача | Сложность | Описание |
|---|--------|-----------|----------|
| P1/P2 | CFL fix в planet diffusion | Низкая | `dt_per_pass = dt_yr / n_passes` в строках 3181-3204 |

### 🟡 Высокие (фудж-факторы, баги, потеря физики)

| # | Задача | Сложность | Описание |
|---|--------|-----------|----------|
| P3 | H* = 1.0m + перекалибровка | Средняя | Строка 5478. Снизить до рекомендованного, подстроить V_s |
| P4 | Shelf: sediment infill model | Высокая | Строки 3617-3697. Заменить BFS override |
| P5 | spherical_fbm в evolve | Низкая | Строки 1431-1436. Заменить sin/cos |
| P6 | Hotspot lat convention | Низкая | Строки 2910-2913. Использовать CellCache |
| P7 | Valley carving 40x | Низкая | Строка 4837. Снизить с 80x до 40x |
| P8 | Chamfer fix | Низкая | Строки 4187/4204. Убрать (-1,1) из forward |
| P9 | Peneplain time-dependent | Средняя | Строка 3444. tau_denudation model |
| P10 | Coastline perturbation unification | Средняя | 3 системы -> 1-2 |
| G4 | Возраст литосферы -> K_eff | Средняя | Модулировать K_eff по тектонической активности |
| G8 | Priority-Flood pit-filling | Средняя | Заменить текущий lake detection |
| G9 | Sediment gravity routing | Высокая | Заменить weighted redistribution |

### ⚪ Низкие (неустранимо или незначимо)

- F1-F3, F5-F8: Plate growth heuristics — неустранимо без мантийной конвекции
- F15, F18: Safety valves — редко срабатывают
- E1, E2: Voronoi growth — архитектурно неустранимо
- E7: Hypsometric correction — conditional safety valve
- G6: Эвстазия — низкий приоритет

### Архитектурные задачи (не физика)

| Задача | Описание |
|--------|----------|
| Phase 1.2: Unified crop scope | Island blob fix — `fade_land_edges=true` уничтожает форму |

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
| Level 4.2 | Кратонные пенеплены | King 1967 |
| Level 4.3 | Океанические плато / LIPs | Coffin & Eldholm 1994 |
| Level 5.1 | Срединно-океанические хребты | Macdonald 1982; Stern 2002 |
| Level 5.2 | Эпейрогенический варпинг | Mitrovica 1989; Bond 1976 |
| Level 5.3 | Задуговые бассейны | Karig 1971; Sdrolias & Muller 2006 |
| Phase H13 | Climate-coupled runoff в SPACE | Roe 2003 |
| Phase H16 | Изостатическая разгрузка в B&W solver | Molnar & England 1990 |

---

## Файлы документации

| # | Файл | Тема |
|---|------|------|
| — | [AUDIT_REPORT](AUDIT_REPORT.md) | Полный научно-технический аудит (март 2026) |
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
Где физика упрощена или подобрана вручную — указано явно.
