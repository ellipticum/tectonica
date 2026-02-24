# 13. Генерация Рельефа Островного Масштаба

## 1. Научная основа

Рельеф острова определяется балансом:

```
dh/dt = U(x,y) - E_fluvial(x,y) - E_hillslope(x,y) + D_sediment(x,y)
```

Где:
- `U` — тектонический подъём (mm/yr), зависит от типа острова.
- `E_fluvial` — речная эрозия (stream power law).
- `E_hillslope` — склоновые процессы (диффузия).
- `D_sediment` — депозиция осадков.

В стационарном состоянии `dh/dt = 0`, то есть `U = E`. Это означает, что **форма рельефа полностью определяется пространственным распределением U и K**.

## 2. Тектонические типы островов

### 2.1 Continental Fragment (Tasmania, Madagascar, New Zealand)

**Тектоническая история:**
- Часть континентальной коры, отделившаяся при рифтинге.
- Наследует сложную геологическую структуру: разные типы пород, старые разломы, интрузии.
- Характерная structural grain — NNW-тренд разломов (Тасмания), определяющий horst-graben topography.

**Поле подъёма U(x,y):**
```
U_continental(x,y) = U_base
    + U_horst * horst_mask(x,y)         // +0.2-0.5 mm/yr на поднятых блоках
    - U_graben * graben_mask(x,y)        // -0.1-0.3 mm/yr в грабенах
    + U_asym * asymmetry_factor(x)       // east-west asymmetry
    + noise_low * 0.05                   // stochastic variation
```

**Horst-graben структура:**
```rust
fn horst_graben_mask(
    x_km: f32, y_km: f32,
    fault_angle_rad: f32,    // ~NNW = -20 degrees from N
    fault_spacing_km: f32,   // 40-80 km between faults
    fault_width_km: f32,     // 2-5 km fault zone width
    seed: u32,
) -> f32 {
    // Project coordinates onto fault-perpendicular axis
    let perp = x_km * fault_angle_rad.cos() + y_km * fault_angle_rad.sin();

    // Alternating horst (+1) and graben (-1) with noise
    let phase = perp / fault_spacing_km + noise(x_km, y_km, seed) * 0.3;
    let block = (phase * PI).sin();  // smooth alternation

    // Sharp fault zones
    let fault_proximity = (phase.fract() * fault_spacing_km).abs();
    let fault_factor = smoothstep(fault_width_km, 0.0, fault_proximity);

    block * (1.0 - fault_factor * 0.5)  // weaken near faults
}
```

**Литология K(x,y):**
```
Dolerite cap (resistant):   K = K_base * 0.3    // 30% of island surface
Precambrian metamorphic:    K = K_base * 0.5
Sedimentary basins:         K = K_base * 2.0-3.0
Volcanic/basalt:            K = K_base * 0.7
```

**Характерные параметры Тасмании:**

| Параметр | Значение |
|----------|----------|
| Площадь | ~68,000 км2 |
| Макс. высота | 1617 м (Mt Ossa) |
| Средняя высота | ~220 м |
| Structural grain | NNW (-20° от N) |
| Uplift rate | 0.1-0.5 mm/yr (историческое) |
| Dominant rock | Jurassic dolerite sills |
| Fractal D (coastline) | ~1.25-1.35 |
| Hypsometric integral | ~0.35 |

### 2.2 Island Arc (Japan, Philippines)

**Поле подъёма U(x,y):**
```
U_arc(x,y) = U_arc_max * exp(-d_perp^2 / (2*sigma_arc^2))
           + U_forearc * exp(-d_forearc^2 / (2*sigma_forearc^2))
           - U_trench * exp(-d_trench^2 / (2*sigma_trench^2))
           - U_backarc * exp(-d_backarc^2 / (2*sigma_backarc^2))
```

Где `d_perp` — расстояние от оси дуги; `sigma_arc ~ 30-50 km`.

| Параметр | Значение |
|----------|----------|
| U_arc_max | 2-5 mm/yr |
| Arc width | 50-100 км |
| Trench depth | 7-11 км |
| Arc-trench distance | 100-300 км |
| Volcano spacing | 30-70 км |
| Макс. высота | 2000-3776 м (Fuji) |

### 2.3 Hotspot Volcanic (Hawaii, Reunion)

**Поле подъёма U(x,y):**
```
U_hotspot(x,y) = U_max * exp(-r^2 / (2*sigma^2))
               - U_subsidence                    // general plate subsidence
```

| Параметр | Значение |
|----------|----------|
| U_max (active) | 0.2-0.5 mm/yr (long-term), up to 900 mm/yr (eruption) |
| Subsidence | 1.7-4.8 mm/yr near active center |
| sigma | island_radius / 2 |
| Shape | Near-circular shield |
| Profile | Convex (young), concave (old) |
| Fractal D | ~1.1-1.2 (simple shield) |

### 2.4 Rift Island (Corsica, Sardinia)

**Поле подъёма U(x,y):**
```
U_rift(x,y) = U_shoulder * exp(-d_rift / lambda_rift)  // exponential decay from rift
            - U_subsidence * (1 - exp(-d_rift / lambda_sub))  // subsidence toward rift
```

| Параметр | Значение |
|----------|----------|
| U_shoulder | 0.5-1.0 mm/yr |
| lambda_rift | 50-100 км |
| Asymmetry | Strong: steep on rift side, gentle on other |
| Грабен | На rift-facing стороне |

## 3. Stream Power Erosion

### 3.1 Уравнение

```
E_fluvial = K_eff(x,y) * A(x,y)^m * S(x,y)^n
```

Стандартные значения: `m = 0.5`, `n = 1.0` (unit stream power model).

### 3.2 Effective erodibility

```
K_eff(x,y) = K_lithology(x,y) * K_climate(x,y)
```

Где:
- `K_lithology` — из карты горных пород (0.3-3.0 × K_base).
- `K_climate` — модуляция осадками: `K_climate = (precipitation / P_ref)^0.7`.

**Климатическая асимметрия:**
- Наветренная сторона: precipitation_factor = 2-4.
- Подветренная сторона: precipitation_factor = 0.5-1.0.
- Это создаёт характерную асимметричную эрозию (глубокие долины на wet side).

### 3.3 Стандартные значения K

Из глобального анализа (Harel et al., 2016):

| Порода | K (m^0.5/yr) | Примечание |
|--------|--------------|------------|
| Massive basalt / rhyolite | ~3 × 10^-6 | Самая устойчивая |
| Metamorphic | ~5 × 10^-6 | |
| Mixed volcanic | ~1 × 10^-5 | |
| Sandstone / conglomerate | ~3 × 10^-5 | |
| Shale / mudstone | ~1 × 10^-4 | Наименее устойчивая |

Диапазон: ~75× между самой крепкой и самой слабой породой.

## 4. Алгоритм Braun-Willett (O(n) implicit solver)

### 4.1 Почему именно он

1. **O(n)** — линейная сложность по числу ячеек.
2. **Implicit** — безусловно устойчив, допускает большие dt.
3. **Детерминистичен** при фиксированном seed.

### 4.2 Алгоритм

```
Для каждого временного шага dt:

1. Вычислить flow directions (D8: стейпест descent к одному из 8 соседей).
2. Отсортировать ячейки по убыванию высоты (topological order).
3. Вычислить drainage area A: пройти от верхних ячеек к нижним, накапливая площадь.
4. Обновить высоты от нижних ячеек к верхним (implicit scheme):

   h_new[i] = (h_old[i] + U[i]*dt + K*dt*(A[i])^m * h_new[receiver[i]] / dx^n)
            / (1 + K*dt*(A[i])^m / dx^n)

   Где receiver[i] — ячейка, куда стекает i.
   Решается за один проход, т.к. receiver[i] уже обновлён.

5. Добавить hillslope diffusion:
   h[i] += kappa * dt * laplacian(h)[i]
```

### 4.3 Реализация в Rust

```rust
fn stream_power_step(
    height: &mut [f32],
    uplift: &[f32],         // U(x,y) в m/yr
    k_eff: &[f32],          // K_eff(x,y) в m^0.5/yr
    dt_yr: f32,             // timestep в годах
    dx_m: f32,              // размер ячейки в метрах
    m: f32,                 // area exponent (0.5)
    n: f32,                 // slope exponent (1.0)
    kappa: f32,             // hillslope diffusivity m2/yr
    grid: &GridConfig,
) {
    let size = grid.width * grid.height;

    // 1. Flow directions (D8)
    let receivers = compute_d8_receivers(height, grid);

    // 2. Topological sort (by descending height)
    let order = topological_sort_descending(height, &receivers);

    // 3. Drainage area (upstream -> downstream)
    let mut area = vec![dx_m * dx_m; size];  // each cell contributes its own area
    for &i in order.iter() {
        let r = receivers[i];
        if r != i {  // not a sink
            area[r] += area[i];
        }
    }

    // 4. Implicit elevation update (downstream -> upstream)
    for &i in order.iter().rev() {
        let r = receivers[i];
        if r == i { continue; }  // sink: no erosion

        let k = k_eff[i];
        let a_pow = area[i].powf(m);
        let factor = k * dt_yr * a_pow / dx_m.powf(n);

        height[i] = (height[i] + uplift[i] * dt_yr + factor * height[r])
                   / (1.0 + factor);
    }

    // 5. Hillslope diffusion
    if kappa > 0.0 {
        apply_hillslope_diffusion(height, kappa, dt_yr, dx_m, grid);
    }
}
```

### 4.4 Аналитическое стационарное решение (для fast preset)

При `dh/dt = 0`:

```
S = (U / (K * A^m))^(1/n)
```

Можно вычислить высоту каждой ячейки за один проход (от устья к истокам):

```
h[i] = h[receiver[i]] + S[i] * distance_to_receiver[i]
```

Где `S[i] = (U[i] / (K[i] * A[i]^m))^(1/n)`.

Это даёт мгновенный стационарный рельеф без итерирования. Подходит для "ultra" preset.

## 5. Coastline Generation

### 5.1 Проблема

Текущий island scope использует деформированный эллипс. Нужна береговая линия, которая:
1. Отражает тектоническую структуру (fault-controlled bays).
2. Имеет правильную фрактальную размерность (1.1-1.4 в зависимости от типа).
3. Формируется естественно из баланса uplift/erosion/sea level.

### 5.2 Подход

Не рисовать береговую линию вручную. Вместо этого:

1. Задать начальную высоту как `isostatic_elevation(crust_thickness)`.
2. Запустить stream power erosion.
3. Установить sea level.
4. Береговая линия = `{(x,y) : h(x,y) = sea_level}`.

Форма береговой линии возникает АВТОМАТИЧЕСКИ из:
- Паттерна uplift (horst/graben → зубчатая береговая линия с полуостровами).
- Литологии (resistant rock = мысы, weak rock = бухты).
- Дренажной сети (речные долины → эстуарии и фьорды при подъёме уровня моря).

### 5.3 Контроль размера острова

Чтобы остров заполнял ~30-60% grid:

```
sea_level = percentile(height_field, target_ocean_fraction)
```

Или через контроль средней высоты:

```
mean_uplift * sim_time ≈ desired_max_elevation * 1.5
```

## 6. Закон Hack'а и валидация дренажной сети

### 6.1 Уравнение

```
L = c * A^h
```

Где:
- `L` — длина главного русла (км).
- `A` — площадь бассейна (км2).
- `c ≈ 1.4`
- `h ≈ 0.6` (диапазон 0.5-0.7).

### 6.2 Метрика качества

После генерации извлечь бассейны, измерить L и A, проверить:
- `h` в диапазоне 0.5-0.7.
- Нет бассейнов-outlier'ов (слишком круглых или слишком вытянутых).

## 7. Гипсометрия и валидация

### 7.1 Hypsometric Integral (HI)

```
HI = (mean_elevation - min_elevation) / (max_elevation - min_elevation)
```

| Тип острова | Ожидаемый HI |
|-------------|-------------|
| Young volcanic (shield) | > 0.5 |
| Mature volcanic (eroded) | 0.3-0.5 |
| Continental fragment | 0.3-0.5 |
| Island arc (active) | 0.4-0.6 |

### 7.2 Hypsometric curve

Для continental fragment кривая должна быть S-образной с:
- Широким плато в нижней части (lowlands, ~60% площади ниже 200 м).
- Крутым переходом.
- Узким пиком (highlands, ~5% площади выше 1200 м).

## 8. Масштабные пресеты

| Пресет | Масштаб | Grid | dx (м) | Sim time | dt | Stream power steps | Пример |
|--------|---------|------|---------|----------|----|-------------------|--------|
| Small island | 50 км | 512x512 | 100 | 5 Myr | 1000 yr | 5000 | Hawaii-like |
| Medium island | 200 км | 1024x512 | 200 | 20 Myr | 2000 yr | 10000 | Corsica-like |
| Large island | 500 км | 1024x512 | 500 | 50 Myr | 5000 yr | 10000 | Tasmania-like |
| Continental chunk | 1000 км | 2048x1024 | 500 | 100 Myr | 10000 yr | 10000 | New Zealand-like |

**Detail presets scale steps:**

| Preset | Steps multiplier | Erosion rounds |
|--------|-----------------|----------------|
| Ultra | 0.1x | 500-1000 |
| Fast | 0.3x | 1500-3000 |
| Balanced | 1.0x | 5000-10000 |
| Detailed | 2.0x | 10000-20000 |

## 9. Связь с planet scope

### 9.1 Crop mode (будущее)

Вместо синтетического контекста можно:
1. Сгенерировать полную планету.
2. Найти остров подходящего размера.
3. Вырезать region + boundary padding.
4. Перегенерировать на мелкой сетке с тем же uplift field.

Это даёт максимальную физическую согласованность, но дорого.

### 9.2 Единые формулы

Ключевое: формулы climate, biomes, settlement **одинаковы** для planet и island scope. Разница только в:
- Источнике тектонического контекста (full plates vs synthetic).
- Масштабе сетки и числе шагов.
- Используемой координатной системе (sphere vs plane).

## 10. Что удаляется из текущего кода

1. `generate_tasmania_height_map()` — 170 строк hardcoded эллипс+spine.
2. `apply_events_island()` — заменяется общим event handler.
3. `run_tasmania_scope()` — заменяется `run_island_scope()` с unified pipeline.
4. Все `ISLAND_WIDTH/HEIGHT/SIZE` константы — заменяются динамическими параметрами.
5. Термин "tasmania" из кода и UI.
