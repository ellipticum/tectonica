# 09. Worklog

## 2026-02-23

## Сделано

1. Создан модуль `docs/tectonics`.
2. Зафиксирована проблема «ломтиковых» плит как системный дефект геометрии.
3. Описана кинематика на полюсах Эйлера и классификация границ по `v_n/v_t`.
4. Зафиксирована физическая схема для конвергенции/дивергенции/трансформ/диффузных поясов.
5. Описана научная батиметрическая модель через `ocean_age` + age-depth law + boundary anomalies + sediment.
6. Добавлен набор primary references для калибровки.
7. Запущен первый этап внедрения в `rust/planet_engine/src/lib.rs`:
   - скорость плит переведена на эйлерову кинематику `v = (Ω × r) * R`,
   - классификация границ переведена на локальные `v_n/v_t`,
   - добавлены вторичные ядра роста плит для менее «ломтиковой» геометрии.
8. Добавлен этап «исторических ядер» (trajectory nuclei) вдоль направления движения плиты, чтобы форма плит отражала миграцию во времени, а не одношаговое разбиение.
9. Добавлена топологическая очистка мелких осколков плит:
   - абсолютный порог малых компонент,
   - относительный порог к крупнейшей компоненте каждой плиты.

## Наблюдения

1. Визуальные артефакты океана не лечатся фильтрами, пока нет `ocean_age`-модели.
2. «Ровные» горы и континенты — следствие статичной геометрии плит и тонких boundary-stroke uplift масок.
3. Для Earth-like результата нужно seed-ranking по метрикам, а не ручной перебор картинок.
4. После перехода на `Ω × r` нужна следующая фаза: эволюция контуров во времени (адвекция полигонов), иначе плиты все еще остаются кусочно-мозаичными.
5. Даже после очистки компонент остаются локальные микроплитные зоны; это ожидаемо для промежуточной схемы «field-based growth», но не является финальной физической моделью.

## Следующие шаги

1. Реализовать `plate_omega` и расчет `v_n/v_t` в Rust.
2. Перевести типизацию границ на кинематику.
3. Внедрить `ocean_age` и термальную батиметрию.
4. Подключить scoring из `06_DATASETS_AND_CALIBRATION.md`.
5. Реализовать time-step эволюцию `PlatePolygon(t)` и `BoundaryGraph(t)` вместо статичной генерации поля плит.

---

## 2026-02-24

## Сделано

1. Проведён полный аудит текущего кода тектоники (`compute_plates`, `build_irregular_plate_field`, `evolve_plate_field`, boundary detection).
2. Проведено научное исследование пробелов в текущей модели:
   - **Euler poles**: текущие omega генерируются случайно, без NNR constraint и slab-speed корреляции.
   - **Driving forces**: отсутствуют (slab pull, ridge push). Все плиты двигаются с ~одинаковой скоростью.
   - **Isostasy**: отсутствует. Высота рельефа не связана с толщиной коры физически.
   - **Ocean age**: отсутствует. Нет GDH1 age-depth relationship.
   - **Rheology-based deformation width**: ширина контролируется buoyancy_smooth_passes, не физикой.
   - **Subduction geometry**: нет slab dip, trench depth, arc-trench distance, back-arc physics.
3. Исследована проблема island scope:
   - `generate_tasmania_height_map()` использует эллипс+spine+noise — физически некорректна.
   - Результат: "бычий глаз" без тектонической структуры.
   - Решение: единый пайплайн через синтетический тектонический контекст + stream power erosion.
4. Создан блюпринт единой модели:
   - `11_UNIFIED_SCOPE_MODEL.md` — абстракция GridConfig, синтетический plate context для island.
   - `12_ISOSTASY_AND_CRUSTAL_PHYSICS.md` — Airy isostasy, flexural, GDH1, driving forces, NNR, slab dip.
   - `13_ISLAND_SCALE_TERRAIN.md` — stream power law, типы островов, Braun-Willett solver, литология, валидация.
5. Обновлён план внедрения `07_IMPLEMENTATION_PLAN.md` v2: добавлены фазы G-J.
6. Обновлён `00_INDEX.md` до v2 с новыми инвариантами.

## Ключевые научные источники, использованные в исследовании

- NNR-MORVEL56 (Argus et al. 2011): Euler poles и скорости 56 плит.
- GDH1 (Stein & Stein 1992): age-depth relationship океанической коры.
- Slab2 (Hayes et al. 2018): геометрия субдуцирующих слэбов.
- Braun & Willett (2013): O(n) implicit solver для stream power erosion.
- Cordonnier et al. (2016): terrain generation через uplift map + stream power.
- Tzathas et al. (2024): аналитические стационарные решения stream power.
- Parsons & Sclater (1977): классическая модель остывания океанической литосферы.
- Harel et al. (2016): глобальный анализ erodibility по типам пород.

## Наблюдения

1. Island scope требует принципиально другого подхода: вместо одной функции с noise+ellipse нужен полный физический пайплайн с тектоническим контекстом.
2. Stream power erosion (Braun-Willett) — ключевой компонент для реалистичных долин и дренажных сетей.
3. Изостатика — необходима для связи crust_thickness (уже есть) с elevation (сейчас эмпирическая).
4. Для continental fragment island'ов главный контролирующий фактор — horst-graben fault structure + lithology variation.
5. Береговая линия не должна задаваться аналитически (эллипс) — она должна возникать из пересечения рельефа с уровнем моря.

## Следующие шаги

1. Реализовать фазу H: GridConfig abstraction.
2. Реализовать фазу G: Airy isostasy.
3. Реализовать фазу I: generate_island_plate_context (начать с Continental type).
4. Реализовать фазу J: Braun-Willett stream power solver.
