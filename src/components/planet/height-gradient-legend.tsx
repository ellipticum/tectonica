"use client";

import { buildHeightGradient, formatNum } from "@/lib/planet/layer-legend";

const clamp = (v: number, min: number, max: number) => (v < min ? min : v > max ? max : v);

export function HeightGradientLegend({
  minHeight,
  maxHeight,
}: {
  minHeight: number;
  maxHeight: number;
}) {
  const seaSplit = minHeight < 0 && maxHeight > 0
    ? clamp((0 - minHeight) / Math.max(1, maxHeight - minHeight), 0, 1)
    : null;

  return (
    <div className="mt-2 space-y-2">
      <div className="relative h-7 overflow-hidden rounded-md border border-white/15">
        <div className="absolute inset-0" style={{ background: buildHeightGradient(minHeight, maxHeight) }} />
        {seaSplit !== null ? (
          <div
            className="absolute inset-y-0 w-px bg-white/75"
            style={{ left: `${(seaSplit * 100).toFixed(2)}%` }}
            title="Уровень моря"
          />
        ) : null}
      </div>
      <div className="flex items-center justify-between text-[11px] text-slate-300">
        <span>{formatNum(minHeight)} м</span>
        <span>0 м</span>
        <span>{formatNum(maxHeight)} м</span>
      </div>
      <p className="text-[11px] text-slate-400">
        Океан: светло-голубой → глубокий кобальтовый, суша: зелёный → жёлто-охристый → красно-коричневый → серо-белый.
      </p>
    </div>
  );
}
