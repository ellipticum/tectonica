"use client";

import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";

const clamp = (v: number, min: number, max: number) => (v < min ? min : v > max ? max : v);

export function NumericRange({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
}) {
  const [draft, setDraft] = useState(value);

  useEffect(() => {
    setDraft(value);
  }, [value]);

  const commit = (next: number) => {
    if (!Number.isFinite(next)) {
      return;
    }
    const clamped = clamp(next, min, max);
    setDraft(clamped);
    onChange(clamped);
  };

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <Label className="text-slate-300">{label}</Label>
        <span className="text-xs text-cyan-200">{draft.toFixed(step < 1 ? 2 : step < 10 ? 1 : 0)}</span>
      </div>
      <div className="grid gap-2 sm:grid-cols-[1fr_auto]">
        <Slider
          value={[draft]}
          min={min}
          max={max}
          step={step}
          onValueChange={(vals) => setDraft(vals[0] ?? draft)}
          onValueCommit={(vals) => commit(vals[0] ?? draft)}
        />
        <Input
          type="number"
          value={draft}
          min={min}
          max={max}
          step={step}
          onChange={(event) => commit(Number(event.target.value))}
          className="w-24"
        />
      </div>
    </div>
  );
}

export function LabeledInput({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="space-y-1.5">
      <Label className="text-slate-300">{label}</Label>
      <Input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}
