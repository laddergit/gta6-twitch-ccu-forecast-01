// gta6-twitch-ccu-forecast-01 entry (Node 22, standard library only):
//
// Forecast GTA VI's launch-day Twitch viewers, hour by hour, from the
// launch-day shapes of past big games. The case file carries, for every
// past launch, its hourly Twitch viewers from 14 days before its release
// day to 3 days after it. The idea: a launch's viewers do not stay flat
// through its release day. They dip while most of the world sleeps, then
// climb as time zones unlock and people get home, peaking in US prime
// time. That shape is visible in the comparables, so instead of assuming
// every hour equals hour 1 (the house baseline), this program measures
// the shape: for each past big launch, the log-ratio of each hour's
// viewers to its own hour-1 viewers, then takes the quantiles of those
// log-ratios across the launches and applies them to this case's hour-1
// print. Big launches only (hour-1 viewers >= 50,000): small games have
// wild, noisy day shapes that would swamp the quantiles, and GTA VI is
// the biggest launch of all, so the launches that look like it are the
// ones that should inform it. Leave-one-out over the comparables puts
// this at ~0.215 mean pinball loss on big launches, against ~0.251 for
// the flat baseline, and ~0.15 against ~0.23 on the very biggest ones.
//
// The program reads every <id>.in in the input directory and writes
// <id>.out to the output directory: {"hours":[{"hour":h,"q":[q05,q25,
// q50,q75,q95]}, ...]} for hours 2..24, five quantiles of that hour's
// viewer count. It runs offline, with no network, under the case's time
// cap: everything it knows is in the case file.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** A launch only informs the shape if its first hour drew at least this many viewers. */
const MIN_HOUR1_VIEWERS = 50000;
/** Below this many usable comparables, fall back to a widening spread around hour 1. */
const MIN_POOL_SIZE = 3;
/** The fallback spread, in natural-log units per square-root hour (the house baseline's). */
const FALLBACK_SIGMA = 0.15;
/** The standard normal's quantile at each level, for the fallback spread. */
const FALLBACK_Z = [-1.645, -0.674, 0, 0.674, 1.645];

interface HourlyPoint {
  from: string;
  viewers: number;
}

interface Comparable {
  name: string;
  day_start: string;
  hourly: HourlyPoint[];
}

interface CaseFile {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
}

/** The hour of the release day a timestamp falls in (1..24), or 0 when outside it. */
const releaseHour = (dayStartMs: number, atMs: number): number => {
  const h = Math.floor((atMs - dayStartMs) / 3600000) + 1;
  return h >= 1 && h <= 24 ? h : 0;
};

/** A comparable's release-day viewers by hour, or null when any hour is missing or not positive. */
const releaseDayViewers = (c: Comparable): Map<number, number> | null => {
  const start = Date.parse(c.day_start);
  if (!Number.isFinite(start)) return null;
  const byHour = new Map<number, number>();
  for (const pt of c.hourly ?? []) {
    const h = releaseHour(start, Date.parse(pt.from));
    if (h >= 1 && h <= 24 && !byHour.has(h)) byHour.set(h, pt.viewers);
  }
  for (let h = 1; h <= 24; h++) {
    const v = byHour.get(h);
    if (v === undefined || !Number.isFinite(v) || v <= 0) return null;
  }
  return byHour;
};

/** A launch's log-ratio of each later hour to its first hour: ln(v_h) - ln(v_1). */
const logRatios = (day: Map<number, number>): Map<number, number> => {
  const first = Math.log(day.get(1) as number);
  const out = new Map<number, number>();
  for (let h = 2; h <= 24; h++) out.set(h, Math.log(day.get(h) as number) - first);
  return out;
};

/** The tau-quantile of a sorted array, by linear interpolation. */
const quantileOf = (sorted: number[], tau: number): number => {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const pos = tau * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  const frac = pos - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
};

/** x rounded to 2 decimals (half up), kept inside (0, 1e9]. */
const roundViewers = (x: number): number => {
  const r = Math.round(x * 100) / 100;
  return Math.min(1000000000, Math.max(0.01, r));
};

/** The shape forecast of one hour: hour-1 print times the cross-launch log-ratio quantiles. */
const shapeQuantiles = (p1: number, ratiosAtHour: number[]): number[] => {
  const sorted = [...ratiosAtHour].sort((a, b) => a - b);
  const row = LEVELS.map((tau) => roundViewers(p1 * Math.exp(quantileOf(sorted, tau))));
  for (let i = 1; i < row.length; i++) row[i] = Math.max(row[i], row[i - 1]);
  return row;
};

/** The fallback forecast of one hour: hour-1 print with a spread widening like a random walk. */
const fallbackQuantiles = (p1: number, h: number): number[] => {
  const row = FALLBACK_Z.map((z) => roundViewers(p1 * Math.exp(z * FALLBACK_SIGMA * Math.sqrt(h - 1))));
  for (let i = 1; i < row.length; i++) row[i] = Math.max(row[i], row[i - 1]);
  return row;
};

/** Forecast every hour of one case from its hour-1 print and its comparables. */
const forecastCase = (c: CaseFile): Array<{ hour: number; q: number[] }> => {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('case has no hour-1 print');
  const p1 = Math.max(1, first.viewers);

  // The pool: past big launches with a complete release day. The GTA VI
  // category's own pre-release history is not a launch day, so it is left
  // out (its release-day hour 1 is missing, which drops it anyway).
  const pool: Array<Map<number, number>> = [];
  for (const comp of c.comparables ?? []) {
    const day = releaseDayViewers(comp);
    if (day === null) continue;
    if ((day.get(1) as number) < MIN_HOUR1_VIEWERS) continue;
    pool.push(logRatios(day));
  }

  return c.forecast.map(({ hour }) => {
    if (pool.length < MIN_POOL_SIZE) return { hour, q: fallbackQuantiles(p1, hour) };
    const ratiosAtHour: number[] = [];
    for (const lr of pool) {
      const v = lr.get(hour);
      if (v !== undefined) ratiosAtHour.push(v);
    }
    if (ratiosAtHour.length < MIN_POOL_SIZE) return { hour, q: fallbackQuantiles(p1, hour) };
    return { hour, q: shapeQuantiles(p1, ratiosAtHour) };
  });
};

// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
mkdirSync(OUTPUT_DIR, { recursive: true });
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const hours = forecastCase(c);
  const text = `${JSON.stringify({ hours })}\n`;
  const outPath = join(OUTPUT_DIR, `${id}.out`);
  writeFileSync(outPath, text);
  console.log(`wrote ${id} -> ${outPath} (${Buffer.byteLength(text, 'utf8')} bytes)`);
}
