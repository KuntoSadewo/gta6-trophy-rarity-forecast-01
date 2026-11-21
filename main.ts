// GTA VI trophy rarity forecaster (Node 22, standard library only).
//
// Reads one case file /input/<id>.in: Exophase earn rates for every GTA VI
// PS5 trophy, recorded up to the case checkpoint, and forecasts each trophy's
// earn rate at the case target as five quantiles (5/25/50/75/95%).
//
// Model, per trophy, in logit space:
//
//   * Earners and audience are modelled separately. Earners E = rate * players
//     tracked / 100 grow as players progress through the game; the tracked
//     audience P grows as Exophase picks up new players. Both are fitted as
//     straight lines of ln E and ln P against ln(1 + hours since release)
//     over the last 24 h of reads, then read at the target. The rate forecast
//     is E*/P*. A second fit, logit(rate) against ln(1 + hours), is blended
//     in as a cross-check, and the blend is shrunk toward the last observed
//     rate with a weight that grows with the horizon.
//   * Spread is the residual scatter of the logit fit plus a random-walk term
//     that grows with the square root of the horizon in days, so a far target
//     honestly admits a wider band than a near one.
//   * Stale page: when every read in the history is byte-identical (same
//     players tracked, same rates), Exophase has not refreshed since before
//     the first read, so the page the case shows is stale and a refresh may
//     land before the target. The forecast is then a two-component mixture:
//     no refresh (rate stays) versus a refresh that dilutes the rate as the
//     tracked audience jumps (the one refresh observed so far multiplied it
//     ~6x; newcomers earn an easy trophy quickly and a rare one barely at
//     all, so the diluted rate lets the newcomers' earn share grow with the
//     rate itself). Quantiles come from the mixture CDF, so the low
//     quantiles sit near the diluted rate while the high ones stay near the
//     last rate.
//
// Everything is deterministic: same input bytes, same output bytes. No
// network, no randomness, no clock reads. Output files are written, never
// deleted; the output directory itself is only created, never removed.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Release of GTA VI, the origin of the log-time axis. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
/** Levels scored, in the order each output row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantiles at the scored levels. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

interface Trophy {
  id: number;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
/** Hours from release to an ISO timestamp. */
const hoursSinceRelease = (iso: string): number =>
  Math.max(0, (Date.parse(iso) - RELEASE_MS) / 3600000);

/** Ordinary least squares of y on x; returns [intercept, slope]. */
function ols(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx <= 0) return [my, 0];
  const slope = sxy / sxx;
  return [my - slope * mx, slope];
}

/** Standard normal CDF, Abramowitz-Stegun 7.1.26 (absolute error < 1.5e-7). */
function normCdf(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp(-(x * x) / 2);
  const p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return x >= 0 ? 1 - p : p;
}

/**
 * Quantile of a two-component normal mixture in logit space, by bisection
 * on the mixture CDF. Components are (mean, sd, weight) triples.
 */
function mixtureQuantile(tau: number, comps: Array<[number, number, number]>): number {
  let lo = -14;
  let hi = 14;
  for (let iter = 0; iter < 80; iter++) {
    const mid = (lo + hi) / 2;
    let cdf = 0;
    for (const [m, s, w] of comps) cdf += w * normCdf((mid - m) / s);
    if (cdf < tau) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

/** One trophy's five quantiles, in percent, non-decreasing. */
function forecastTrophy(c: Case, idx: number): number[] {
  const last = c.history[c.history.length - 1];
  const rLast = clampRate(last.rates[idx] ?? 0);
  const pLast = Math.max(1, last.players_tracked);
  const horizonDays = c.horizon_minutes / 1440;
  const targetHours = hoursSinceRelease(c.at) + c.horizon_minutes / 60;
  const xTarget = Math.log1p(targetHours);

  // Is the whole page stale? Every read identical in players and rates.
  let stale = true;
  for (const h of c.history) {
    if (h.players_tracked !== last.players_tracked) { stale = false; break; }
    for (let j = 0; j < c.trophies.length; j++) {
      if ((h.rates[j] ?? -1) !== (last.rates[j] ?? -1)) { stale = false; break; }
    }
    if (!stale) break;
  }

  // A trophy nearly everyone holds cannot be diluted by a refresh: its
  // earners grow with the audience. Nothing to forecast either way.
  if (rLast >= 99.5) {
    const q = [99.5, 99.8, 99.9, 99.95, 99.99];
    return q.map(round4);
  }

  if (stale) {
    // Mixture: a refresh before the target with probability 1/2 dilutes
    // the rate as the tracked audience jumps. The one refresh observed so
    // far multiplied the audience ~6x, so the diluted component uses a 6x
    // jump. Newcomers earn an easy trophy quickly and a rare one barely at
    // all, so the share theta of newcomers who earn it by the target grows
    // with the rate itself (theta = sqrt(rate)): the diluted rate is
    // r * (1 + theta * 5) / 6. A trophy nearly everyone holds is handled
    // above and never reaches this branch.
    const theta = Math.sqrt(rLast / 100);
    const rDiluted = (rLast * (1 + theta * 5)) / 6;
    const lStay = logit(rLast);
    const lDiluted = logit(rDiluted);
    const sd = 0.3;
    const comps: Array<[number, number, number]> = [
      [lStay, sd, 0.5],
      [lDiluted, sd, 0.5],
    ];
    const qs = LEVELS.map((tau) => clampRate(invLogit(mixtureQuantile(tau, comps))));
    return qs.map(round4);
  }

  // Trend regime: fits over the last 24 h of reads (all reads if fewer).
  const checkpointMs = Date.parse(c.at);
  let rows = c.history.filter((h) => Date.parse(h.at) >= checkpointMs - 24 * 3600000);
  if (rows.length < 3) rows = c.history;
  const xs: number[] = [];
  const lnE: number[] = [];
  const lnP: number[] = [];
  const lg: number[] = [];
  for (const h of rows) {
    const r = h.rates[idx];
    if (r === null || r === undefined) continue;
    const p = Math.max(1, h.players_tracked);
    const e = Math.max(0.5, (Math.max(0, r) / 100) * p);
    const x = Math.log1p(hoursSinceRelease(h.at));
    xs.push(x);
    lnE.push(Math.log(e));
    lnP.push(Math.log(p));
    lg.push(logit(r));
  }
  // Too little usable history: persistence with a random-walk spread.
  if (xs.length < 3) {
    const s = 0.5 * Math.sqrt(horizonDays) + 0.1;
    const l = logit(rLast);
    return Z.map((z) => round4(clampRate(invLogit(l + z * s))));
  }

  const [bE, aE] = ols(xs, lnE);
  const [bP, aP] = ols(xs, lnP);
  const [bL, aL] = ols(xs, lg);
  const eLast = Math.max(0.5, (rLast / 100) * pLast);
  let eStar = Math.exp(bE + aE * xTarget);
  eStar = Math.min(Math.max(eStar, eLast * 0.95), eLast * 200);
  let pStar = Math.exp(bP + aP * xTarget);
  pStar = Math.min(Math.max(pStar, pLast * 0.9), pLast * 100);
  const rEp = clampRate((100 * eStar) / pStar);
  const rLl = clampRate(invLogit(bL + aL * xTarget));

  const wTrend = horizonDays / (horizonDays + 0.75);
  const trendLogit = 0.6 * logit(rEp) + 0.4 * logit(rLl);
  const medianLogit = wTrend * trendLogit + (1 - wTrend) * logit(rLast);

  // Residual scatter of the logit fit around its own line.
  let ss = 0;
  for (let i = 0; i < xs.length; i++) {
    const d = lg[i] - (bL + aL * xs[i]);
    ss += d * d;
  }
  const residSd = Math.sqrt(ss / Math.max(1, xs.length - 2));
  const s = Math.min(1.2, Math.sqrt(Math.max(residSd, 0.1) ** 2 + (0.3 * Math.sqrt(horizonDays)) ** 2));

  const qs = Z.map((z) => clampRate(invLogit(medianLogit + z * s)));
  // Enforce non-decreasing after rounding.
  const out = qs.map(round4);
  for (let i = 1; i < out.length; i++) if (out[i] < out[i - 1]) out[i] = out[i - 1];
  return out;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = c.trophies.map((t, i) => ({ id: t.id, q: forecastTrophy(c, i) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
