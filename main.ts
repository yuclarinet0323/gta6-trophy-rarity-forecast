// GTA VI trophy earn-rate forecaster for make.fun (gta6-trophy-rarity-forecast-01).
//
// Idea: an earn rate is a ratio, earners / players tracked, and the two sides
// move for different reasons in launch week. The tracked-player count jumps
// in steps every time Exophase refreshes its counts, diluting every rate at
// once, while each trophy's earner count only grows as players reach it.
// So this program, per case:
//   1. collapses the history to the distinct snapshots (reads between two
//      refreshes repeat the same page and carry no new information),
//   2. fits a pooled trend for the player count and a per-trophy trend for
//      earner counts, both against ln(1 + hours since release),
//   3. fits a direct per-trophy trend of the logit rate as a second view,
//   4. blends the two views with the last rate as a small anchor, and
//   5. spreads the five quantiles by the baseline random-walk width plus
//      the fit residuals and the disagreement between the two views.
// Everything is deterministic, standard library only, and offline: all the
// information it uses is inside the case file it is given.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

interface TrophyInfo {
  id: number;
  tier?: string;
}
interface ReadRow {
  slot: string;
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface CaseFile {
  levels: number[];
  horizon_minutes: number;
  target: string;
  game: { release_date: string };
  trophies: TrophyInfo[];
  history: ReadRow[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logitOf = (pct: number): number => {
  const p = clampPct(pct);
  return Math.log(p / (100 - p));
};
const pctOf = (logit: number): number => 100 / (1 + Math.exp(-logit));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));

// Weighted least squares y ~ a + b*x. Returns the fit and the weighted
// residual standard deviation, which later widens the quantile spread.
interface Fit { a: number; b: number; resid: number }
function fitLine(xs: number[], ys: number[], ws: number[]): Fit {
  let sw = 0, sx = 0, sy = 0;
  for (let i = 0; i < xs.length; i++) { sw += ws[i]; sx += ws[i] * xs[i]; sy += ws[i] * ys[i]; }
  const mx = sx / sw, my = sy / sw;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - mx;
    sxx += ws[i] * dx * dx;
    sxy += ws[i] * dx * (ys[i] - my);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = my - b * mx;
  let sr = 0;
  for (let i = 0; i < xs.length; i++) { const e = ys[i] - (a + b * xs[i]); sr += ws[i] * e * e; }
  return { a, b, resid: Math.sqrt(sr / sw) };
}

function forecastCase(c: CaseFile, fileName: string): { trophies: Array<{ id: number; q: number[] }> } {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${fileName} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (!c.history || c.history.length === 0) throw new Error(`${fileName} has no history`);

  const days = c.horizon_minutes / 1440;
  const baseSpread = 0.5 * Math.sqrt(days);
  const last = c.history[c.history.length - 1];

  // Collapse to distinct snapshots: a new snapshot starts when the player
  // count or any rate changes against the previously kept read.
  const snaps: ReadRow[] = [];
  for (const row of c.history) {
    const prev = snaps[snaps.length - 1];
    if (!prev) { snaps.push(row); continue; }
    if (prev.players_tracked !== row.players_tracked) { snaps.push(row); continue; }
    let changed = false;
    for (let i = 0; i < row.rates.length; i++) {
      if (prev.rates[i] !== row.rates[i]) { changed = true; break; }
    }
    if (changed) snaps.push(row);
  }

  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const hoursOf = (iso: string): number => Math.max(0, (Date.parse(iso) - releaseMs) / 3600000);
  const xOf = (iso: string): number => Math.log(1 + hoursOf(iso));
  const xs = snaps.map((s) => xOf(s.slot || s.at));
  const xTarget = xOf(c.target);
  const xLast = xs[xs.length - 1];
  // Recency weights: a snapshot from two days ago says less about the
  // target week than yesterday's, but it still pins down the trend.
  const ws = snaps.map((s, j) => 0.2 + Math.exp(-(xLast - xs[j]) / 1.5));

  // Pooled player-count trend, shared by every trophy in the case.
  // Predictions are anchored at the last snapshot: the fit supplies the
  // slope, the last read supplies the level, so a level error in the fit
  // cannot shift every forecast at once.
  const dx = Math.max(0, xTarget - xLast);
  let nTarget = last.players_tracked;
  let nResid = 0;
  if (snaps.length >= 3) {
    const logN = snaps.map((s) => Math.log(Math.max(1, s.players_tracked)));
    const fitN = fitLine(xs, logN, ws);
    const slopeN = clamp(fitN.b, 0, 5);
    const logNT = logN[logN.length - 1] + slopeN * dx;
    nTarget = clamp(Math.exp(logNT), last.players_tracked, last.players_tracked * 30);
    nResid = fitN.resid;
  }

  // First pass: fit both views for every trophy and collect the earner
  // slopes, so the second pass can shrink each trophy's earner slope a
  // little toward the field's median slope. Tiny rates are printed to 2
  // decimals, so a single trophy's slope is mostly rounding noise; the
  // field's typical growth is a steadier reference.
  interface TrophyFit {
    lastLogit: number;
    predA: number;
    slopeB: number;
    lastLogE: number;
    residA: number;
  }
  const fits: TrophyFit[] = c.trophies.map((trophy, i) => {
    const lastLogit = logitOf(last.rates[i] ?? 0);
    if (snaps.length < 3) return { lastLogit, predA: lastLogit, slopeB: 0, lastLogE: 0, residA: 0 };
    const logits = snaps.map((s) => logitOf(s.rates[i] ?? 0));
    const fitA = fitLine(xs, logits, ws);
    const slopeA = clamp(fitA.b, -4, 4);
    const logE = snaps.map((s) => {
      const earners = ((s.rates[i] ?? 0) * Math.max(1, s.players_tracked)) / 100;
      return Math.log(earners + 0.5);
    });
    const fitB = fitLine(xs, logE, ws);
    return {
      lastLogit,
      predA: lastLogit + slopeA * dx,
      slopeB: clamp(fitB.b, 0, 8),
      lastLogE: logE[logE.length - 1],
      residA: fitA.resid,
    };
  });
  const sortedSlopes = fits.map((f) => f.slopeB).sort((a, b) => a - b);
  const medianSlopeB = sortedSlopes.length ? sortedSlopes[Math.floor(sortedSlopes.length / 2)] : 0;

  const rows = c.trophies.map((trophy, i) => {
    const fit = fits[i];
    const lastLogit = fit.lastLogit;

    // With too few distinct snapshots there is no trend to read: fall
    // back to persistence around the last rate.
    if (snaps.length < 3) {
      const spread = baseSpread;
      return { id: trophy.id, q: Z.map((z) => round4(pctOf(lastLogit + z * spread))) };
    }

    // View B: earners over players, with the shrunk earner slope.
    const slopeB = 0.75 * fit.slopeB + 0.25 * medianSlopeB;
    const eTarget = Math.max(0, Math.exp(fit.lastLogE + slopeB * dx) - 0.5);
    const rateB = clampPct((100 * Math.min(eTarget, nTarget)) / Math.max(1, nTarget));
    const predB = logitOf(rateB);
    const predA = fit.predA;

    // Blend the two views, then damp the total move from the last rate:
    // a saturating process extrapolated in a straight line overshoots,
    // and 0.7 keeps most of a real trend while trimming that overshoot.
    // The move is also capped, tighter upward for trophies that are
    // already common (their remaining headroom is mostly players who
    // will never earn them) and looser for rare ones, whose logit can
    // legitimately swing a long way in a week of dilution and recovery.
    const blended = 0.5 * predA + 0.5 * predB;
    const lastRatePct = pctOf(lastLogit);
    const upCap = lastRatePct > 20 ? 1.4 : 2.2;
    const move = clamp(0.7 * (blended - lastLogit), -2.2, upCap);
    const medianLogit = lastLogit + move;

    // Spread: random-walk width for the horizon, plus how badly the
    // trends fit this trophy, plus how far the two views disagree.
    const disagree = Math.abs(predA - predB) / 2;
    const residPart = Math.sqrt(fit.residA * fit.residA + nResid * nResid) / 2;
    let spread = Math.sqrt(baseSpread * baseSpread + residPart * residPart + disagree * disagree);
    spread = clamp(spread, 0.15, 1.8);

    return { id: trophy.id, q: Z.map((z) => round4(pctOf(medianLogit + z * spread))) };
  });

  return { trophies: rows };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const parsed = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const out = forecastCase(parsed, file);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
