// Shared data for the "Transformers With a Velocity Ledger" panels (Arc D2).
// One memoized fetch per JSON export of scripts/export_velocity_viz.py (which
// only RESHAPES scripts/results/kgl_blog-velocity-ledger-v2/velocity_ledger.npz;
// no training). summary.json = the 4-way best-val tie + the depth-dynamics
// table + the loss curves; depth.json = the per-sub-update depth telemetry and
// the faithful 2D residual-stream trajectory (real segment lengths, real turns).
import { loadJSON } from './engine/io.js';

const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');
const DIR = `${BASE}/velocity-ledger`;

export const loadSummary = () => loadJSON(`${DIR}/summary.json`);
export const loadDepth = () => loadJSON(`${DIR}/depth.json`);

// stable variant order + colours across every panel in the post
export const ORDER = ['plain', 'ngpt_lite', 'ledger', 'ngpt_ledger'];
export const VCOL = {
  plain: '#c2553a',        // no ledger, first-order  (warm)
  ngpt_lite: '#9a4f9c',    // first-order on the sphere
  ledger: '#4a7fb3',       // the velocity ledger      (cool)
  ngpt_ledger: '#3a8f5e',  // ledger + retraction
};
export const VLABEL = {
  plain: 'plain', ngpt_lite: 'ngpt-lite',
  ledger: 'ledger', ngpt_ledger: 'ngpt-ledger',
};
