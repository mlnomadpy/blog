// Shared data module for the "White-Box Survival Model on Trial" panels.
// One memoized load per exported JSON in public/survival-trial/. Every number
// originates in scripts/deepsurv_trial.py and is reshaped by
// scripts/export_trial_viz.py; the panels only re-slice it, never invent it.

import { loadJSON } from './engine/io.js';

const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');
const DIR = `${BASE}/survival-trial`;

let _forest = null, _ablation = null, _calib = null, _ood = null, _plaus = null;

export const loadForest = () => (_forest ||= loadJSON(`${DIR}/forest.json`));
export const loadAblation = () => (_ablation ||= loadJSON(`${DIR}/ablation.json`));
export const loadCalib = () => (_calib ||= loadJSON(`${DIR}/calibration.json`));
export const loadOod = () => (_ood ||= loadJSON(`${DIR}/ood.json`));
export const loadPlaus = () => (_plaus ||= loadJSON(`${DIR}/plausibility.json`));

// pretty dataset labels shared by several panels
export const DS_LABEL = {
  metabric: 'METABRIC', support: 'SUPPORT', gbsg: 'GBSG',
  whas500: 'WHAS500', flchain: 'FLCHAIN',
};
