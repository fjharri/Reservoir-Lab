# Reservoir Lab — client-side recurrent learning experiments

Reservoir Lab contains interactive browser experiments about recurrent learning under constrained compute. All model training and inference run on the visitor's device in Web Workers. Firebase Hosting only serves static files. No database, Firebase SDK, API key or server-side ML is needed.

- **01 · Hydraulic comparison:** a fixed echo-state network (ESN) and a trainable gated recurrent unit (GRU) classify hydraulic valve condition under measured training budgets.
- **02 · Adaptive double pendulum:** an explicitly engineered parameter-switch stress test where a sparse ESN adapts online, an identical frozen readout provides the no-learning counterfactual, and a monotonic scheduler reports the achieved rate, late ticks, dropped observations, and response deadlines.

## Run locally

Requires Node.js 18 or newer. No installation step or build is needed to run the demo.

```sh
npm run serve
```

Open `http://localhost:8765`. On a phone connected to the same Wi-Fi, open `http://YOUR_COMPUTER_LAN_IP:8765`. The computer's firewall must allow the local server. Opening `index.html` directly as a file will not work: workers and data loading need an HTTP server.

## Put it on Firebase

Create or choose a Firebase project with Hosting enabled. The included `firebase.json` deploys only `public/`. From this folder, with the Firebase CLI installed:

```sh
firebase login
firebase deploy --only hosting --project YOUR_FIREBASE_PROJECT_ID
```

Use the URL Firebase prints as the QR-code destination for the presentation. No Firebase deployment has been performed in this package. Follow the [official Hosting quickstart](https://firebase.google.com/docs/hosting/quickstart) for project setup. The app has no client-side dependency on Google services once its files have loaded.

## Experiment protocol

Each browser run produces three GRU checkpoints. First, it saves the last complete GRU update within the ESN's measured active training time, including GRU model setup. A complete batch may cross the deadline; that batch is excluded from the equal-budget checkpoint. If setup itself exceeds the budget, the result reports no eligible checkpoint.

The second checkpoint is the first evaluated GRU model that reaches the ESN's validation accuracy minus the selected tolerance. Training then continues with validation-based early stopping. After the target has been reached, six consecutive validation checks without a new best accuracy stop training. The third, finalised checkpoint restores the highest-validation-accuracy model observed during the whole run. The selected training cap remains a safety limit; if the GRU never reaches the target, it trains to that cap and the best observed validation checkpoint is retained.

Test accuracy is reported only after training and checkpoint selection finish. The learning chart uses validation accuracy, not test accuracy. Download the JSON for the complete configuration, split hashes, timing components, checkpoint curve, confusion matrices, environment and protocol notes. CSV contains the GRU learning curve.

Stop terminates the worker immediately. Each new run starts from fresh weights and recomputes reservoir features. There is no concealed reuse of training results.

## Models and clocks

| Detail | ESN | GRU |
|---|---|---|
| Default width | 500 fixed neurons | 16 trainable units |
| Recurrence | Ring connection plus two random links per neuron | TensorFlow.js GRU |
| Initialization | Seeded uniform input weights/biases, signed recurrent weights | Seeded Glorot input, recurrent and output weights |
| Sequence representation | Mean hidden state over 120 steps | Mean hidden state over 120 steps |
| State between cycles | Reset to zero | Independent sequences |
| Readout | Ridge to four one-hot targets, argmax scores | Four-way softmax |
| Training | Train-only feature standardisation and primal/dual ridge solve | Adam, batch size 8, default learning rate 0.01 |
| Default fitted weights | 2,004 (plus fitted feature scaling statistics) | 1,172 |
| Training clock | Train reservoir extraction, feature scaling and readout fit | Model setup, tensor preparation and gradient updates |
| Evaluation clock | Validation/test features and predictions | Validation/test predictions |

Reservoir leak is 0.5, input weights are uniform in [-0.7, 0.7], and biases in [-0.2, 0.2]. The absolute recurrent row sum is at most 0.9: this bounds the recurrent operator and gives a contractive update for this tanh construction. It is not a measured spectral radius. Ridge penalty is `n × λ` in the Gram system; default λ = 0.005. Feature scaling uses training feature means and standard deviations, floored at 1e-6. Inputs use only training sensor means and scales, with no clipping.

Both implementations run on **CPU** in the worker. ESN uses custom sparse typed-array JavaScript; GRU uses vendored TensorFlow.js 4.22.0. This measures these browser implementations, rather than isolating mathematical algorithm cost. A WebGL/WebGPU GRU could give a different trade-off. The widths and parameter counts are deliberately visible; this is not a matched-capacity comparison.

Active training time is accumulated using `performance.now()`, excluding downloads, validation/test, snapshot copies, status rendering and yielding. Total wall time is also exported and will be longer. Model setup and the first training operation are included; neither model is pretrained. TensorFlow.js library loading/backend availability is outside the training clock.

GRU validation runs every 8 updates, at epoch boundaries and on the first budget crossing. Thus time to target is the first *observed* crossing, not the exact earliest possible crossing. Early-stopping patience is counted in these validation checks, not epochs. The selected equal-budget snapshot is evaluated after training even when it lies between routine validation checkpoints. Time caps can overrun by one batch. No deadlines are padded to manufacture a training advantage.

## Data

The included 640-cycle sample comes from **Condition monitoring of hydraulic systems**, Helwig, Pignanelli and Schütze (2015), [DOI 10.24432/C5CW21](https://doi.org/10.24432/C5CW21), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- [Original UCI dataset](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
- [Kaggle mirror](https://www.kaggle.com/datasets/jjacostupa/condition-monitoring-of-hydraulic-systems)

The pilot was prepared from UCI, not the Kaggle mirror. Changes: six raw sensor channels (PS1, PS2, PS3, FS1, FS2, EPS1), half-second bin averages, training-only z-score scaling, and balanced subsampling. Each 60-second cycle retains 120 time steps. Virtual efficiency channels are excluded. Valve labels are optimal 100%, small lag 90%, severe lag 80%, and near failure 73%.

There are 100 training, 30 validation and 30 test cycles per valve class. Contiguous runs with identical first four profile columns are grouped; runs shorter than five cycles are excluded. Group assignment is seeded and valve-stratified, then cycles are sampled within each split. No cycle or acquisition run crosses splits. This reduces adjacent-cycle leakage but does not establish transfer to a different rig, and repeated configurations may still occur in different runs. The sample is balanced by design, so its class proportions do not represent field prevalence.

`public/examples/hydraulic/data/metadata.json` records cycle IDs, acquisition-run IDs, all original five profile values, class names, split method, train scaling, source attribution and SHA-256 hashes. Cooler, pump and accumulator labels are retained for a future shared-reservoir extension; their additional readout UI is not implemented in v0.1. Random seeds are separate for data preparation (20261004) and models (default 42).

To reproduce data preparation, download the [original archive](https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip) and run with Python 3.9+:

```sh
python3 scripts/prepare_data.py /path/to/hydraulic.zip
```

## Validation and experiments

```sh
npm test
npm run experiment -- '{"seed":42,"maxSeconds":60}' experiment-result.json
```

In Windows PowerShell, pass the JSON directly to Node so its quotes are preserved:

```powershell
$experimentConfig = '{"seed":42,"maxSeconds":60}'
node .\scripts\run_experiment.cjs $experimentConfig .\experiment-result.json
```

The Node command runs the same JS training engine and TensorFlow.js CPU backend as the browser. Its timings are desktop JS timings, not phone measurements. `experiments/` contains exploratory pilot results and a short interpretation. Hyperparameters were explored during development, so these artifacts are a functionality demonstration rather than a preregistered research evaluation.

Optional browser checks require Playwright and its Chromium binary:

```sh
npm install
npx playwright install chromium
npm run serve
# In a second terminal:
npm run test:browser
```

The optional browser check is designed to verify cancellation, a completed budget experiment, deadline selection, result download, and 390-pixel layout overflow; it saves screenshots in `work/`. For other locally installed Chrome binaries, set `PILOT_CHROME_PATH`.

Completed checks for this package: numerical/data tests, Node experiments, and cloud-browser runs of the matching protocol, plus a phone-width iframe layout with no horizontal overflow and working cancellation. These are not physical-phone measurements. The cloud browser displayed valid JSON/CSV blob links, but its download-event checks timed out; file saving should be checked on the presentation phones. Local automated Chromium could not launch in the build environment, so that optional harness has not been certified here.

For lab evidence, freeze a configuration before collecting runs. Repeat model seeds and record results on actual phones, with browser version and device model. Avoid simultaneous competing workloads. Compare distributions of accuracy and training time, and report the fraction of runs that reach the target. Later extend to a GRU width/learning-rate sweep and multiple ESN widths/regularisation values using validation only, then evaluate once on a fresh test split.

## Source map

- `public/index.html`: experiment 01 and the site landing page.
- `public/assets/`: shared site shell and navigation styles.
- `public/examples/hydraulic/esn.js`: reservoir dynamics, pooled features, ridge solve and metrics.
- `public/examples/hydraulic/experiment.js`: shared ESN/GRU experiment and selection protocol.
- `public/examples/hydraulic/worker.js`: data loading and CPU training away from the UI thread.
- `public/examples/hydraulic/app.js` and `styles.css`: hydraulic controls, results, curves and exports.
- `public/examples/double-pendulum/`: live double-pendulum simulation, sparse ESN worker, online recursive-least-squares readout and deadline/error visualisation for experiment 02.
- `scripts/`: data preparation, numerical/data checks, local server and browser checks.
- `public/vendor/`: bundled TensorFlow.js and its Apache 2.0 license.

No production dependency install, transpilation, analytics, sign-in or data submission is needed. External links in the footer are references only. The vendored TFJS source map is not included.
