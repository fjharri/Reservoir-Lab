# Exploratory pilot experiments

These runs establish that the implementation trains real models and exposes the intended time/accuracy trade-off. They do not predict phone timings or establish that an ESN is generally superior. Files contain full curves, settings, split hashes and confusion matrices. All figures below are observed desktop Node.js / TensorFlow.js CPU results; the browser uses the same training engine.

| Run | ESN width | Model seed | GRU learning rate | ESN training | ESN validation / test | GRU same-budget test | Time to validation target | Matched GRU test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `desktop-seed42-match.json` | 500 | 42 | 0.01 | 1.281 s | 77.5% / 65.8% | 36.7% (6 updates) | 15.531 s (100 updates) | 79.2% |
| `desktop-seed7-budget.json` | 250 | 7 | 0.01 | 0.663 s | 68.3% / 50.8% | 38.3% (2 updates) | Budget mode | — |
| `desktop-seed42-lr003-cap10s.json` | 500 | 42 | 0.003 | 1.380 s | 77.5% / 65.8% | See JSON | Not reached before 10 s cap | — |

For the first run, the GRU needed approximately **14.25 seconds more active training** (12.1× the ESN's total training time) to reach the ESN's validation accuracy. The GRU then generalized better on this test split. This is an educational example of fast fitting versus additional optimization, not a case in which one model must win every comparison.

The learning rate of 0.003 initially learned slowly. The 0.01 experiment showed real learning, reaching 91.25% training accuracy and the validation target at its second epoch boundary. The UI therefore defaults to 0.01 but exposes all three candidate rates. No ESN accuracy threshold is hardcoded: each run obtains its own target from validation.

Width/seed changes also affected ESN performance. The gap between validation and test results warrants further exploration and repeated runs. Treat these splits as development/demo splits if performing further tuning; create a new untouched test split for formal conclusions.

Run isolation: the principal seed-42 matching experiment completed before browser tests. The seed-7 run exercised budget selection with a smaller reservoir. Small setup/timing variations change how many GRU updates fit within a short ESN budget. The initial 10-second run predates extra curve annotations and final training-accuracy logging; model computations and split are unchanged.

The application has also completed an actual browser budget experiment at default settings: ESN 1.19 s / 65.8% test; GRU 1.17 s used / 36.7% test with 6 complete updates. Browser wall time was approximately 10.22 s because validation and test inference are additional work. This was a desktop cloud browser, not a physical phone.

A subsequent run in a phone-width iframe reached the validation target after 16.63 s versus ESN 1.16 s, with matched GRU test accuracy of 79.2%. Its equal-budget snapshot retained four updates and scored 25.8% on test. The iframe used desktop CPU and is a layout check only. This variation in equal-budget updates is expected from timing/setup overhead on short deadlines. The final responsive-chart QA repeated the budget run: ESN 1.17 s / 65.8%, GRU 1.04 s used / 25.8%, four updates. JSON/CSV link preparation was observed, but native download verification was blocked by timeouts in this cloud browser.

Next experiment: freeze a configuration, collect several model seeds on several real phones, and export each run. Report median/IQR training time, test accuracy and target-reaching rate. Include a tuned GRU and optional GPU backend before making broader efficiency claims.
