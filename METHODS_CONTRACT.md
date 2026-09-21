# Methodological contract


Cell annotation joins must use sample and original cell identity. Training-only scaling excludes all target-time cells in global time holdout and all cells of the held donor in donor transfer. The field acts on frozen Gaussian-encoder means, not simplex weights. It uses position and time, so heterogeneous donor-average displacements do not prove a shared field is misspecified.

Training cycles donor-segment pairs with nearly equal pair weight. Endpoints are independent draws within the same donor, defining a product coupling. The target velocity is endpoint displacement divided by the positive global-time interval; MSE averages batch examples and coordinates. This is conditional flow matching, not globally optimal transport. The separate entropic transport baseline uses squared Euclidean costs divided by their maximum, regularization 0.05 and a nearest-neighbor barycentric displacement extension.

The ideal regression identity and weak continuity equation describe the specified training path. They do not guarantee an unobserved intermediate biological marginal, identify individual ancestry, or establish a unique classical ODE flow for empirical atomic endpoints. A unique characteristic representation needs additional regularity. Conservation of normalized probability is not a model of proliferation, death or absolute cell abundance.

Training approximately balances donors but the original pooled evaluation weights cells by yield. Fixed-field equal-donor-mass evaluation is a sensitivity analysis; it does not change fitting or create new donor-transfer folds. Changing the time axis refits a different straight-bridge problem with changed interval velocities and loss weighting. It is not merely applying the time-coordinate chain rule to an already fitted field.

Context masking draws one Bernoulli mask per example, shared across its whole context vector. Its keep probability is 0.8. This regularizer does not guarantee agreement with a separately fitted shared model or stable extrapolation. Test context is computed from the source only and remains fixed during integration.

The initial timepoint energy-distance cap is 1,500 cells; donor transfer, the common benchmark and donor-count analyses use 1,200. The V-statistic includes diagonal zeros. Weighted sensitivity uses every source and target cell and exact weighted pairwise distances. The single target split-half reference is a descriptive scale, not a calibrated hypothesis test or error bound. Endpoint distance ordering does not prove geodesic interpolation.

Global time holdout, retained-donor sensitivity and genuine held-donor transfer are different tasks. Three human donors are the biological replication for the acute human cohort. Three random seeds and overlapping training subsets are computational repetitions. Each mouse arm has one animal per time point. No analysis supplies independent patient-level diabetic-foot-ulcer prognosis or a required cohort size.


Historical report fingerprints describe the producing computation. Current source includes documentation corrections and input-domain guards; it is not retroactively attributed to earlier runs. The release manifest identifies this exact distributed version.
