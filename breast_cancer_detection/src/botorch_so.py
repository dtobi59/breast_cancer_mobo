"""
Single-Objective Bayesian Optimization with BoTorch.

Implements standard BO for maximizing a single objective (PR-AUC).
Uses qExpectedImprovement acquisition function with SingleTaskGP model.
"""

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine


class SingleObjectiveGPModel:
    """
    Gaussian Process model for single-objective optimization.

    Uses SingleTaskGP with Matern 5/2 kernel (default) and ARD.
    Simpler than ModelListGP since we only model one objective.
    """

    def __init__(self, n_vars=5, bounds=None):
        """
        Initialize single-objective GP model.

        Args:
            n_vars: Number of decision variables (5 hyperparameters)
            bounds: (2, n_vars) tensor with [lower, upper] bounds in normalized [0,1] space
        """
        self.n_vars = n_vars
        self.bounds = bounds if bounds is not None else torch.tensor([
            [0.0] * n_vars,  # Lower bounds (normalized)
            [1.0] * n_vars   # Upper bounds (normalized)
        ], dtype=torch.float64)

        self.model = None
        self.train_X = None
        self.train_Y = None

    def update_data(self, X_new, Y_new):
        """
        Add new evaluations to training data.

        Args:
            X_new: (n_new, n_vars) tensor of inputs in normalized space
            Y_new: (n_new,) or (n_new, 1) tensor of objective values
        """
        # Ensure Y is 2D for consistency with BoTorch SingleTaskGP
        if Y_new.ndim == 1:
            Y_new = Y_new.unsqueeze(-1)

        if self.train_X is None:
            self.train_X = X_new
            self.train_Y = Y_new
        else:
            self.train_X = torch.cat([self.train_X, X_new], dim=0)
            self.train_Y = torch.cat([self.train_Y, Y_new], dim=0)

    def fit_model(self):
        """
        Fit Gaussian Process to current training data.

        Uses:
        - SingleTaskGP (standard for single objective)
        - Matern 5/2 kernel with ARD (automatic relevance determination)
        - Automatic output standardization
        - MLE for hyperparameter optimization
        """
        if self.train_X is None or len(self.train_X) < 2:
            raise ValueError(f"Need at least 2 training points to fit GP (have {len(self.train_X) if self.train_X is not None else 0})")

        # Create GP model
        # SingleTaskGP automatically handles output standardization
        self.model = SingleTaskGP(
            train_X=self.train_X,
            train_Y=self.train_Y,
            # Default kernel: Matern 5/2 with ARD
        )

        # Fit hyperparameters via marginal likelihood maximization
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        return self.model

    def predict(self, X):
        """
        Predict objective with uncertainty.

        Args:
            X: (n_candidates, n_vars) tensor in normalized space

        Returns:
            means: (n_candidates,) predicted means
            stds: (n_candidates,) predicted standard deviations
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_model() first.")

        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X)
            mean = posterior.mean.squeeze(-1)
            std = posterior.variance.sqrt().squeeze(-1)

        return mean, std


class ExpectedImprovementAcquisition:
    """
    Manages qExpectedImprovement acquisition function.

    qEI is the standard acquisition function for single-objective BO.
    Supports batch acquisition (q > 1) via sequential greedy construction.
    """

    def __init__(self, model, best_f, bounds, sampler=None):
        """
        Initialize Expected Improvement acquisition function.

        Args:
            model: Fitted SingleTaskGP model
            best_f: Current best objective value (for minimization)
            bounds: (2, n_vars) bounds tensor in normalized [0,1] space
            sampler: Optional sampler for MC acquisition (default: SobolQMCNormalSampler)
        """
        self.model = model
        self.best_f = best_f
        self.bounds = bounds

        # Create sampler for MC acquisition (needed for q > 1)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        # Create qEI acquisition function
        self.acq_func = qExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler
        )

    def optimize(self, q=4, num_restarts=20, raw_samples=512):
        """
        Optimize acquisition function to suggest next candidates.

        Args:
            q: Batch size (number of candidates to suggest)
            num_restarts: Number of optimization restarts
            raw_samples: Initial random samples for optimization

        Returns:
            candidates: (q, n_vars) tensor of suggested points in normalized space
            acq_value: Acquisition function value at candidates
        """
        candidates, acq_value = optimize_acqf(
            acq_function=self.acq_func,
            bounds=self.bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={
                "batch_limit": 5,  # Max candidates per batch for memory
                "maxiter": 200,    # Max iterations for L-BFGS-B
            },
            sequential=True  # Sequential greedy for batches (better for q > 1)
        )

        return candidates, acq_value


def suggest_candidates_ei(
    gp_model,
    batch_size=4,
    num_restarts=20,
    raw_samples=512
):
    """
    High-level function to suggest next candidates using Expected Improvement.

    This is the main interface for single-objective BO candidate suggestion.

    Args:
        gp_model: Fitted SingleObjectiveGPModel
        batch_size: Number of candidates to suggest
        num_restarts: Optimization restarts for acquisition maximization
        raw_samples: Initial random samples for optimization

    Returns:
        candidates: (batch_size, n_vars) tensor in normalized space
        acq_value: Acquisition function value
    """
    if gp_model.model is None:
        raise ValueError("GP model must be fitted first. Call gp_model.fit_model().")

    # Get current best value (minimum, since we minimize)
    # For maximization objectives (like PR-AUC), we negate them for BO
    best_f = gp_model.train_Y.min().item()

    # Create acquisition function
    acq = ExpectedImprovementAcquisition(
        model=gp_model.model,
        best_f=best_f,
        bounds=gp_model.bounds
    )

    # Optimize to get candidates
    candidates, acq_value = acq.optimize(
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples
    )

    return candidates, acq_value


def initial_sobol_sampling(n_vars, n_samples, bounds):
    """
    Generate initial Sobol samples in normalized space.

    Sobol sequences are low-discrepancy quasi-random sequences that provide
    better space-filling properties than random sampling for initial exploration.

    Args:
        n_vars: Number of variables (dimensionality)
        n_samples: Number of samples to generate
        bounds: (2, n_vars) bounds tensor with [lower, upper] bounds

    Returns:
        X: (n_samples, n_vars) Sobol samples in normalized [0,1] space
    """
    # Create Sobol engine
    sobol = SobolEngine(dimension=n_vars, scramble=True, seed=None)

    # Generate samples in [0, 1]^n_vars
    X_norm = sobol.draw(n_samples, dtype=torch.float64)

    # Unnormalize to [bounds[0], bounds[1]]
    # For normalized bounds [0, 1], this is just X_norm
    # But kept for consistency with multi-objective code
    X = unnormalize(X_norm, bounds)

    return X
