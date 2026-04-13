import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


def compute_jaccard(v1, v2, ks=None):
    """Jaccard index between the top-k elements of two vectors, for each k."""
    if ks is None:
        ks = [5, 10, 25, 50]
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    n = min(len(v1), len(v2))
    order1 = np.argsort(v1)[::-1]
    order2 = np.argsort(v2)[::-1]
    results = []
    for k in ks:
        k_ = min(k, n)
        top1 = set(order1[:k_])
        top2 = set(order2[:k_])
        results.append(len(top1 & top2) / len(top1 | top2))
    return np.array(results)


class MMDTest:
    def __init__(self, kernel_type="rbf", sigma=None):
        self.kernel_type = kernel_type
        self.sigma = sigma

    def k(self, X, Y):
        X_ = X[:, None]
        Y_ = Y[None, :]
        if self.kernel_type == "rbf":
            dists = ((X_ - Y_) ** 2).sum(-1)
            return np.exp(-dists / (2 * self.sigma**2))
        elif self.kernel_type == "linear":
            return (X_ * Y_).sum(-1)
        else:
            raise ValueError(f"Invalid kernel type: {self.kernel_type}")

    def _estimate_mmd(self, X, Y):
        K_XX = self.k(X, X)
        K_XX[np.diag_indices_from(K_XX)] = 0.0
        K_YY = self.k(Y, Y)
        K_YY[np.diag_indices_from(K_YY)] = 0.0
        K_XY = self.k(X, Y)

        contrib_xx = K_XX.sum() / (X.shape[0] * (X.shape[0] - 1))
        contrib_yy = K_YY.sum() / (Y.shape[0] * (Y.shape[0] - 1))
        contrib_xy = K_XY.sum() / (X.shape[0] * Y.shape[0])

        return contrib_xx + contrib_yy - 2 * contrib_xy

    def _preprocess(self, X, Y):
        if (self.kernel_type == "rbf") and (self.sigma is None):
            Z = np.concatenate([X, Y])
            dists = ((Z[:, None] - Z[None, :]) ** 2).sum(-1)
            upper_tri = dists[np.triu_indices_from(dists, k=1)]
            median_sq_dist = np.median(upper_tri)
            self.sigma = np.sqrt(median_sq_dist) if median_sq_dist > 0 else 1.0
        else:
            pass

    def compute_mmd(self, X, Y):
        """
        Computes the MMD statistic between X and Y.
        """
        return self._estimate_mmd(X, Y)

    def test(self, X, Y, n_iter=10000):
        self._preprocess(X, Y)
        null_dist = []
        Z = np.concatenate([X, Y])
        m = X.shape[0]
        for _ in range(n_iter):
            indices = np.random.choice(np.arange(Z.shape[0]), size=Z.shape[0], replace=True)
            Z_boot = Z[indices]
            X_boot = Z_boot[:m]
            Y_boot = Z_boot[m:]
            null_dist.append(self._estimate_mmd(X_boot, Y_boot))

        null_dist = np.array(null_dist)
        p_value = (null_dist >= self._estimate_mmd(X, Y)).mean()
        return p_value


class MMDTestJax:
    def __init__(self, kernel_type="rbf", sigma=None, max_n=50):
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.max_n = max_n  # Maximum expected sample size per group

    def k_matrix(self, X, Y):
        # Computes pairwise kernel matrix
        # Uses standard implementation that works with padded inputs
        # (padding usually results in dummy distances, masked out later)
        X_sq = jnp.sum(X**2, axis=1, keepdims=True)
        Y_sq = jnp.sum(Y**2, axis=1, keepdims=True)
        dists = X_sq + Y_sq.T - 2 * jnp.dot(X, Y.T)

        if self.kernel_type == "rbf":
            return jnp.exp(-dists / (2 * self.sigma**2))
        elif self.kernel_type == "linear":
            return jnp.dot(X, Y.T)
        else:
            raise ValueError(f"Invalid kernel type")

    def _estimate_mmd_masked(self, K_XY, m, n):
        # Create masks for valid data
        # K_XY is (N_total, N_total) where N_total = m + n (padded conceptually)
        # In the bootstrap, we treat the first m indices as X, next n as Y
        mask_x = jnp.arange(K_XY.shape[0]) < m
        mask_y = (jnp.arange(K_XY.shape[0]) >= m) & (jnp.arange(K_XY.shape[0]) < (m + n))

        # Expand masks for matrix operations
        MxMx = mask_x[:, None] * mask_x[None, :]
        MyMy = mask_y[:, None] * mask_y[None, :]
        MxMy = mask_x[:, None] * mask_y[None, :]

        # Zero out diagonals for unbiased estimator
        diag_mask = jnp.eye(K_XY.shape[0], dtype=bool)

        # Sums with masks
        sum_xx = jnp.sum(K_XY * MxMx * (~diag_mask))
        sum_yy = jnp.sum(K_XY * MyMy * (~diag_mask))
        sum_xy = jnp.sum(K_XY * MxMy)  # XY block doesn't involve self-pairs usually

        # Normalization
        # Use jnp.maximum to avoid division by zero if m=1 (though MMD requires m>1)
        c_xx = sum_xx / jnp.maximum(1.0, m * (m - 1))
        c_yy = sum_yy / jnp.maximum(1.0, n * (n - 1))
        c_xy = sum_xy / jnp.maximum(1.0, m * n)

        return c_xx + c_yy - 2 * c_xy

    @partial(jax.jit, static_argnames=["self"])
    def _compute_p_value_padded(self, keys, Z_padded, m, n):
        # 1. Compute Kernel on ALL data (padded)
        # Note: We compute K on 2*max_n size, but we only care about the top (m+n)x(m+n)
        # Optimization: We can just compute K on Z_padded once.
        K_Z = self.k_matrix(Z_padded, Z_padded)

        # Total valid samples
        N = m + n

        # Function to run one bootstrap iteration
        def body(key):
            # Sample N indices from range [0, N)
            # We use random.uniform and floor to avoid dynamic shape in random.choice
            rand_float = jax.random.uniform(key, shape=(self.max_n * 2,))
            idx = jnp.floor(rand_float * N).astype(jnp.int32)

            # Since we only need 'N' samples, but arrays must be static size 2*max_n:
            # We just use the sampled indices to gather from K_Z.
            # The mask in _estimate_mmd_masked will ignore the garbage at the end.

            # Resample Kernel
            K_boot = K_Z[idx][:, idx]

            # Calculate MMD on the first m (as X) and next n (as Y)
            return self._estimate_mmd_masked(K_boot, m, n)

        # Vectorize bootstrap
        null_dist = jax.vmap(body)(keys)

        # Compute observed statistic (no resampling)
        # The observed data corresponds to indices 0..m-1 (X) and m..m+n-1 (Y) of Z
        obs_stat = self._estimate_mmd_masked(K_Z, m, n)

        return (null_dist >= obs_stat).mean()

    def _preprocess(self, X, Y):
        # Heuristic for sigma (median heuristic) using numpy for safety/simplicity
        if (self.kernel_type == "rbf") and (self.sigma is None):
            # Downsample for sigma estimation if too large to avoid O(N^2)
            # (Optional optimization, here doing exact)
            Z = np.concatenate([X, Y])
            if Z.shape[0] > 2000:
                idx = np.random.choice(Z.shape[0], 2000, replace=False)
                Z = Z[idx]
            dists = ((Z[:, None] - Z[None, :]) ** 2).sum(-1)
            upper = dists[np.triu_indices_from(dists, k=1)]
            median = np.median(upper)
            self.sigma = np.sqrt(median) if median > 0 else 1.0

    @partial(jax.jit, static_argnames=["self"])
    def _compute_mmd_padded(self, Z_padded, m, n):
        K_Z = self.k_matrix(Z_padded, Z_padded)
        return self._estimate_mmd_masked(K_Z, m, n)

    def compute_mmd(self, X, Y):
        """
        Computes the MMD statistic between X and Y using JAX.
        """
        self._preprocess(X, Y)

        m = X.shape[0]
        n = Y.shape[0]

        target_size = 2 * self.max_n

        if m + n > target_size:
            raise ValueError(f"Input size {m+n} exceeds max_n capacity {target_size}")

        Z = np.concatenate([X, Y])
        pad_width = ((0, target_size - Z.shape[0]), (0, 0))
        Z_padded = np.pad(Z, pad_width, mode="constant")

        return float(self._compute_mmd_padded(jnp.array(Z_padded), m, n))

    @partial(jax.jit, static_argnames=["self", "batch_size"])
    def compute_distance_matrix(self, X_all, counts, batch_size=None):
        """
        Computes pairwise MMD distances for a stack of padded datasets.

        Args:
            X_all: (n_groups, max_n, n_features) padded data
            counts: (n_groups,) actual number of samples per group
            batch_size: Optional batch size for rows to avoid OOM

        Returns:
            (n_groups, n_groups) symmetric distance matrix
        """
        n_groups = X_all.shape[0]

        # Helper to compute MMD for a single pair of indices (i, j)
        def compute_pair(i, j):
            # Extract and form Z (concatenated)
            # Both X_i and X_j are already padded to max_n.
            # We want Z to be size 2*max_n.
            Xi = X_all[i]
            Xj = X_all[j]
            Z_padded = jnp.concatenate([Xi, Xj], axis=0)

            # The counts are actual sizes
            m = counts[i]
            n = counts[j]

            # Compute Kernel Matrix
            K_Z = self.k_matrix(Z_padded, Z_padded)

            # Masks
            # Valid indices for i: 0 to m-1
            # Valid indices for j: max_n to max_n + n - 1
            idx = jnp.arange(2 * self.max_n)
            mask_x = idx < m
            mask_y = (idx >= self.max_n) & (idx < self.max_n + n)

            # Expand masks
            MxMx = mask_x[:, None] * mask_x[None, :]
            MyMy = mask_y[:, None] * mask_y[None, :]
            MxMy = mask_x[:, None] * mask_y[None, :]

            # Zero out diagonals
            diag_mask = jnp.eye(2 * self.max_n, dtype=bool)

            # Sums
            sum_xx = jnp.sum(K_Z * MxMx * (~diag_mask))
            sum_yy = jnp.sum(K_Z * MyMy * (~diag_mask))
            sum_xy = jnp.sum(K_Z * MxMy)

            # Normalize
            c_xx = sum_xx / jnp.maximum(1.0, m * (m - 1))
            c_yy = sum_yy / jnp.maximum(1.0, n * (n - 1))
            c_xy = sum_xy / jnp.maximum(1.0, m * n)

            return c_xx + c_yy - 2 * c_xy

        # Vectorize over j (compute one row)
        compute_row = jax.vmap(compute_pair, in_axes=(None, 0))

        idx = jnp.arange(n_groups)

        if batch_size is None or batch_size >= n_groups:
            # Full vmap
            compute_matrix_fn = jax.vmap(compute_row, in_axes=(0, None))
            return compute_matrix_fn(idx, idx)
        else:
            # Batched execution using lax.map or scan would be ideal for JIT
            # But here we can use a simpler approach: vmap over batches of rows
            # To keep it fully JIT-able with static batch_size is tricky without knowing n_groups statically.
            # Best approach for JIT: use lax.scan over chunks.

            # We will use jax.lax.map which maps over the leading axis.
            # However, we need to map over rows i.
            # compute_row(i, idx) computes the i-th row against all j.

            # Let's use a simpler vmap with map over the first axis
            return jax.lax.map(lambda i: compute_row(i, idx), idx)

    def test(self, X, Y, n_iter=10000):
        # 1. Preprocess (Sigma) - run on CPU/Numpy to avoid JIT overhead
        self._preprocess(X, Y)

        m = X.shape[0]
        n = Y.shape[0]

        # 2. Pad Inputs to static size
        # Total buffer size needed is 2 * max_n (fit both X and Y max)
        target_size = 2 * self.max_n

        if m + n > target_size:
            raise ValueError(f"Input size {m+n} exceeds max_n capacity {target_size}")

        # Concatenate and Pad
        Z = np.concatenate([X, Y])
        pad_width = ((0, target_size - Z.shape[0]), (0, 0))
        Z_padded = np.pad(Z, pad_width, mode="constant")

        # 3. Jitted Execution
        keys = jax.random.split(jax.random.PRNGKey(42), n_iter)

        # Note: m and n are passed as dynamic args (tracers), not static.
        # Shapes of Z_padded are static.
        return float(self._compute_p_value_padded(keys, jnp.array(Z_padded), m, n))
