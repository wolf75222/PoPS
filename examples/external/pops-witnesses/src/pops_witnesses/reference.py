"""Independent discrete oracle. No PoPS, generated-code or authoring imports."""

from dataclasses import dataclass

import numpy as np

BASE_REVISION = "620ac6b6d2ad532a7616652d1bdaac9ca507287a"
Q_CEILING = 3.0
RHS_CEILING = 4.0
CG_RTOL, CG_ATOL = 1e-12, 1e-14


@dataclass(frozen=True)
class Physics:
    epsilon: float = 0.0
    beta: float = 0.0
    cells: tuple = (16, 12)
    lengths: tuple = (2.0, 3.0)
    dt: float = 1e-4
    diffusion: float = 0.15
    decay: float = 0.8
    screenings: tuple = (3.0, 5.0)
    donor_psi: float = -0.75

    @property
    def weights(self):
        return 0.2 + self.epsilon, 0.4 - self.epsilon


DEFAULT_PHYSICS = Physics()

VARIANTS = {
    "baseline": Physics(),
    "linear": Physics(epsilon=0.05),
    "linear-renamed": Physics(epsilon=0.05),
    "cubic": Physics(epsilon=0.05, beta=0.05),
    "cubic-renamed": Physics(epsilon=0.05, beta=0.05),
    "beta-zero": Physics(epsilon=0.05, beta=0.0),
}


def cell_means(p=DEFAULT_PHYSICS):
    """Integrate cosines over cell faces; independent of author's center/sinc formula."""
    nx, ny = p.cells
    x = 2 * np.pi * np.arange(nx + 1) / nx
    y = 2 * np.pi * np.arange(ny + 1) / ny
    mode = np.diff(np.sin(y))[:, None] * np.diff(np.sin(x))[None, :]
    mode /= (2 * np.pi / ny) * (2 * np.pi / nx)
    return {
        "q": np.ascontiguousarray((2.0 + 0.125 * mode)[None]),
        "d": np.ascontiguousarray((0.6 + 0.07 * mode)[None]),
    }


def laplacian(u, p=DEFAULT_PHYSICS):
    nx, ny = p.cells
    hx, hy = p.lengths[0] / nx, p.lengths[1] / ny
    out = np.empty_like(u)
    for y in range(ny):
        for x in range(nx):
            out[0, y, x] = (
                u[0, y, (x - 1) % nx] - 2 * u[0, y, x] + u[0, y, (x + 1) % nx]
            ) / hx**2 + (u[0, (y - 1) % ny, x] - 2 * u[0, y, x] + u[0, (y + 1) % ny, x]) / hy**2
    return out


def screened(rhs, sigma, p=DEFAULT_PHYSICS):
    nx, ny = p.cells
    hx, hy = p.lengths[0] / nx, p.lengths[1] / ny
    symbol = (
        4 * np.sin(np.pi * np.arange(nx) / nx)[None, :] ** 2 / hx**2
        + 4 * np.sin(np.pi * np.arange(ny) / ny)[:, None] ** 2 / hy**2
    )
    return np.fft.ifftn(np.fft.fftn(rhs, axes=(-2, -1)) / (sigma + symbol), axes=(-2, -1)).real


def dense_screened(rhs, sigma, p=DEFAULT_PHYSICS):
    """Separate dense matrix assembly; does not call the FFT symbol or stencil."""
    nx, ny = p.cells
    hx, hy = p.lengths[0] / nx, p.lengths[1] / ny
    matrix = np.zeros((nx * ny, nx * ny))
    for y in range(ny):
        for x in range(nx):
            row = y * nx + x
            matrix[row, row] = sigma + 2 / hx**2 + 2 / hy**2
            for yy, xx, scale in (
                (y, (x - 1) % nx, hx),
                (y, (x + 1) % nx, hx),
                ((y - 1) % ny, x, hy),
                ((y + 1) % ny, x, hy),
            ):
                matrix[row, yy * nx + xx] -= 1 / scale**2
    return np.linalg.solve(matrix, rhs.ravel()).reshape(rhs.shape)


def fields_rate(q, d, p, *, cubic_power=3):
    phi = screened(q + d, p.screenings[0], p)
    psi = screened(q + p.donor_psi * d, p.screenings[1], p)
    a, b = p.weights
    rate = p.diffusion * laplacian(q, p) + a * phi + b * psi - p.decay * q
    if p.beta:
        rate -= p.beta * q**cubic_power
    return phi, psi, rate


def reference(p, *, freeze_fields=False, cubic_power=3):
    initial = cell_means(p)
    q0, d = initial["q"], initial["d"]
    first = fields_rate(q0, d, p, cubic_power=cubic_power)

    def evaluate(q):
        if not freeze_fields:
            return fields_rate(q, d, p, cubic_power=cubic_power)
        a, b = p.weights
        rate = p.diffusion * laplacian(q, p) + a * first[0] + b * first[1] - p.decay * q
        if p.beta:
            rate -= p.beta * q**cubic_power
        return first[0], first[1], rate

    q1 = q0 + p.dt * first[2]
    second = evaluate(q1)
    q2 = 0.75 * q0 + 0.25 * q1 + 0.25 * p.dt * second[2]
    third = evaluate(q2)
    qfinal = q0 / 3 + (2 / 3) * q2 + (2 / 3) * p.dt * third[2]
    out = dict(q0=q0, q1=q1, q2=q2, qfinal=qfinal, d_initial=d, d_final=d.copy())
    for stage, field in enumerate((first, second, third)):
        out["phi" + str(stage)], out["psi" + str(stage)] = field[:2]
    return out


def modal_reference(p):
    """Independent scalar affine recurrence, valid only for beta=0."""
    if p.beta:
        raise ValueError("a single-mode oracle cannot represent the cubic source")
    nx, ny = p.cells
    hx, hy = p.lengths[0] / nx, p.lengths[1] / ny
    lam = 4 * np.sin(np.pi / nx) ** 2 / hx**2 + 4 * np.sin(np.pi / ny) ** 2 / hy**2
    mode = cell_means(p)["q"] - 2.0
    mode /= 0.125
    coefficients = []
    w0, w1 = p.weights
    for eigenvalue, q, d in ((0.0, 2.0, 0.6), (lam, 0.125, 0.07)):
        a = -p.diffusion * eigenvalue - p.decay
        a += w0 / (eigenvalue + 3) + w1 / (eigenvalue + 5)
        b = w0 / (eigenvalue + 3) - 0.75 * w1 / (eigenvalue + 5)
        q1 = q + p.dt * (a * q + b * d)
        q2 = 0.75 * q + 0.25 * (q1 + p.dt * (a * q1 + b * d))
        end = q / 3 + (2 / 3) * (q2 + p.dt * (a * q2 + b * d))
        coefficients.append((q, q1, q2, end))
    return {
        name: coefficients[0][i] + coefficients[1][i] * mode
        for i, name in enumerate(("q0", "q1", "q2", "qfinal"))
    }


def budget(p):
    """Freeze before execution; never fit tolerances to native differences."""
    nx, ny = p.cells
    hx, hy = p.lengths[0] / nx, p.lengths[1] / ny
    eps = np.finfo(np.float64).eps
    residual = 2 * nx * ny * RHS_CEILING * CG_RTOL + CG_ATOL / min(1.0, hx * hy)
    rounding, seed = 128 * eps * RHS_CEILING, 16 * eps * RHS_CEILING
    phi_error, psi_error = residual / 3 + rounding, residual / 5 + rounding
    w0, w1 = p.weights
    lipschitz = p.diffusion * 4 * (1 / hx**2 + 1 / hy**2) + abs(p.decay)
    lipschitz += abs(w0) / 3 + abs(w1) / 5 + 3 * abs(p.beta) * Q_CEILING**2
    cross = abs(w0) / 3 + 0.75 * abs(w1) / 5
    force = abs(w0) * phi_error + abs(w1) * psi_error + cross * seed
    force += 16 * eps * abs(p.beta) * Q_CEILING**3
    e0 = seed
    e1 = (1 + p.dt * lipschitz) * e0 + p.dt * force + rounding
    e2 = 0.75 * e0 + 0.25 * (1 + p.dt * lipschitz) * e1 + 0.25 * p.dt * force + rounding
    ef = e0 / 3 + (2 / 3) * (1 + p.dt * lipschitz) * e2 + (2 / 3) * p.dt * force + rounding
    bounds = dict(q0=e0, q1=e1, q2=e2, qfinal=ef, d_initial=seed, d_final=seed)
    for stage, error in enumerate((e0, e1, e2)):
        bounds["phi" + str(stage)] = phi_error + (error + seed) / 3
        bounds["psi" + str(stage)] = psi_error + (error + 0.75 * seed) / 5
    return {
        "bounds": bounds,
        "Q": Q_CEILING,
        "rhs_ceiling": RHS_CEILING,
        "residual": residual + 256 * eps * (4 / hx**2 + 4 / hy**2 + 5) * RHS_CEILING,
        "lipschitz": lipschitz,
        "premises": [
            "Discrete periodic screened M-matrix, sigma_min=3; existing CG success policy.",
            "CG max_iter=2000, rtol=1e-12, atol=1e-14; RHS ceiling=4.",
            "Every initial/stage/accepted q in Q=3; cube of stored cell mean.",
            "Supported density/integral row and normalized/unscaled norm conventions.",
        ],
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_arrays(arrays, p):
    expected, limits = reference(p), budget(p)
    require(set(arrays) == set(expected), "missing or unexpected output arrays")
    defects = {}
    for name, target in expected.items():
        value = arrays[name]
        require(value.dtype == np.float64 and value.shape == target.shape, "shape/dtype: " + name)
        require(np.isfinite(value).all(), "nonfinite: " + name)
        if name.startswith("q"):
            require(np.max(np.abs(value)) <= Q_CEILING, "Q=3 premise violated: " + name)
        error = float(np.max(np.abs(value - target)))
        require(error <= limits["bounds"][name], "oracle mismatch: " + name)
        defects[name] = {"error": error, "bound": limits["bounds"][name]}
    require(arrays["d_final"].tobytes() == arrays["d_initial"].tobytes(), "donor changed")
    residuals = {}
    for stage in range(3):
        q, d = arrays["q" + str(stage)], arrays["d_initial"]
        for role, sigma, rhs in (("phi", 3, q + d), ("psi", 5, q - 0.75 * d)):
            require(np.max(np.abs(rhs)) <= RHS_CEILING, "RHS ceiling violated")
            r = -laplacian(arrays[role + str(stage)], p) + sigma * arrays[role + str(stage)] - rhs
            error = float(np.max(np.abs(r)))
            require(error <= limits["residual"], "field residual: " + role + str(stage))
            residuals[role + str(stage)] = error
    return {"arrays": defects, "residuals": residuals}


def cosine_20(value, p=DEFAULT_PHYSICS):
    """2/(Nx Ny) projection onto cos(2 theta_x), theta_x=2 pi (x+1/2)/Nx."""
    nx, ny = p.cells
    theta = 2 * np.pi * (np.arange(nx) + 0.5) / nx
    return float(2 * np.sum(value * np.cos(2 * theta)[None, None, :]) / (nx * ny))


def analytic_cubic_20(p):
    nx, ny = p.cells
    amplitude = 0.125 * np.sinc(1 / nx) * np.sinc(1 / ny)
    return -p.dt * p.beta * 3 * 2.0 * amplitude**2 / 4


def check_pairs(captures):
    results = {}
    for first, second in (("baseline", "linear"), ("linear", "cubic")):
        a, b = captures[first], captures[second]
        p, r = VARIANTS[first], VARIANTS[second]
        gap = b["qfinal"] - a["qfinal"]
        limit = budget(p)["bounds"]["qfinal"] + budget(r)["bounds"]["qfinal"]
        predicted = reference(r)["qfinal"] - reference(p)["qfinal"]
        require(np.max(np.abs(gap)) > 100 * limit, "change invisible above fixed budgets")
        require(np.max(np.abs(gap - predicted)) <= limit, "final gap mismatch")
        stage_gap = b["q1"] - a["q1"]
        expected = (
            p.dt * r.epsilon * (a["phi0"] - a["psi0"])
            if first == "baseline"
            else -r.dt * r.beta * a["q0"] ** 3
        )
        stage_limit = budget(p)["bounds"]["q1"] + budget(r)["bounds"]["q1"]
        require(np.max(np.abs(stage_gap - expected)) <= stage_limit, "first-stage relation")
        results[first + "->" + second] = {
            "max_final_gap": float(np.max(np.abs(gap))),
            "bound": limit,
        }
    for original, renamed in (
        ("linear", "linear-renamed"),
        ("cubic", "cubic-renamed"),
        ("linear", "beta-zero"),
    ):
        require(
            all(
                captures[original][k].tobytes() == captures[renamed][k].tobytes()
                for k in captures[original]
            ),
            "rename/beta-zero changes numbers: " + renamed,
        )
    observed = cosine_20(captures["cubic"]["q1"] - captures["linear"]["q1"])
    expected = analytic_cubic_20(VARIANTS["cubic"])
    limit = 2 * (
        budget(VARIANTS["cubic"])["bounds"]["q1"] + budget(VARIANTS["linear"])["bounds"]["q1"]
    )
    require(abs(observed - expected) <= limit and abs(observed) > 100 * limit, "harmonic (2,0)")
    results["cosine_20"] = {"observed": observed, "analytic": expected, "bound": limit}
    return results
