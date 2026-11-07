# Virasoro Extremal Projector (`virasoro_projector.py`)

Exact symbolic computation of the **Virasoro extremal projector** components $P[n]$, the partial projectors $P_N = \sum_{n=0}^N P[n]$, the **Shapovalov matrices** $S^{(n)}(L_0, c)$, the **primitive Kac polynomials** $K_n(L_0, c)$, and the exact **$N$-th regularizing denominators** $D_N(L_0, c)$ over $\mathbb{Q}(L_0, c)$.

---

## Mathematical Overview

Let $\mathrm{Vir} = \bigoplus_{m \in \mathbb{Z}} \mathbb{C} L_m \oplus \mathbb{C} c$ be the Virasoro Lie algebra with commutation relations:
$$[L_m, L_n] = (m - n) L_{m+n} + \frac{m^3 - m}{12} \delta_{m+n, 0} c, \qquad [c, \mathrm{Vir}] = 0.$$

With respect to the triangular decomposition $\mathrm{Vir} = \mathrm{Vir}_- \oplus \mathfrak{h} \oplus \mathrm{Vir}_+$ (where $\mathfrak{h} = \mathbb{C} L_0 \oplus \mathbb{C} c$ and $\mathrm{Vir}_\pm = \bigoplus_{m > 0} \mathbb{C} L_{\pm m}$), the **extremal projector** $P$ is the unique formal operator in a completion of the localized universal enveloping algebra $U'(\mathrm{Vir})$ satisfying:
$$L_m P = 0 \quad (m > 0), \qquad P L_{-m} = 0 \quad (m > 0), \qquad [L_0, P] = 0, \qquad P^2 = P, \qquad P[0] = 1.$$

For each integer partition $\lambda = (\lambda_1 \le \lambda_2 \le \dots \le \lambda_k)$ of $n$ ($\lambda \vdash n$), define the PBW monomials:
$$L_\lambda = L_{\lambda_1} L_{\lambda_2} \cdots L_{\lambda_k}, \qquad L_\lambda^\dagger = L_{-\lambda_k} \cdots L_{-\lambda_2} L_{-\lambda_1}.$$
Then $P = \sum_{n=0}^\infty P[n]$, where the level-$n$ component $P[n]$ in **left-placement form** is:
$$P[n] = \sum_{\lambda, \mu \vdash n} b^{(n)}_{\lambda, \mu}(L_0, c) \, L_\lambda^\dagger L_\mu,$$
with rational coefficients $b^{(n)}_{\lambda, \mu}(L_0, c) \in \mathbb{Q}(L_0, c)$ satisfying $b^{(n)}_{\lambda, \mu} = b^{(n)}_{\mu, \lambda}$.

### Recursive Computation on the Universal Verma Module

Let $M = U'(\mathrm{Vir}) / U'(\mathrm{Vir})\mathrm{Vir}_+$ be the localized universal Verma module with canonical cyclic generator $|1\rangle$. On the weight space $M[-n] = \mathrm{span}_{U'(\mathfrak{h})}\{L_\eta^\dagger |1\rangle : \eta \vdash n\}$, the defining condition $P|_{M[-n]} = 0$ ($n \ge 1$) yields the exact recursive linear system:
$$B^{(n)}_{\mathrm{shifted}}(L_0, c) \, S^{(n)}(L_0, c) = -\sum_{k=0}^{n-1} M_{P[k] \to n}(L_0, c),$$
where:
* $S^{(n)}_{\mu, \eta}(L_0, c) |1\rangle = L_\mu L_\eta^\dagger |1\rangle$ is the level-$n$ **Shapovalov matrix**,
* $\big(B^{(n)}_{\mathrm{shifted}}(L_0, c)\big)_{\lambda, \mu} = b^{(n)}_{\lambda, \mu}(L_0 + n, c)$ are the right-shifted projector coefficients, and
* $M_{P[k] \to n}(L_0, c)$ is the action matrix of $P[k]$ on $M[-n]$.

This package solves this recursion symbolically using SymPy's `DomainMatrix` over $\mathbb{Q}(L_0, c)$, enabling fast exact computation up to $N = 4$ (and beyond) in seconds.

---

## Repository Structure

```text
virasoro-extremal-projector/
├── virasoro_projector.py       # Main symbolic library and CLI tool
├── test_virasoro_projector.py  # Automated unit test suite
├── requirements.txt            # Python dependencies (sympy>=1.12)
├── README.md                   # Documentation and mathematical reference
├── LICENSE                     # MIT License
└── .gitignore                  # Standard Python/Jupyter gitignore
```

---

## Installation

1. Clone this repository (or copy the folder):
   ```bash
   git clone https://github.com/<your-username>/virasoro-extremal-projector.git
   cd virasoro-extremal-projector
   ```
2. Install the dependency (`sympy`):
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Command-Line Interface (CLI)

By default, running `virasoro_projector.py` computes $P[n]$, $S^{(n)}$, $K_n$, and the least common denominators $D_N(L_0, c)$ up to level $N = 3$, followed by a LaTeX summary block:

```bash
python3 virasoro_projector.py
```

To compute up to level $N = 4$ (runs in ~5 seconds):

```bash
python3 virasoro_projector.py --max-level 4
```

To output only the LaTeX `align*` environment for the denominators $D_1, \dots, D_N$:

```bash
python3 virasoro_projector.py --max-level 4 --latex-only
```

### 2. Python API / Jupyter Notebook Usage

`virasoro_projector.py` uses `parse_known_args()` so it can also be run or imported directly inside a Jupyter Notebook:

```python
from virasoro_projector import (
    L0, c,
    compute_extremal_projector,
    shapovalov_matrix,
    kac_polynomial_Km,
    print_report,
)

# Compute all projector data up to level N = 4
data = compute_extremal_projector(max_level=4)

# Access the level-3 left-placement coefficient matrix b^{(3)}_{lambda, mu}(L0, c)
B3_left = data["B_left"][3]

# Access the exact least common denominator D_4(L0, c) of P_4 = P[0] + P[1] + P[2] + P[3] + P[4]
D4 = data["LCD_PN"][4]
print("D_4(L0, c) =", D4)
```

---

## Summary of Computed Denominators $D_N(L_0, c)$ ($N = 1, 2, 3, 4$)

For the partial projector $P_N = \sum_{n=0}^N P[n]$, the exact least common denominator $D_N(L_0, c)$ satisfies:
$$D_N(L_0, c) = \prod_{m=1}^N \tau_{-m}\big(K_m(L_0, c)\big) = \prod_{m=1}^N K_m(L_0 - m, c),$$
where $K_m(L_0, c) = \prod_{rs = m} \psi_{r,s}(L_0, c)$ is the primitive level-$m$ Kac polynomial:

$$\begin{aligned}
D_{1}(L_0, c) &= L_0 - 1 \\
D_{2}(L_0, c) &= \left(L_0 - 1\right) \left(16 L_0^{2} + 2 L_0 c - 74 L_0 - 3 c + 84\right) \\
D_{3}(L_0, c) &= \left(L_0 - 1\right) \left(3 L_0^{2} + L_0 c - 25 L_0 - 2 c + 50\right) \left(16 L_0^{2} + 2 L_0 c - 74 L_0 - 3 c + 84\right) \\
D_{4}(L_0, c) &= \left(L_0 - 1\right) \left(8 L_0 + c - 33\right) \left(3 L_0^{2} + L_0 c - 25 L_0 - 2 c + 50\right) \left(16 L_0^{2} + 2 L_0 c - 74 L_0 - 3 c + 84\right) \\
&\quad \times \left(16 L_0^{2} + 10 L_0 c - 210 L_0 - 25 c + 650\right)
\end{aligned}$$

---

## Running Tests

Run the automated unit test suite via `unittest`:

```bash
python3 -m unittest test_virasoro_projector.py -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).
