"""3D-SynTree: Structure-Based Molecular Design via Reaction-Constrained
Synthon Assembly with 3D-Conformational Guidance.

The package decomposes pocket-conditioned molecular generation into a
Markov Decision Process over a reaction-constrained synthon action space:

    pi(a_t | P, M_t) = P(r_t | P, M_t)          # reaction selection
                     * P(B_t | r_t, P, M_t)      # synthon selection
                     * p(phi_t | B_t, r_t, P, M_t)  # dihedral torsion

Every generated molecule is accompanied by an actionable multi-step
synthesis recipe built from purchasable building blocks.
"""

__version__ = "0.1.0"
__all__ = [
    "chemistry",
    "data",
    "models",
    "engine",
    "utils",
]
