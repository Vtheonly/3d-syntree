"""Suite A / 2 — PoseBusters-related validity filters (issue #32 TEST-01).

Chemical-validity regression battery over everything the reaction engine
can emit (constitution #45 §12.5: no reward or filter ever replaces hard
chemical validity).  The battery implements the documented checks:

* **valence sanity** — every product of every certified reaction
  sanitizes with zero valence errors (RDKit fail-closed);
* **charge bounds** — per-atom formal charges stay within |q| <= 4 and
  whole molecules within |Q| <= 2 (documented V2 bounds for drug-like
  synthons);
* **unstable motifs** — peroxides (``[OX2][OX2]``) and cumulenes
  (``[CX2]=[CX2]=[CX2]``) are detected and flagged; no certified
  reaction product contains either motif.

Positive controls prove the detectors actually fire on known offenders
(hydrogen peroxide, allene, hypervalent nitrogen, extreme charge).
"""

import sys
from pathlib import Path

import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from syntree.chemistry.reactions import (  # noqa: E402
    CERTIFIED_REACTIONS,
    run_certified_reaction,
)

#: Documented V2 validity bounds (drug-like synthon scope).
MAX_ATOM_FORMAL_CHARGE = 4
MAX_MOLECULE_FORMAL_CHARGE = 2

PEROXIDE_SMARTS = Chem.MolFromSmarts("[OX2][OX2]")
CUMULENE_SMARTS = Chem.MolFromSmarts("[C]=[C]=[C]")


def valence_sane(mol: Chem.Mol) -> bool:
    """The molecule sanitizes cleanly (valence + aromaticity + rings)."""
    try:
        probe = Chem.Mol(mol)
        Chem.SanitizeMol(probe)
        return True
    except Exception:  # noqa: BLE001 - the check IS the exception test
        return False


def charge_ok(mol: Chem.Mol):
    """(atoms_ok, molecule_ok) against the documented charge bounds."""
    atoms_ok = all(
        abs(a.GetFormalCharge()) <= MAX_ATOM_FORMAL_CHARGE for a in mol.GetAtoms()
    )
    return atoms_ok, abs(Chem.GetFormalCharge(mol)) <= MAX_MOLECULE_FORMAL_CHARGE


def has_peroxide(mol: Chem.Mol) -> bool:
    return bool(mol.GetSubstructMatches(PEROXIDE_SMARTS))


def has_cumulene(mol: Chem.Mol) -> bool:
    return bool(mol.GetSubstructMatches(CUMULENE_SMARTS))


def _all_products():
    """Every product molecule of every certification pair (once each)."""
    seen = set()
    for reaction in CERTIFIED_REACTIONS.values():
        for pair in reaction.certification:
            for product in run_certified_reaction(reaction, pair.reagents):
                smi = Chem.MolToSmiles(product)
                if smi not in seen:
                    seen.add(smi)
                    yield product


def test_battery_fires_on_products():
    """The product corpus is non-trivial (the battery has teeth)."""
    products = list(_all_products())
    assert len(products) >= 30


@pytest.mark.parametrize("product", list(_all_products()), ids=lambda m: Chem.MolToSmiles(m))
def test_products_valence_sane(product):
    assert valence_sane(product), f"valence failure: {Chem.MolToSmiles(product)}"


@pytest.mark.parametrize("product", list(_all_products()), ids=lambda m: Chem.MolToSmiles(m))
def test_products_charge_bounded(product):
    atoms_ok, molecule_ok = charge_ok(product)
    assert atoms_ok and molecule_ok, (
        f"charge bound violation: {Chem.MolToSmiles(product)}"
    )


@pytest.mark.parametrize("product", list(_all_products()), ids=lambda m: Chem.MolToSmiles(m))
def test_products_no_unstable_motifs(product):
    assert not has_peroxide(product), f"peroxide motif: {Chem.MolToSmiles(product)}"
    assert not has_cumulene(product), f"cumulene motif: {Chem.MolToSmiles(product)}"


class TestDetectorPositiveControls:
    """The filters must detect known offenders (no silent pass-through)."""

    def test_peroxide_detected(self):
        for smi in ("OO", "COOC", "COOc1ccccc1"):
            mol = Chem.MolFromSmiles(smi)
            assert has_peroxide(mol), f"peroxide not detected: {smi}"

    def test_cumulene_detected(self):
        for smi in ("C=C=C", "CC=C=CC", "C=C=C(C)C"):
            mol = Chem.MolFromSmiles(smi)
            assert has_cumulene(mol), f"cumulene not detected: {smi}"

    def test_normal_ether_not_flagged_as_peroxide(self):
        mol = Chem.MolFromSmiles("COC")
        assert not has_peroxide(mol)

    def test_conjugated_diene_not_flagged_as_cumulene(self):
        mol = Chem.MolFromSmiles("C=CC=C")
        assert not has_cumulene(mol)

    def test_hypervalent_carbon_fails_valence(self):
        # a hexavalent carbon cannot sanitize (parsed with sanitize=False;
        # sanitization must fail closed)
        mol = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)
        assert mol is not None
        assert not valence_sane(mol)

    def test_extreme_charge_flagged(self):
        # per-atom bound: an atom with formal charge +6 (built via RWMol —
        # such species are not SMILES-expressible in this RDKit build)
        from rdkit import Chem as _Chem

        rw = _Chem.RWMol()
        rw.AddAtom(_Chem.Atom(6))
        rw.AddAtom(_Chem.Atom(1))
        rw.AddBond(0, 1, _Chem.BondType.SINGLE)
        rw.GetAtomWithIdx(0).SetFormalCharge(6)
        atoms_ok, _ = charge_ok(rw.GetMol())
        assert not atoms_ok

    def test_net_charge_beyond_bound_flagged(self):
        # whole-molecule bound: three ammonium centres, net charge +3 > 2
        mol = Chem.MolFromSmiles("[NH3+]C([NH3+])([NH3+])C")
        assert mol is not None
        _, molecule_ok = charge_ok(mol)
        assert not molecule_ok

    def test_reasonable_charges_pass(self):
        for smi in ("CC(=O)[O-]", "C[NH3+]", "C[N+](C)(C)C"):
            mol = Chem.MolFromSmiles(smi)
            assert valence_sane(mol)
            atoms_ok, molecule_ok = charge_ok(mol)
            assert atoms_ok and molecule_ok, f"false flag: {smi}"
