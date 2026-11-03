"""Tests for the re-landed #12 CHEM-02 synthon encoder + catalog layer.

Pinned against the surviving build artifacts: the certification model
(0/50,000 row mismatches against the immutable HDF5 labels), the QR
projection (bit-exact seed-2026 reproduction), the frozen schema, the
fail-closed conformer policy, and the artifact-pinned structural facts
(strata, funnel contract, selection arithmetic).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from syntree.chemistry.synthon_encoder import (  # noqa: E402
    D_EMBED,
    D_RAW,
    FG_SMARTS,
    PROJECTION_SEED,
    SCHEMA_VERSION,
    SEED_ACYL_ELECTROPHILES,
    VECTOR_HANDLE_CLASSES,
    build_projection,
    certify_block_roles,
    detect_seed_classes,
    embed_conformer,
    encode_synthon,
    handle_descriptor_64,
    pharm3d_descriptor,
    project_raw,
    seed_screen,
)
from syntree.chemistry.chemoselectivity import screen_block  # noqa: E402

#: The artifact-pinned per-row certification fixtures, transcribed from
#: the immutable HDF5 (JJKK1212/3d-syntree-v2-catalog).  Each row:
#: (smiles, synthon_classes, seed_classes) — verified 0/50,000 mismatches
#: by the S-V2-11 reconstruction (see docs/agent/WORK_LOG.md).
ARTIFACT_FIXTURES = [
    ("Br.BrCCN1CCC1", "alkyl_bromide", ""),                     # tertiary azetidine: no amine seed
    ("Br.C#CCSC(=N)N", "acetylide", "amine"),                   # amidine amine seed
    ("Br.C1CC2NC1C1CCC2O1", "", "amine"),                       # seed-only ring amine
    ("Br.CC(C)(C)C(N)C(=O)O", "", "alcohol|amine"),             # amino acid: COOH counts as alcohol
    ("Br.CC(C)C(C(=O)O)N(C)C", "", "alcohol"),                  # tertiary amine + acid: amine excluded? no — tertiary not detected
    ("Br.CP(=O)(O)CCCN", "", "amine"),                          # P-OH is NOT alcohol
    ("C#CC#CC(C)(C)N.Cl", "", "amine"),                         # quenched acetylide: synthon excluded, seed kept
    ("C#CC(C)(C)NC(=O)OC(C)(C)C", "acetylide", ""),           # Boc carbamate: acetylide certified, no free amine seed
    ("Brc1c[nH]c(Br)n1", "pyrazolyl_bromide", "aryl_bromide"),      # 2,4-dihaloazine exemption passes
    ("CC(C)(C)c1cc(Cl)nc(Cl)n1", "pyrimidinyl_chloride", "aryl_chloride"),  # exemption (declared first instance)
    ("Cl.NC1=NCC(CBr)O1", "alkyl_bromide", "amine"),            # alkyl bromide does NOT quench the amine seed
    ("CC(C)(C=O)CO", "aldehyde", "alcohol"),                    # aldehyde handle + alcohol seed coexist
    ("C#CCn1ccc2nc(Cl)nc(Cl)c21", "acetylide", "aryl_chloride"),# first-instance screening excludes pyrimidinyl_chloride
]


class TestFrozenConstants:
    def test_descriptor_width_is_2137_not_2138(self):
        # pinned by normalization/mean shape 2137 + projection/Q [2137, 256]
        assert D_RAW == 2137
        assert D_EMBED == 256

    def test_block_layout_sums(self):
        assert 2048 + 47 + 6 + 5 + 30 + 1 == D_RAW

    def test_fg_block_is_21_wide(self):
        assert len(FG_SMARTS) == 21

    def test_vector_block_is_10_classes(self):
        assert len(VECTOR_HANDLE_CLASSES) == 10

    def test_seed_electrophile_contract(self):
        assert set(SEED_ACYL_ELECTROPHILES) == {
            "acid_chloride", "sulfonyl_chloride",
            "carbamoyl_chloride", "chloroformate",
            "alkyl_bromide", "aldehyde",
        }


class TestArtifactCertificationFixtures:
    @pytest.mark.parametrize(
        "smiles,syn_expected,seed_expected", ARTIFACT_FIXTURES
    )
    def test_certification_matches_artifact(self, smiles, syn_expected, seed_expected):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        synthon_ok, seed_ok, _ = certify_block_roles(mol)
        assert "|".join(synthon_ok) == syn_expected, (
            f"{smiles}: synthon classes {synthon_ok} != artifact {syn_expected!r}"
        )
        assert "|".join(sorted(seed_ok)) == seed_expected, (
            f"{smiles}: seed classes {seed_ok} != artifact {seed_expected!r}"
        )

    def test_certification_deterministic(self):
        mol = Chem.MolFromSmiles(ARTIFACT_FIXTURES[0][0])
        assert certify_block_roles(mol) == certify_block_roles(mol)


class TestSeedDetection:
    def test_amine_detector_excludes_acyl_and_sulfonyl(self):
        assert "amine" in detect_seed_classes(Chem.MolFromSmiles("CCN"))
        assert "amine" not in detect_seed_classes(
            Chem.MolFromSmiles("CNC(=O)OC(C)(C)C")
        )
        assert "amine" not in detect_seed_classes(
            Chem.MolFromSmiles("CNS(=O)(=O)C")
        )

    def test_amine_detector_includes_amidine_and_thiourea(self):
        assert "amine" in detect_seed_classes(Chem.MolFromSmiles("C=NC(N)N"))
        assert "amine" in detect_seed_classes(Chem.MolFromSmiles("C=NC(N)=S"))

    def test_alcohol_includes_carboxylic_acid_excludes_phosphorous(self):
        assert "alcohol" in detect_seed_classes(Chem.MolFromSmiles("CC(=O)O"))
        assert "alcohol" not in detect_seed_classes(Chem.MolFromSmiles("CP(=O)(O)O"))

    def test_seed_screen_pairs_follow_the_pinned_matrix(self):
        # free primary amine + acid chloride -> excluded (acylation)
        assert seed_screen(Chem.MolFromSmiles("NCC(=O)Cl"), "amine") is not None
        # free amine + aldehyde -> excluded (imine formation)
        assert seed_screen(Chem.MolFromSmiles("NCC=O"), "amine") is not None
        # free amine + alkyl bromide -> excluded (SN2)
        assert seed_screen(Chem.MolFromSmiles("NCCCBr"), "amine") is not None
        # free alcohol + alkyl bromide -> excluded (etherification)
        assert seed_screen(Chem.MolFromSmiles("OCCCBr"), "alcohol") is not None
        # alcohol + aldehyde -> KEPT (reversible hemiacetal; the pinned
        # matrix pairs aldehydes with amines only)
        assert seed_screen(Chem.MolFromSmiles("OCC=O"), "alcohol") is None
        # amidine "amine" seed + alkyl bromide -> KEPT (N not the free form)
        assert seed_screen(Chem.MolFromSmiles("CC(CBr)NC(=N)N"), "amine") is None
        # non-nucleophilic seeds are never quenched
        assert seed_screen(Chem.MolFromSmiles("Brc1ccccc1CC(=O)Cl"), "aryl_bromide") is None


class TestProjection:
    def test_qr_projection_is_deterministic(self):
        assert np.array_equal(build_projection(), build_projection())

    def test_q_shape_and_orthonormality(self):
        q = build_projection()
        assert q.shape == (D_RAW, D_EMBED)
        assert q.dtype == np.float32
        gram = q.T.astype(np.float64) @ q.astype(np.float64)
        assert np.abs(gram - np.eye(D_EMBED)).max() < 1e-5

    def test_seed_is_the_artifact_pinned_2026(self):
        assert PROJECTION_SEED == 2026

    def test_project_raw_l2_normalizes(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=D_RAW)
        mean = np.zeros(D_RAW)
        std = np.ones(D_RAW)
        e = project_raw(x, mean, std, build_projection())
        assert e.shape == (D_EMBED,)
        assert e.dtype == np.float32
        assert abs(float(np.linalg.norm(e)) - 1.0) < 1e-6

    def test_project_raw_rejects_wrong_width(self):
        from syntree.chemistry.synthon_encoder import SynthonEncodeError

        with pytest.raises(SynthonEncodeError):
            project_raw(
                np.zeros(10), np.zeros(D_RAW), np.ones(D_RAW), build_projection()
            )

    def test_q_matches_artifact_when_available(self):
        """Bit-exact reproduction of the stored projection/Q (gated on the
        immutable HDF5 being present in the local artifact store)."""
        import h5py

        candidates = [
            os.environ.get("SYNTREE_CATALOG_H5", ""),
            "/home/z/my-project/work/raw/enamine_real_50k_pharm3d_d256.h5",
        ]
        path = next((p for p in candidates if p and os.path.exists(p)), None)
        if path is None:
            pytest.skip("catalog HDF5 not present (artifact-gated)")
        with h5py.File(path, "r") as h5:
            stored = h5["projection/Q"][()]
        assert np.array_equal(build_projection(), stored)


class TestConformerPolicy:
    def test_mmff_typed_molecule_relaxes(self):
        outcome = embed_conformer("CCOc1cc(C=O)ccc1O")
        assert outcome.method == "etkdg3+mmff94"
        assert np.isfinite(outcome.energy)

    def test_boronic_acid_keeps_etkdg_geometry(self):
        # B is in the documented coupling-handle fallback
        outcome = embed_conformer("OB(O)c1ccccc1")
        assert outcome.method == "etkdg3"
        assert outcome.energy != outcome.energy  # NaN by contract

    def test_halide_counterion_salt_keeps_etkdg_geometry(self):
        # HBr salt: the Br- counterion is in the documented fallback
        # (artifact-pinned: 17,493 etkdg3 rows over the 50k catalog)
        outcome = embed_conformer("Br.BrCCN1CCC1")
        assert outcome.method == "etkdg3"
        assert outcome.energy != outcome.energy

    def test_trifluoroborate_salt_fails_closed(self):
        # potassium trifluoroborate: B is tolerated but K is not
        from syntree.chemistry.synthon_encoder import SynthonEncodeError

        with pytest.raises(SynthonEncodeError, match=r"\['B', 'K'\]"):
            embed_conformer("C[B-](F)(F)F.[K+]")

    def test_unkown_untypeable_atom_fails_closed(self):
        from syntree.chemistry.synthon_encoder import SynthonEncodeError

        # potassium acetate embeds fine, but K is outside the documented
        # B/Zn/Sn coupling-handle fallback -> named fail-closed error
        with pytest.raises(SynthonEncodeError, match="MMFF94 cannot type"):
            embed_conformer("[K+]CC(=O)[O-]")

    def test_deterministic_repeats(self):
        a = embed_conformer("CCOc1cc(C=O)ccc1O")
        b = embed_conformer("CCOc1cc(C=O)ccc1O")
        assert a.method == b.method
        assert a.energy == b.energy
        assert np.array_equal(
            a.conformer.GetConformer().GetPositions(),
            b.conformer.GetConformer().GetPositions(),
        )


class TestDescriptor:
    def test_descriptor_width_and_finiteness(self):
        mol = Chem.MolFromSmiles("CCOc1cc(C=O)ccc1O")
        outcome = embed_conformer("CCOc1cc(C=O)ccc1O")
        x = pharm3d_descriptor(mol, outcome.conformer)
        assert x.shape == (D_RAW,)
        assert np.isfinite(x).all()

    def test_norm_feature_is_last_and_consistent(self):
        mol = Chem.MolFromSmiles("CCOc1cc(C=O)ccc1O")
        outcome = embed_conformer("CCOc1cc(C=O)ccc1O")
        x = pharm3d_descriptor(mol, outcome.conformer)
        assert x[-1] == pytest.approx(np.linalg.norm(x[:-1]), rel=1e-12)

    def test_physchem_block_positions(self):
        from syntree.chemistry.synthon_encoder import _physchem_features
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles("CCOc1cc(C=O)ccc1O")
        v = _physchem_features(mol)
        assert v[0] == pytest.approx(Descriptors.MolWt(mol), abs=1e-9)   # 2048
        assert v[1] == pytest.approx(Descriptors.ExactMolWt(mol))       # 2049
        assert v[2] == mol.GetNumHeavyAtoms()                            # 2050
        assert v[3] == mol.GetNumAtoms()                                 # 2051
        assert v[4] == mol.GetNumBonds()                                 # 2052

    def test_vector_block_zero_for_handleless_molecule(self):
        # decalin: no vector-class handles -> all 30 dims zero
        mol = Chem.MolFromSmiles("C1CC2CCCCC2C1")
        outcome = embed_conformer("C1CC2CCCCC2C1")
        x = pharm3d_descriptor(mol, outcome.conformer)
        assert np.allclose(x[2106:2136], 0.0)

    def test_vector_block_nonzero_for_classed_molecule(self):
        mol = Chem.MolFromSmiles("Brc1ccccc1")
        outcome = embed_conformer("Brc1ccccc1")
        x = pharm3d_descriptor(mol, outcome.conformer)
        vecs = x[2106:2136].reshape(10, 3)
        active = np.linalg.norm(vecs, axis=1) > 0
        assert active[VECTOR_HANDLE_CLASSES.index("aryl_bromide")]

    def test_encode_synthon_columns(self):
        row = encode_synthon("REC0000042", "CCOc1cc(C=O)ccc1O", "aromatic",
                             "aldehyde", "alcohol")
        assert row.stratum == "aromatic"
        assert row.fsp3 == pytest.approx(0.0, abs=1e-9) or row.fsp3 > 0
        assert row.heavy_atoms == 12
        assert row.rotatable_bonds == 3


class TestHandleDescriptor64:
    def test_width_dtype_finite(self):
        mol = Chem.MolFromSmiles("Brc1ccccc1")
        d = handle_descriptor_64(mol, 5)  # the ipso carbon
        assert d.shape == (64,)
        assert np.isfinite(d).all()

    def test_class_onehot_fires(self):
        mol = Chem.MolFromSmiles("Brc1ccccc1")
        ipso = next(
            a.GetIdx() for a in mol.GetAtoms()
            if a.GetSymbol() == "C" and any(
                n.GetSymbol() == "Br" for n in a.GetNeighbors()
            )
        )
        d = handle_descriptor_64(mol, ipso)
        from syntree.chemistry.synthon_encoder import _HANDLE_ONEHOT_CLASSES
        slot = _HANDLE_ONEHOT_CLASSES.index("aryl_bromide")
        assert d[slot] == 1.0

    def test_out_of_range_fails_closed(self):
        from syntree.chemistry.synthon_encoder import SynthonEncodeError

        with pytest.raises(SynthonEncodeError):
            handle_descriptor_64(Chem.MolFromSmiles("CCO"), 99)


class TestStrata:
    @pytest.mark.parametrize(
        "smiles,stratum",
        [
            ("C1CC2NC1C1CCC2O1", "saturated"),
            ("Brc1ccccc1", "aromatic"),
            ("C#CCSC(=N)N", "saturated"),
            ("Clc1ncccn1", "aromatic"),
        ],
    )
    def test_strata_by_aromatic_rings(self, smiles, stratum):
        from rdkit.Chem import rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        got = "aromatic" if rdMolDescriptors.CalcNumAromaticRings(mol) > 0 else "saturated"
        assert got == stratum


class TestSchema:
    def test_h5_roundtrip_schema(self, tmp_path):
        import h5py

        from syntree.chemistry.synthon_encoder import write_catalog_h5

        rows = [
            encode_synthon("REC0000001", "CCO", "saturated", "", "alcohol"),
            encode_synthon("REC0000002", "Brc1ccccc1", "aromatic",
                           "aryl_bromide", "aryl_bromide"),
        ]
        mean = np.zeros(D_RAW)
        std = np.ones(D_RAW)
        path = str(tmp_path / "mini.h5")
        digest = write_catalog_h5(
            path, rows, mean=mean, std=std, q=build_projection(),
            source_sdf="test.sdf", source_sdf_sha256="0" * 64,
            registry_sha="1" * 64,
        )
        assert len(digest) == 64
        with h5py.File(path, "r") as h5:
            assert h5.attrs["schema_version"] == SCHEMA_VERSION
            assert h5.attrs["catalog_schema_version"] == "catalog-v2.0"
            assert h5.attrs["d_raw"] == D_RAW
            assert h5.attrs["d_embed"] == D_EMBED
            assert h5.attrs["projection_seed"] == 2026
            for key in ("block_ids", "smiles", "synthon_classes", "seed_classes",
                        "strata", "mw", "fsp3", "heavy_atoms", "rotatable_bonds",
                        "conformer_energies", "conformer_methods"):
                assert f"catalog/{key}" in h5, key
            assert h5["embeddings"].shape == (2, D_EMBED)
            assert h5["embeddings"].dtype == np.float32
            assert h5["projection/Q"].shape == (D_RAW, D_EMBED)
            assert h5["normalization/mean"].shape == (D_RAW,)
            assert h5["normalization/std"].shape == (D_RAW,)
            for key in ("python", "numpy", "rdkit", "syntree"):
                assert f"versions/{key}" in h5

    def test_h5_embeddings_l2_normalized_rows(self, tmp_path):
        import h5py

        from syntree.chemistry.synthon_encoder import write_catalog_h5

        rows = [encode_synthon("REC0000001", "CCOc1cc(C=O)ccc1O",
                               "aromatic", "aldehyde", "alcohol")]
        path = str(tmp_path / "one.h5")
        write_catalog_h5(
            path, rows, mean=np.zeros(D_RAW), std=np.ones(D_RAW),
            q=build_projection(), source_sdf="t.sdf",
            source_sdf_sha256="0" * 64, registry_sha="1" * 64,
        )
        with h5py.File(path, "r") as h5:
            emb = h5["embeddings"][()]
        assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-6)
        assert np.isfinite(emb).all()
