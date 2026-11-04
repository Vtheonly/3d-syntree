"""#34 EVAL-01 — the unlinked independent retrosynthesis evaluation library.

The evaluation stack of V2.  Issue #34 exists because claiming
retrosynthetic solvability *by definition* — "the molecule was assembled
by forward SMARTS, therefore it is synthesizable" — is circular
reasoning (#34 §1, "Vulnerability 9").  An independent retrosynthetic
planner may fail on generated molecules for stock reasons (the vendor
inventory does not contain a precursor) or for chemistry reasons
(unselective functional groups, missing template knowledge).  This module
makes that distinction **measurable**:

* :func:`load_generated_molecules` parses generated SDF artifacts
  (multi-record) into a fail-closed corpus with an explicit rejection
  vocabulary — malformed records are recorded with their rejection code,
  and unreadable corpus inputs raise immediately;
* :func:`run_aizynthfinder` drives an **isolated** AiZynthFinder
  subprocess (``aizynthcli``) with an externally supplied planner
  configuration and stock — the decoupling boundary #34 demands: the
  planner's template library, policy models and stock never come from
  the generator's own knowledge (the public USPTO-derived policy is the
  independent template corpus; the stocks are #44's two explicitly
  separated tracks);
* :func:`parse_aizynth_output` reads the planner's flat JSON result into
  :class:`RetroSearchResult` records, classifying each unsolved target as
  ``no_route_found`` (a genuine planner failure) or ``stock_mismatch``
  (routes exist but their leaves are not purchasable — #44's required
  distinction, never misreported as a planner failure);
* the physicochemical metric surface (chemical validity, Fsp³,
  molecular weight, fingerprint diversity) is computed with RDKit
  directly on the parsed corpus;
* the docking / structure-validity backends (Vina re-scoring through
  the #30 ``DockingResult`` contract; PoseBusters physical validity) are
  **optional, fail-closed adapters**: they activate only when the
  optional dependencies are importable and the required structure files
  exist, and they report their backend identity and version in every
  record (provenance; #45 §14).

Everything this module reports is evidence-labelled: the report schema
(:func:`build_evaluation_report`) carries the corpus description, the
tool/provenance block, and per-molecule records, so a reader can always
see *what* was measured, *with which stock*, and *by which planner*.

The module deliberately imports no torch: the evaluation path must stay
runnable in a light planner-side environment (the same interpreter that
runs ``aizynthcli``).
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
import rdkit.Chem.rdFingerprintGenerator as _fp_gen

#: Version of the evaluation report schema emitted by this module.
EVAL_SCHEMA_VERSION = "2.0.0"

#: Rejection vocabulary for SDF parsing (fail-closed, closed set).
REJECTION_CODES = frozenset(
    {
        "record_unparsable",
        "record_unsanitizable",
    }
)

#: Failure categories assigned to unsolved retrosynthesis targets.
RETRO_FAILURE_CATEGORIES = frozenset(
    {
        "no_route_found",
        "stock_mismatch",
    }
)

_SANITIZE_FLAGS = (
    Chem.SanitizeFlags.SANITIZE_ALL
    | Chem.SanitizeFlags.SANITIZE_PROPERTIES
    | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
)


class EvaluationError(ValueError):
    """Raised when the evaluation pipeline is asked to do something
    ill-defined (missing planner output, malformed report input,
    unusable corpus record).  Evaluation must fail loudly (#45 §16)."""


# ---------------------------------------------------------------------------
# 1. Corpus loading — parse generated SDF artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluatedMolecule:
    """One parsed generated molecule (one SDF record).

    Attributes:
        molecule_id: stable corpus identifier (source basename + record
            index; the SDF ``_Name`` is kept separately).
        source_path: the SDF artifact the record was read from.
        record_index: 0-based position within the SDF file.
        smiles: canonical isomeric SMILES (RDKit canonicalisation).
        inchi_key: the structure's InChIKey (stock bookkeeping key).
        num_atoms: number of heavy atoms.
        valid: the record sanitised under RDKit's strict valence rules.
        rejection_reason: one of ``REJECTION_CODES`` when ``valid`` is
            ``False``; ``None`` otherwise.
        name: the SDF ``_Name`` field when non-empty.
    """

    molecule_id: str
    source_path: str
    record_index: int
    smiles: str
    inchi_key: str
    num_atoms: int
    valid: bool
    rejection_reason: Optional[str]
    name: Optional[str] = None


def _parse_record(record: str, source: str, index: int) -> EvaluatedMolecule:
    mol = Chem.MolFromMolBlock(record, sanitize=False, removeHs=True)
    reason: Optional[str] = None
    if mol is None:
        reason = "record_unparsable"
    else:
        try:
            Chem.SanitizeMol(mol, _SANITIZE_FLAGS)
        except Exception:  # noqa: BLE001 — closed vocabulary above
            reason = "record_unsanitizable"
    if reason is not None:
        return EvaluatedMolecule(
            molecule_id=f"{os.path.basename(source)}#{index}",
            source_path=source,
            record_index=index,
            smiles="",
            inchi_key="",
            num_atoms=0,
            valid=False,
            rejection_reason=reason,
            name=None,
        )
    smiles = Chem.MolToSmiles(mol)
    inchi = MolToInchi(mol)
    inchi_key = InchiToInchiKey(inchi) if inchi else ""
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else None
    return EvaluatedMolecule(
        molecule_id=f"{os.path.basename(source)}#{index}",
        source_path=source,
        record_index=index,
        smiles=smiles,
        inchi_key=inchi_key,
        num_atoms=mol.GetNumAtoms(),
        valid=True,
        rejection_reason=None,
        name=(name or None),
    )


def load_generated_molecules(paths: Sequence[str]) -> List[EvaluatedMolecule]:
    """Parse generated SDF artifacts into a fail-closed corpus.

    Multi-record SDFs are supported (``.gz`` artifacts are transparently
    decompressed — the #18 bundle persists its ligand records gzipped);
    every record either sanitises or carries an explicit rejection code.
    Corpus inputs that cannot be read raise :class:`EvaluationError` — an
    evaluation must never run on a corpus whose inputs silently vanished
    (#45 §16 "fail loudly").
    """
    molecules: List[EvaluatedMolecule] = []
    for raw_path in paths:
        path = os.fspath(raw_path)
        if not os.path.exists(path):
            raise EvaluationError(f"generated SDF not found: {path}")
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt", errors="strict") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError) as err:
            raise EvaluationError(
                f"generated SDF is unreadable: {path} ({err})"
            ) from err
        if not text.strip():
            raise EvaluationError(f"generated SDF is empty: {path}")
        records = [r for r in text.split("$$$$\n") if r.strip()]
        if not records:
            raise EvaluationError(f"generated SDF has no MolBlock records: {path}")
        unparsable = 0
        for index, record in enumerate(records):
            molecule = _parse_record(record, path, index)
            if molecule.rejection_reason == "record_unparsable":
                unparsable += 1
            molecules.append(molecule)
        if unparsable == len(records):
            # every single record failed to *parse* (not merely to
            # sanitise): this is a wrong-file operator error, not a
            # chemistry-level rejection — fail loudly.
            raise EvaluationError(
                f"generated SDF has no parseable records: {path}"
            )
    if not molecules:
        raise EvaluationError("corpus is empty: no records parsed")
    return molecules


# ---------------------------------------------------------------------------
# 2. Physicochemical metrics (RDKit, no planner involved)
# ---------------------------------------------------------------------------


def molecular_metrics(smiles: str) -> Dict[str, Any]:
    """Per-molecule physicochemical summary of a valid SMILES.

    Molecular weight, fraction sp³ (the RDKit heavy-atom-carbon sp3
    ratio), rotatable bonds, ring count and net formal charge.  Raises
    :class:`EvaluationError` on an invalid SMILES: the caller routes
    invalid corpus records to the rejection statistics instead of
    measuring them.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise EvaluationError(f"invalid SMILES for metrics: {smiles!r}")
    return {
        "molecular_weight": round(float(Descriptors.MolWt(mol)), 4),
        "fraction_sp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "num_atoms": int(mol.GetNumAtoms()),
    }


def fingerprint_diversity(
    smiles_list: Sequence[str], radius: int = 2, fp_size: int = 2048
) -> float:
    """Mean pairwise topological Tanimoto distance over the corpus.

    Diversity is computed over *valid unique* molecules; duplicated
    SMILES collapse to one entry (a corpus that generates duplicates is
    *not* diverse — the uniqueness statistic itself carries that fact).
    Returns ``0.0`` for singletons — the honest value, not ``NaN``.
    """
    unique = sorted(set(s for s in smiles_list if s))
    if len(unique) < 2:
        return 0.0
    generator = _fp_gen.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fps = [
        generator.GetSparseFingerprint(Chem.MolFromSmiles(smi))
        for smi in unique
    ]
    total = 0.0
    pairs = 0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            total += DataStructs.TanimotoSimilarity(fps[i], fps[j])
            pairs += 1
    mean_similarity = total / pairs
    return float(1.0 - mean_similarity)


# ---------------------------------------------------------------------------
# 3. Retrosynthesis — the isolated AiZynthFinder boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetroSearchResult:
    """One AiZynthFinder search outcome for one target molecule.

    Attributes:
        target: the SMILES as written to the planner input.
        is_solved: the planner's own solved flag (a complete route with
            every leaf purchasable in the active stock).
        number_of_routes: total routes returned.
        number_of_solved_routes: routes with all leaves in stock.
        number_of_steps: the top route's retrosynthetic depth.
        search_time: the planner's measured wall-clock seconds.
        precursors_in_stock: leaf SMILES the stock contains.
        precursors_not_in_stock: leaf SMILES missing from the stock —
            the #44 stock-mismatch evidence.
        failure_category: ``None`` when solved; otherwise
            ``stock_mismatch`` (routes found, leaves missing from the
            stock) or ``no_route_found`` (the planner failed on the
            template/policy side — never conflated with stock issues).
    """

    target: str
    is_solved: bool
    number_of_routes: int
    number_of_solved_routes: int
    number_of_steps: int
    search_time: float
    precursors_in_stock: Tuple[str, ...]
    precursors_not_in_stock: Tuple[str, ...]
    failure_category: Optional[str]

    @property
    def stock_mismatch(self) -> bool:
        """True when unsolved *because* leaves are outside the stock."""
        return self.failure_category == "stock_mismatch"


def _classify_failure(row: Dict[str, Any]) -> Optional[str]:
    """Map a planner row onto the #44 failure vocabulary.

    A target with routes but unsolved, whose leaves are not purchasable,
    is a **stock mismatch** — it must be reported as such and never as a
    planner failure.  A target with no routes at all is a genuine
    planner failure (``no_route_found``).
    """
    if row.get("is_solved"):
        return None
    routes = int(row.get("number_of_routes") or 0)
    not_in_stock = row.get("precursors_not_in_stock") or ""
    if routes > 0 and not_in_stock.strip():
        return "stock_mismatch"
    return "no_route_found"


def parse_aizynth_output(path: str) -> Dict[str, RetroSearchResult]:
    """Parse the ``aizynthcli`` flat JSON output into result records.

    The planner writes ``{"schema": ..., "data": [row, ...]}`` with one
    row per target (the ``aizynthfinder`` >= 4 "pandas records" format).
    Rows missing mandatory keys raise :class:`EvaluationError` — a
    truncated or foreign-format planner output must never be silently
    interpreted (#45 §16).
    """
    with open(path, "r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "data" not in payload:
        raise EvaluationError(
            f"aizynthfinder output is not the flat records format: {path}"
        )
    results: Dict[str, RetroSearchResult] = {}
    for row in payload["data"]:
        for key in (
            "target",
            "is_solved",
            "number_of_routes",
            "number_of_solved_routes",
            "number_of_steps",
            "search_time",
        ):
            if key not in row:
                raise EvaluationError(
                    f"aizynthfinder row missing key {key!r} in {path}"
                )
        target = row["target"]
        results[target] = RetroSearchResult(
            target=target,
            is_solved=bool(row["is_solved"]),
            number_of_routes=int(row["number_of_routes"]),
            number_of_solved_routes=int(row["number_of_solved_routes"]),
            number_of_steps=int(row["number_of_steps"]),
            search_time=float(row["search_time"]),
            precursors_in_stock=tuple(
                s.strip()
                for s in (row.get("precursors_in_stock") or "").split(",")
                if s.strip()
            ),
            precursors_not_in_stock=tuple(
                s.strip()
                for s in (row.get("precursors_not_in_stock") or "").split(",")
                if s.strip()
            ),
            failure_category=_classify_failure(row),
        )
    return results


def run_aizynthfinder(
    smiles: Sequence[str],
    config_path: str,
    output_path: str,
    python_exe: Optional[str] = None,
    timeout_seconds: float = 600.0,
    extra_args: Sequence[str] = (),
) -> Dict[str, RetroSearchResult]:
    """Run the isolated AiZynthFinder CLI over a corpus of SMILES.

    The planner runs as a **subprocess** — the isolation boundary of
    #34: its process, its configuration, its stock, its own dependency
    stack; nothing in the generator's process is shared with it.  The
    SMILES corpus is written to a temporary file and passed via
    ``--smiles``; results land in ``output_path`` (the flat JSON).

    Raises :class:`EvaluationError` if the planner is not installed in
    ``python_exe``'s environment, exits non-zero, times out, or writes no
    output — never silently returns partial results.

    Returns the parsed per-target records (see :func:`parse_aizynth_output`).
    """
    if not smiles:
        raise EvaluationError("retrosynthesis target list is empty")
    exe = python_exe or sys.executable
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".smi", delete=False
    ) as smi_file:
        smi_file.write("\n".join(smiles) + "\n")
        smi_path = smi_file.name
    cmd = [
        exe,
        "-m",
        "aizynthfinder.interfaces.aizynthcli",
        "--config",
        os.path.abspath(config_path),
        "--smiles",
        smi_path,
        "--output",
        os.path.abspath(output_path),
        *extra_args,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as err:
        raise EvaluationError(
            f"aizynthfinder exceeded {timeout_seconds:.0f}s timeout"
        ) from err
    finally:
        os.unlink(smi_path)
    if proc.returncode != 0:
        raise EvaluationError(
            "aizynthfinder failed (exit "
            f"{proc.returncode}): {proc.stderr.strip()[:400]}"
        )
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise EvaluationError(
            f"aizynthfinder wrote no output to {output_path}"
        )
    return parse_aizynth_output(output_path)


# ---------------------------------------------------------------------------
# 4. Optional structure-validity / docking backends (fail-closed adapters)
# ---------------------------------------------------------------------------

#: The #30 docking contract types (imported lazily to keep this module
#: torch-free at import time — see module docstring).
DockingResult = Any


def vina_rescoring_backend(
    ligand_sdf_path: str,
    pocket_pdb_path: str,
    box_center: Sequence[float],
    box_size: Sequence[float] = (20.0, 20.0, 20.0),
    python_exe: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-score a posed ligand in a pocket with AutoDock Vina (optional).

    This is the evaluation-stack docking engine that **injects through
    the #30 ``DockingResult`` contract** at the reward level; here it is
    used as the reported affinity metric of #34/#35.  The receptor is
    prepared with meeko's ``mk_prepare_receptor`` (the documented
    PDB→PDBQT conversion) and the posed ligand with
    ``meeko.MoleculePreparation``; the Vina python API then computes the
    rigid pose's ``vina`` score in the supplied box.

    Fail-closed: the optional ``vina``/``meeko`` dependencies must be
    importable and the structure files must exist, otherwise
    :class:`EvaluationError` — the metric never silently degrades to a
    surrogate score under the Vina label.
    """
    if not os.path.exists(ligand_sdf_path):
        raise EvaluationError(f"ligand SDF not found: {ligand_sdf_path}")
    if not os.path.exists(pocket_pdb_path):
        raise EvaluationError(f"pocket PDB not found: {pocket_pdb_path}")
    try:
        import meeko  # noqa: F401
        from vina import Vina
    except ImportError as err:
        raise EvaluationError(
            "the vina/meeko optional backend is not installed — the "
            "docking metric cannot be computed in this environment"
        ) from err
    from rdkit import Chem

    mol = next(
        (m for m in Chem.SDMolSupplier(ligand_sdf_path, removeHs=False)),
        None,
    )
    if mol is None:
        raise EvaluationError(f"cannot parse ligand SDF: {ligand_sdf_path}")
    meeko_prep = meeko.MoleculePreparation()
    meeko_prep.prepare(mol)
    ligand_pdbqt = meeko_prep.write_pdbqt_string()

    # Receptor preparation: the documented meeko PDBQT conversion.
    exe = python_exe or sys.executable
    out_base = ligand_sdf_path + ".receptor"
    proc = subprocess.run(
        [
            exe,
            "-m",
            "meeko.cli.mk_prepare_receptor",
            "--read_pdb",
            os.path.abspath(pocket_pdb_path),
            "-o",
            out_base,
            "-p",
            out_base + ".pdbqt",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0 or not os.path.exists(out_base + ".pdbqt"):
        raise EvaluationError(
            "meeko receptor preparation failed for "
            f"{pocket_pdb_path}: {proc.stderr.strip()[:300]}"
        )
    vina_engine = Vina(sf_name="vina", verbosity=0)
    vina_engine.set_receptor(out_base + ".pdbqt")
    vina_engine.compute_vina_maps(
        center=list(map(float, box_center)),
        box_size=list(map(float, box_size)),
    )
    vina_engine.set_ligand_from_string(ligand_pdbqt)
    score = float(vina_engine.score())
    return {
        "affinity_kcal_mol": score,
        "backend": "autodock_vina",
        "ligand": ligand_sdf_path,
        "pocket": pocket_pdb_path,
        "box_center": [float(x) for x in box_center],
    }


def posebusters_backend(
    ligand_sdf_path: str,
    reference_sdf_path: Optional[str] = None,
    config_name: str = "mol",
) -> Dict[str, Any]:
    """Run a PoseBusters physical-validity check (optional, fail-closed).

    Uses the ``posebusters`` package with the named configuration
    (``posebusters`` 0.6 conventions): ``mol`` busts the ligand's own 3D
    chemical geometry (sanitisation, radicals, bond lengths/angles, ring
    flatness, internal clash/energy); ``redock`` additionally compares
    against ``reference_sdf_path`` (the crystal/reference ligand).  The
    corpus convention is one single-record SDF per artifact.  Returns
    the per-bust pass table and the overall pass flag; unavailable
    dependencies raise :class:`EvaluationError`.
    """
    if not os.path.exists(ligand_sdf_path):
        raise EvaluationError(f"ligand SDF not found: {ligand_sdf_path}")
    if (
        reference_sdf_path is not None
        and not os.path.exists(reference_sdf_path)
    ):
        raise EvaluationError(
            f"reference ligand SDF not found: {reference_sdf_path}"
        )
    try:
        from posebusters import PoseBusters
        import posebusters as posebusters_pkg
    except ImportError as err:
        raise EvaluationError(
            "the posebusters optional backend is not installed — the "
            "physical-validity metric cannot be computed here"
        ) from err
    buster = PoseBusters(config_name)
    outcome = buster.bust(
        mol_pred=ligand_sdf_path, mol_true=reference_sdf_path
    )
    if outcome.empty:
        raise EvaluationError(
            f"posebusters returned no result row for {ligand_sdf_path}"
        )
    row = outcome.iloc[0].to_dict()
    passed = all(bool(v) for k, v in row.items() if k != "num_errors")
    return {
        "passed": passed,
        "busts": {k: bool(v) for k, v in row.items() if k != "num_errors"},
        "backend": "posebusters",
        "posebusters_version": getattr(
            posebusters_pkg, "__version__", "unknown"
        ),
    }


# ---------------------------------------------------------------------------
# 5. Report assembly
# ---------------------------------------------------------------------------


def build_evaluation_report(
    corpus: List[EvaluatedMolecule],
    retro_tracks: Dict[str, Dict[str, RetroSearchResult]],
    track_provenance: Dict[str, Dict[str, Any]],
    corpus_description: str,
    forward_replay_status: Optional[Dict[str, Any]] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the ``evaluation_report.json`` payload.

    Parameters:
        corpus: the parsed generated-molecule corpus.
        retro_tracks: ``{track_name: {target_smiles: result}}`` — the
            retrosynthesis records per track (#44: exactly one entry per
            separately-labelled track; the caller never merges them and
            this function renders them as separate top-level sections).
        track_provenance: per track: stock identity, stock checksum,
            planner configuration, versions — the #44 reproducibility
            payload.
        corpus_description: the honest label of what the corpus *is*
            (e.g. "forward-replay-verified reconstructed crystal ligands
            of the #18 census — NOT trained-policy samples").
        forward_replay_status: what the internal forward-reaction replay
            established about this corpus (#44 requires the report to
            distinguish replay from planner retrosynthesis).
        extra_metrics: optional additional sections (docking, posebusters)
            with their own backend provenance.

    The aggregate section computes per-track solve rates, failure
    category counts, validity and Fsp³ statistics, and diversity — all
    from the records, never from caller-supplied numbers.
    """
    if retro_tracks and set(retro_tracks) != set(track_provenance):
        raise EvaluationError(
            "every retrosynthesis track must carry provenance "
            f"(tracks={sorted(retro_tracks)}, "
            f"provenance={sorted(track_provenance)})"
        )
    valid = [m for m in corpus if m.valid]
    metrics = [molecular_metrics(m.smiles) for m in valid]
    report: Dict[str, Any] = {
        "schema": EVAL_SCHEMA_VERSION,
        "corpus": {
            "description": corpus_description,
            "num_records": len(corpus),
            "num_valid": len(valid),
            "validity_rate": (len(valid) / len(corpus)) if corpus else 0.0,
            "rejections": {
                code: sum(
                    1 for m in corpus if m.rejection_reason == code
                )
                for code in sorted(REJECTION_CODES)
            },
            "forward_replay": forward_replay_status,
            "smiles": [
                {
                    "molecule_id": m.molecule_id,
                    "smiles": m.smiles,
                    "inchi_key": m.inchi_key,
                    "num_atoms": m.num_atoms,
                    "source": m.source_path,
                    "record_index": m.record_index,
                }
                for m in corpus
            ],
        },
        "physicochemical": {
            "fraction_sp3": {
                "mean": (
                    sum(m["fraction_sp3"] for m in metrics) / len(metrics)
                    if metrics
                    else 0.0
                ),
                "min": min((m["fraction_sp3"] for m in metrics), default=0.0),
                "max": max((m["fraction_sp3"] for m in metrics), default=0.0),
            },
            "molecular_weight": {
                "mean": (
                    sum(m["molecular_weight"] for m in metrics) / len(metrics)
                    if metrics
                    else 0.0
                ),
                "min": min((m["molecular_weight"] for m in metrics), default=0.0),
                "max": max((m["molecular_weight"] for m in metrics), default=0.0),
            },
            "fingerprint_diversity": fingerprint_diversity(
                [m.smiles for m in valid]
            ),
            "num_unique_valid_smiles": len({m.smiles for m in valid}),
        },
        "retrosynthesis": {},
        "extra": extra_metrics or {},
    }
    for track_name, results in retro_tracks.items():
        targets = list(results)
        solved = [r for r in results.values() if r.is_solved]
        mismatches = [r for r in results.values() if r.stock_mismatch]
        report["retrosynthesis"][track_name] = {
            "provenance": track_provenance[track_name],
            "num_targets": len(targets),
            "num_solved": len(solved),
            "solve_rate": (len(solved) / len(targets)) if targets else 0.0,
            "num_stock_mismatch": len(mismatches),
            "num_no_route_found": len(targets) - len(solved) - len(mismatches),
            "mean_search_time_s": (
                sum(r.search_time for r in results.values()) / len(results)
                if results
                else 0.0
            ),
            "results": {
                r.target: {
                    "is_solved": r.is_solved,
                    "number_of_routes": r.number_of_routes,
                    "number_of_solved_routes": r.number_of_solved_routes,
                    "number_of_steps": r.number_of_steps,
                    "search_time": r.search_time,
                    "precursors_in_stock": list(r.precursors_in_stock),
                    "precursors_not_in_stock": list(
                        r.precursors_not_in_stock
                    ),
                    "failure_category": r.failure_category,
                }
                for r in results.values()
            },
        }
    return report


__all__ = [
    "EVAL_SCHEMA_VERSION",
    "REJECTION_CODES",
    "RETRO_FAILURE_CATEGORIES",
    "EvaluationError",
    "EvaluatedMolecule",
    "RetroSearchResult",
    "load_generated_molecules",
    "molecular_metrics",
    "fingerprint_diversity",
    "parse_aizynth_output",
    "run_aizynthfinder",
    "vina_rescoring_backend",
    "posebusters_backend",
    "build_evaluation_report",
]
