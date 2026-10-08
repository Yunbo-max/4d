"""C04 anchored ellipsoidal motion protection; generated, not scientifically admitted.

The deterministic uncertainty set concerns the gradient of a declared predicted-
mesh kinetic-energy surrogate. It is not a distribution, confidence region, native
metric gradient or guarantee. Shape and budget are prospectively frozen inputs.
All arms share the same C02 geometry repair. Only the robust arm uses its conic
constraint. Float32 trust/backtracking is reported separately from projection.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import time
import numpy as np
from research_math.protected_geometry_candidate import (
    validate_native_arrays, compute_common_geometry_repair,
    digest as _digest, write_json as _write_json,
    _positive, _positive_integer,
)
CANDIDATE_ID = "4d-math-20261006-c04"
ROLES = ("deterministic_protection", "strength_matched_repair", "robust_conic_protection")
METHOD_IDS = {ROLES[0]: "c04-control-deterministic-protection-v1",
              ROLES[1]: "c04-control-strength-matched-repair-v1", ROLES[2]: CANDIDATE_ID}
ROLE_CONTRACTS = {role: role for role in ROLES}
FLOAT_PARAMETERS = ("arap_weight", "temporal_weight", "cg_tolerance", "shape_floor",
                    "shape_gain", "radius", "epsilon", "trust_radius", "finite_budget",
                    "absolute_tolerance", "relative_tolerance")
INTEGER_PARAMETERS = ("iterations", "cg_max_iterations", "max_iterations", "max_backtracks")
SCOPE = "C04 generated_unexecuted three-arm artifact; no B0/B*/scoring/admission"


def validate_parameters(parameters):
    if set(parameters) != set(FLOAT_PARAMETERS + INTEGER_PARAMETERS):
        raise ValueError("Exact complete C04 parameter inventory required")
    return {**{k: _positive(k, parameters[k]) for k in FLOAT_PARAMETERS},
            **{k: _positive_integer(k, parameters[k]) for k in INTEGER_PARAMETERS}}


def motion_surrogate(vertices, times, *, shape_floor, shape_gain):
    """f=.5/(V duration) sum ||x[t+1]-x[t]||²/dt; exact gradient/Hessian.

    Diagonal S repeats a per-frame/per-vertex acceleration magnitude along XYZ.
    Positive floor excludes singular uncertainty axes in this specialization.
    Normalize acceleration by its global RMS; zero acceleration gives isotropy.
    This is deterministic sensitivity weighting, with no calibrated coverage.
    """
    x = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    shape_floor = _positive("shape_floor", shape_floor)
    shape_gain = _positive("shape_gain", shape_gain)
    if x.ndim != 3 or x.shape[-1] != 3 or t.shape != (len(x),) or len(x) < 3:
        raise ValueError("Motion surrogate requires complete XYZ frames and timestamps")
    dt = np.diff(t)
    if not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(dt <= 0):
        raise ValueError("Finite vertices and strictly increasing timestamps required")
    normalization = float(x.shape[1] * (t[-1] - t[0]))
    w = 1.0 / (dt * normalization)
    difference = np.diff(x, axis=0)
    gradient = np.zeros_like(x)
    gradient[:-1] -= w[:, None, None] * difference
    gradient[1:] += w[:, None, None] * difference
    hessian = np.diag(np.r_[w, 0.] + np.r_[0., w])
    hessian += np.diag(-w, 1) + np.diag(-w, -1)
    # Exact temporal Hessian restricted to the zero-update anchor subspace.
    lipschitz = float(np.linalg.eigvalsh(hessian[1:, 1:])[-1])
    velocity = difference / dt[:, None, None]
    acceleration = np.diff(velocity, axis=0) / ((dt[:-1]+dt[1:])/2)[:, None, None]
    magnitude = np.linalg.norm(acceleration, axis=-1)
    magnitude = np.concatenate((magnitude[:1], magnitude, magnitude[-1:]), axis=0)
    rms = float(np.sqrt(np.mean(magnitude * magnitude)))
    normalized = magnitude / rms if rms > 0 else np.zeros_like(magnitude)
    shape = np.broadcast_to((shape_floor + shape_gain * normalized)[..., None], x.shape).copy()
    gradient[0] = 0.0
    return {"g0": gradient, "shape_diagonal": shape, "temporal_hessian": hessian,
            "lipschitz": lipschitz, "normalization": normalization,
            "value": float(.5*np.sum(w[:, None, None]*difference*difference))}


def ellipsoid_support(delta, g0, shape_diagonal, radius):
    return float(abs(np.sum(g0*delta)) + radius*np.sqrt(np.sum(shape_diagonal*delta*delta)))


def _prox_support(d, g, s, lam, radius, max_iterations, tolerance):
    """Prox of lambda (abs(g.x)+r||sqrt(S)x||), positive diagonal S.

    Clipped alpha is the exact minimizer for the absolute-value dual at fixed
    beta. beta=lam*r/||sqrt(S)x|| is solved monotonically, without dense S.
    """
    if lam == 0:
        return d.copy(), 0., np.zeros_like(d)
    if radius == 0:
        gg = float(np.dot(g, g))
        alpha = float(np.clip(np.dot(g, d)/gg, -lam, lam)) if gg else 0.
        return d-alpha*g, alpha, np.zeros_like(d)
    gg_inverse = float(np.sum(g*g/s))
    alpha_zero = float(np.clip(np.sum(g*d/s)/gg_inverse, -lam, lam)) if gg_inverse else 0.
    z = (d-alpha_zero*g)/np.sqrt(s)
    if np.linalg.norm(z) <= lam*radius:
        return np.zeros_like(d), alpha_zero, z/(lam*radius)
    def at(beta):
        inverse = 1./(1.+beta*s)
        gg = float(np.sum(g*g*inverse))
        alpha = float(np.clip(np.sum(g*d*inverse)/gg, -lam, lam)) if gg else 0.
        x = (d-alpha*g)*inverse
        q = float(np.sqrt(np.sum(s*x*x)))
        return x, alpha, q
    lo, hi = 0., 1.
    for _ in range(max_iterations):
        x, alpha, q = at(hi)
        if hi*q >= lam*radius:
            break
        hi *= 2.
        if not np.isfinite(hi):
            raise RuntimeError("Conic inner bracket overflow")
    else:
        raise RuntimeError("Conic inner bracket exhausted")
    for _ in range(max_iterations):
        beta = (lo+hi)/2
        x, alpha, q = at(beta)
        residual = beta*q-lam*radius
        if abs(residual) <= tolerance*max(1.,lam*radius):
            return x, alpha, np.sqrt(s)*x/q
        if residual > 0:
            hi = beta
        else:
            lo = beta
    raise RuntimeError("Conic inner solve did not converge")


def solve_anchored_conic_projection(desired, g0, shape_diagonal, *, radius,
                                     epsilon, absolute_tolerance,
                                     relative_tolerance, max_iterations):
    """Euclidean projection with exact frame-zero anchor and primal/dual gap."""
    arrays = [np.asarray(v, dtype=np.float64) for v in (desired,g0,shape_diagonal)]
    d_full,g_full,s_full = arrays
    if any(a.shape != d_full.shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Conic arrays must have matching finite shapes")
    if d_full.ndim != 3 or d_full.shape[-1] != 3 or np.any(d_full[0] != 0):
        raise ValueError("Complete anchored XYZ desired update required")
    if np.any(s_full <= 0):
        raise ValueError("This declared diagonal specialization requires positive shape floor")
    if isinstance(radius,bool) or not np.isfinite(radius) or radius < 0:
        raise ValueError("Finite nonnegative radius required")
    epsilon = _positive("epsilon",epsilon)
    atol = _positive("absolute_tolerance",absolute_tolerance)
    rtol = _positive("relative_tolerance",relative_tolerance)
    max_iterations = _positive_integer("max_iterations",max_iterations)
    d,g,s = (a[1:].reshape(-1) for a in arrays)
    support = lambda x: ellipsoid_support(x,g,s,radius)
    lam, alpha, u, x = 0.,0.,np.zeros_like(d),d.copy()
    outer_iterations = 0
    if support(d) > epsilon:
        lo,hi = 0.,1.
        for _ in range(max_iterations):
            trial,_,_ = _prox_support(d,g,s,hi,radius,max_iterations,atol*.01)
            if support(trial) <= epsilon:
                break
            hi *= 2.
            if not np.isfinite(hi):
                raise RuntimeError("Conic outer bracket overflow")
        else:
            raise RuntimeError("Conic outer bracket exhausted")
        # Keep the feasible side so every accepted certificate is primal feasible.
        for outer_iterations in range(1,max_iterations+1):
            lam=hi
            x,alpha,u = _prox_support(d,g,s,lam,radius,max_iterations,atol*.01)
            y=alpha*g+lam*radius*np.sqrt(s)*u
            primal=.5*float(np.sum((x-d)**2))
            dual=float(np.dot(d,y)-.5*np.dot(y,y)-lam*epsilon)
            gap=primal-dual
            stationarity=float(np.linalg.norm(x-d+y))
            threshold=atol+rtol*max(primal,float(np.dot(d,d)),1e-30)
            if (support(x) <= epsilon+atol and gap >= -atol and gap <= threshold
                    and stationarity <= atol+rtol*max(1.,float(np.linalg.norm(d)))):
                break
            midpoint=(lo+hi)/2
            trial,_,_ = _prox_support(d,g,s,midpoint,radius,max_iterations,atol*.01)
            if support(trial)>epsilon:
                lo=midpoint
            else:
                hi=midpoint
        else:
            raise RuntimeError("Conic projection failed primal/dual convergence")
    y=alpha*g+lam*radius*np.sqrt(s)*u
    primal=.5*float(np.sum((x-d)**2))
    dual=float(np.dot(d,y)-.5*np.dot(y,y)-lam*epsilon)
    result=np.zeros_like(d_full); result[1:]=x.reshape(d_full[1:].shape)
    return result, {"support":support(x),"epsilon":epsilon,"radius":float(radius),
                    "dual_lambda":lam,"dual_alpha":alpha,"dual_u_norm":float(np.linalg.norm(u)),
                    "primal_objective":primal,"dual_lower_bound":dual,
                    "primal_dual_gap":primal-dual,
                    "stationarity_l2":float(np.linalg.norm(x-d+y)),
                    "outer_iterations":outer_iterations,"anchor_linf":float(np.max(np.abs(result[0]))),
                    "scope":"pre-backtracking float64 projection only"}


def check_finite_step_surrogate(source, exported, times, model, *, radius, epsilon,
                                  trust_radius, finite_budget, absolute_tolerance):
    delta=exported.astype(np.float64)-source.astype(np.float64)
    dt=np.diff(times)
    difference=np.diff(delta,axis=0)
    remainder=float(.5*np.sum(difference*difference/dt[:,None,None])/model["normalization"])
    linear=float(np.sum(model["g0"]*delta))
    support=ellipsoid_support(delta,model["g0"],model["shape_diagonal"],radius)
    norm=float(np.linalg.norm(delta))
    bound=support+remainder
    hessian_bound=.5*model["lipschitz"]*norm*norm
    actual_change=linear+remainder
    tolerance=absolute_tolerance
    if (not np.isfinite(exported).all() or not np.array_equal(exported[0],source[0])
            or support>epsilon+tolerance or norm>trust_radius+tolerance
            or bound>finite_budget+tolerance or remainder>hessian_bound+tolerance):
        raise RuntimeError("Actual float32 export violates conic/trust/finite-step bound")
    return {"support":support,"l2_norm":norm,"linear_change":linear,
            "exact_quadratic_remainder":remainder,"hessian_remainder_bound":hessian_bound,
            "surrogate_change":actual_change,"robust_finite_bound":bound,
            "scope":"declared surrogate only; not native metric or coverage"}


class BacktrackingFailure(RuntimeError):
    def __init__(self, rejected):
        super().__init__("Finite-step backtracking exhausted")
        self.rejected = rejected


def _backtrack(source, projected, times, model, parameters, radius):
    rejected=[]
    initial=min(1.,parameters["trust_radius"]/max(float(np.linalg.norm(projected)),np.finfo(float).tiny))
    for index in range(parameters["max_backtracks"]+1):
        scale=initial*(.5**index)
        exported=(source.astype(np.float64)+scale*projected).astype(np.float32)
        exported[0]=source[0]
        try:
            diagnostic=check_finite_step_surrogate(source,exported,times,model,
                radius=radius,epsilon=parameters["epsilon"],trust_radius=parameters["trust_radius"],
                finite_budget=parameters["finite_budget"],absolute_tolerance=parameters["absolute_tolerance"])
            return exported,{"scale":scale,"backtracks":index,"rejected":rejected,**diagnostic}
        except RuntimeError as error:
            rejected.append({"scale":scale,"reason":str(error)})
    raise BacktrackingFailure(rejected)


def construct_all_arms(source, times, desired, model, parameters):
    """Retain each failure and continue independent deterministic/robust arms."""
    p=validate_parameters(parameters)
    result={}
    for role,radius in ((ROLES[0],0.),(ROLES[2],p["radius"])):
        try:
            projected,certificate=solve_anchored_conic_projection(desired,model["g0"],model["shape_diagonal"],
                radius=radius,epsilon=p["epsilon"],absolute_tolerance=p["absolute_tolerance"],
                relative_tolerance=p["relative_tolerance"],max_iterations=p["max_iterations"])
            exported,finite=_backtrack(source,projected,times,model,p,radius)
            result[role]={"status":"completed","vertices":exported,"projected":projected,
                          "projection":certificate,"finite_step":finite}
        except Exception as error:
            result[role]={"status":"error","error":str(error)[:4096],"exception_type":type(error).__name__,
                          "rejected_trials":getattr(error,"rejected",[])}
    try:
        robust=result[ROLES[2]]
        if robust["status"]!="completed":
            raise RuntimeError("Strength control depends on failed final robust output")
        delta=robust["vertices"].astype(np.float64)-source.astype(np.float64)
        target_norm=float(np.linalg.norm(delta)); desired_norm=float(np.linalg.norm(desired))
        scale=target_norm/desired_norm if desired_norm else 0.
        if scale>1.+p["relative_tolerance"] or (not desired_norm and target_norm):
            raise RuntimeError("Invalid final robust strength")
        matched=(source.astype(np.float64)+scale*desired).astype(np.float32); matched[0]=source[0]
        matched_delta=matched.astype(np.float64)-source.astype(np.float64)
        mismatch=abs(float(np.linalg.norm(matched_delta))-target_norm)
        quantization=float(np.linalg.norm(matched_delta-scale*desired))
        if mismatch>quantization+p["absolute_tolerance"] or (target_norm and mismatch/target_norm>0.005):
            raise RuntimeError("Float32 matching materially changes update strength")
        result[ROLES[1]]={"status":"completed","vertices":matched,"projected":scale*desired,
                         "projection":{"scope":"scalar control, no conic projection"},
                         "finite_step":{"scale":scale,"robust_final_l2":target_norm,
                         "matched_l2":float(np.linalg.norm(matched_delta)),"absolute_mismatch":mismatch,
                         "quantization_l2":quantization,"relative_mismatch_cap":0.005,
                         "protection_claim":False}}
    except Exception as error:
        result[ROLES[1]]={"status":"error","error":str(error)[:4096],"exception_type":type(error).__name__}
    return result

def _artifact_member_paths(output: Path, reports: list[dict]) -> list[Path]:
    members = [output / "candidate.json", output / "manifest.json",
               output / "common-target.npz"]
    for report in reports:
        directory = output / report["candidate_arm"]
        members.append(directory / "report.json")
        if report["status"] == "completed":
            members.extend((directory / "sequence.npz", directory / "certificate.npz"))
    return sorted(members, key=lambda path: path.relative_to(output).as_posix())


def _write_deterministic_archive(output: Path, reports: list[dict],
                                 max_artifact_bytes: int) -> dict:
    max_artifact_bytes = _positive_integer("max_artifact_bytes", max_artifact_bytes)
    if max_artifact_bytes < 10240:
        raise ValueError("max_artifact_bytes must allow one tar record")
    members = _artifact_member_paths(output, reports)
    estimated = 1024
    refs = []
    for path in members:
        size = path.stat().st_size
        estimated += 512 + ((size + 511) // 512) * 512
        refs.append({"path": path.relative_to(output).as_posix(),
                     "sha256": _digest(path), "size_bytes": size})
    estimated = ((estimated + 10239) // 10240) * 10240
    if estimated > max_artifact_bytes:
        raise RuntimeError(
            f"Deterministic artifact archive upper bound {estimated} exceeds "
            f"max_artifact_bytes={max_artifact_bytes}")
    temporary = output / "artifact.tar.tmp"
    archive = output / "artifact.tar"
    with tarfile.open(temporary, mode="w", format=tarfile.USTAR_FORMAT) as bundle:
        for path, ref in zip(members, refs):
            info = tarfile.TarInfo(ref["path"])
            info.size = ref["size_bytes"]
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as stream:
                bundle.addfile(info, stream)
    if temporary.stat().st_size > max_artifact_bytes:
        raise RuntimeError("Written artifact archive exceeds max_artifact_bytes")
    temporary.replace(archive)
    record = {
        "kind": "c04-terminal-artifact-archive", "version": 1,
        "archive": {"path": "artifact.tar", "sha256": _digest(archive),
                    "size_bytes": archive.stat().st_size},
        "members": refs, "max_artifact_bytes": max_artifact_bytes,
        "terminal_candidate_status": json.loads((output / "candidate.json").read_text())["status"],
    }
    _write_json(output / "artifact-archive.json", record)
    return record


def materialize_candidate_archive(archive_path: Path, destination: Path, *,
                                  max_artifact_bytes: int,
                                  max_member_bytes: int) -> list[dict]:
    """Safely materialize a retained terminal artifact without tar.extract."""
    archive_path, destination = Path(archive_path), Path(destination)
    max_artifact_bytes = _positive_integer("max_artifact_bytes", max_artifact_bytes)
    max_member_bytes = _positive_integer("max_member_bytes", max_member_bytes)
    if archive_path.stat().st_size > max_artifact_bytes:
        raise ValueError("Artifact archive exceeds frozen byte ceiling")
    record_path = archive_path.with_name("artifact-archive.json")
    record = json.loads(record_path.read_text())
    if (record.get("archive") != {
            "path": "artifact.tar", "sha256": _digest(archive_path),
            "size_bytes": archive_path.stat().st_size}
            or record.get("max_artifact_bytes", max_artifact_bytes) > max_artifact_bytes):
        raise ValueError("Artifact archive record/hash exceeds materialization contract")
    if destination.exists():
        raise FileExistsError("Archive destination is single-use")
    rows = []
    with tarfile.open(archive_path, mode="r:") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate artifact archive member")
        total = 0
        for member in members:
            relative = Path(member.name)
            if (not member.isfile() or relative.is_absolute() or ".." in relative.parts
                    or member.size > max_member_bytes):
                raise ValueError("Unsafe/nonregular/oversize artifact member")
            total += member.size
            if total > max_artifact_bytes:
                raise ValueError("Expanded artifact exceeds frozen byte ceiling")
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError("Regular artifact member could not be read")
            digest = hashlib.sha256()
            remaining = member.size
            with target.open("xb") as sink:
                while remaining:
                    block = source.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise ValueError("Truncated artifact member")
                    sink.write(block)
                    digest.update(block)
                    remaining -= len(block)
                if source.read(1):
                    raise ValueError("Artifact member grew beyond declared size")
            rows.append({"path": member.name, "sha256": digest.hexdigest(),
                         "size_bytes": member.size})
    if rows != record.get("members"):
        raise ValueError("Materialized member inventory differs from archive record")
    for source, target in ((archive_path, destination / "artifact.tar"),
                           (record_path, destination / "artifact-archive.json")):
        if source.stat().st_size > max_artifact_bytes:
            raise ValueError("Archive support file exceeds byte ceiling")
        remaining = source.stat().st_size
        with source.open("rb") as stream, target.open("xb") as sink:
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("Truncated archive support file")
                sink.write(block)
                remaining -= len(block)
            if stream.read(1):
                raise ValueError("Archive support file grew while copying")
    return rows



def _validate_artifact_archive(output: Path, reports: list[dict]) -> dict:
    record_path = output / "artifact-archive.json"
    archive_path = output / "artifact.tar"
    record = json.loads(record_path.read_text())
    expected_members = [{"path": path.relative_to(output).as_posix(),
                         "sha256": _digest(path), "size_bytes": path.stat().st_size}
                        for path in _artifact_member_paths(output, reports)]
    if (record.get("kind") != "c04-terminal-artifact-archive"
            or record.get("version") != 1
            or record.get("members") != expected_members
            or record.get("terminal_candidate_status")
            != json.loads((output / "candidate.json").read_text()).get("status")
            or not isinstance(record.get("max_artifact_bytes"), int)
            or record["max_artifact_bytes"] < archive_path.stat().st_size
            or record.get("archive") != {
                "path": "artifact.tar", "sha256": _digest(archive_path),
                "size_bytes": archive_path.stat().st_size}):
        raise ValueError("Terminal artifact archive record mismatch")
    with tarfile.open(archive_path, mode="r:") as bundle:
        members = bundle.getmembers()
        if [member.name for member in members] != [row["path"] for row in expected_members]:
            raise ValueError("Artifact archive member inventory/order mismatch")
        for member, expected in zip(members, expected_members):
            if (not member.isfile() or member.size != expected["size_bytes"]
                    or member.mode != 0o644 or member.uid != 0 or member.gid != 0
                    or member.mtime != 0 or member.uname or member.gname):
                raise ValueError("Artifact archive metadata is not canonical")
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError("Artifact archive regular member is unreadable")
            digest = hashlib.sha256()
            remaining = member.size
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("Artifact archive member is truncated")
                digest.update(block)
                remaining -= len(block)
            if stream.read(1) or digest.hexdigest() != expected["sha256"]:
                raise ValueError("Artifact archive member bytes mismatch")
    return {
        "record": {"path": record_path, "sha256": _digest(record_path)},
        "archive": {"path": archive_path, "sha256": _digest(archive_path)},
    }



def _safe_ref(root, ref):
    if not isinstance(ref,dict) or set(ref)!={"path","sha256"}:
        raise ValueError("Exact path/hash ref required")
    relative=Path(ref["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe reference")
    path=root/relative
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("Symlinked evidence rejected")
    path.resolve().relative_to(root.resolve())
    if not path.is_file() or _digest(path)!=ref["sha256"]:
        raise ValueError("Evidence reference hash mismatch")
    return path


def _bound_diagnostics(vertices, lower, upper, policy):
    if policy not in ("preserve_and_report","reject") or not np.isfinite([lower,upper]).all() or lower>=upper:
        raise ValueError("Invalid coordinate bounds contract")
    count=int(np.sum((vertices<lower)|(vertices>upper)))
    if policy=="reject" and count:
        raise ValueError("Coordinates outside declared bounds")
    return {"policy":policy,"lower":lower,"upper":upper,"outside_coordinate_count":count,"clipped":False}


def export_robust_candidate(source_case, output, *, uid, expected_sequence_sha256,
                            source_sequence_ref, source_report_ref, parameters,
                            coordinate_lower,coordinate_upper,bounds_policy,max_artifact_bytes):
    started=time.monotonic(); source_case=Path(source_case); output=Path(output)
    p=validate_parameters(parameters)
    if output.exists():
        raise FileExistsError("Candidate output is single-use")
    if not uid or Path(uid).name!=uid or uid in (".",".."):
        raise ValueError("Filesystem-safe UID required")
    for ref in (source_sequence_ref,source_report_ref):
        if not ref or Path(ref).is_absolute() or ".." in Path(ref).parts:
            raise ValueError("Safe repository-relative source refs required")
    source_path=source_case/"sequence.npz"; report_path=source_case/"report.json"
    source_report=json.loads(report_path.read_text())
    if (_digest(source_path)!=expected_sequence_sha256 or source_report.get("status")!="completed"
        or source_report.get("uid")!=uid or source_report.get("sha256",{}).get("sequence.npz")!=expected_sequence_sha256):
        raise ValueError("Completed receipt-bound same-UID source required")
    seed=source_report.get("seed")
    if isinstance(seed,bool) or not isinstance(seed,int):
        raise ValueError("Integer seed required")
    with np.load(source_path,allow_pickle=False) as saved:
        arrays={key:saved[key].copy() for key in saved.files}
    source,faces,times=validate_native_arrays(arrays)
    _bound_diagnostics(source,coordinate_lower,coordinate_upper,bounds_policy)
    output.mkdir(parents=True,exist_ok=False)
    common_error=None; repair={}; outcomes={}
    common={"source_vertices":source,"faces":faces,"timesteps":times}
    try:
        desired,repair=compute_common_geometry_repair(source,faces,times,
            arap_weight=p["arap_weight"],temporal_weight=p["temporal_weight"],iterations=p["iterations"],
            cg_tolerance=p["cg_tolerance"],cg_max_iterations=p["cg_max_iterations"])
        model=motion_surrogate(source,times,shape_floor=p["shape_floor"],shape_gain=p["shape_gain"])
        common.update(desired_step=desired,**{k:np.asarray(v) for k,v in model.items()})
        outcomes=construct_all_arms(source,times,desired,model,p)
    except Exception as error:
        common_error={"error":str(error)[:4096],"exception_type":type(error).__name__}
        outcomes={role:{"status":"error",**common_error} for role in ROLES}
    common_path=output/"common-target.npz"; np.savez_compressed(common_path,**common)
    common_ref={"path":"common-target.npz","sha256":_digest(common_path)}
    source_refs={"sequence":{"path":source_sequence_ref,"sha256":expected_sequence_sha256},
                 "report":{"path":source_report_ref,"sha256":_digest(report_path)}}
    base={"candidate_id":CANDIDATE_ID,"uid":uid,"seed":seed,"source_refs":source_refs,
          "source_sequence_sha256":expected_sequence_sha256,"source_report_sha256":_digest(report_path),
          "implementation_sha256":_digest(Path(__file__)),"common_target":common_ref,
          "target_parameters":p,"repair":repair,"common_error":common_error,
          "surrogate":"normalized_predicted_kinetic_energy_v1",
          "uncertainty":"positive_diagonal_acceleration_weighted_gradient_ellipsoid_v1",
          "semantic_coverage_verified":False,"probabilistic_coverage_claim":False,
          "native_qualified":False,"scientific_admission":False,"local_method_verified":False,
          "source_delivery_status":"generated_unexecuted_at_authoring"}
    reports=[]
    for role in ROLES:
        directory=output/role; directory.mkdir()
        outcome=outcomes[role]; report={**base,"candidate_arm":role,"method_id":METHOD_IDS[role],
                                      "role_contract":role,"status":outcome["status"],
                                      "failure_phase":None if outcome["status"]=="completed" else "construction"}
        try:
            if outcome["status"]!="completed":
                report.update({k:outcome[k] for k in ("error","exception_type")})
                report["rejected_trials"]=outcome.get("rejected_trials",[])
            else:
                # Retain completed projection and every rejected backtrack even if
                # bounds checks or physical export subsequently fail.
                report.update(projection=outcome["projection"],finite_step=outcome["finite_step"])
                vertices=outcome["vertices"]
                bounds=_bound_diagnostics(vertices,coordinate_lower,coordinate_upper,bounds_policy)
                np.savez_compressed(directory/"sequence.npz",**{**arrays,"vertices":vertices})
                np.savez_compressed(directory/"certificate.npz",projected_step=outcome["projected"],
                                    displacement=vertices.astype(np.float64)-source.astype(np.float64))
                report.update(projection=outcome["projection"],finite_step=outcome["finite_step"],export_bounds=bounds,
                              frames=16,topology_preserved=True,identity_mapping_preserved=True,
                              sha256={name:_digest(directory/name) for name in ("sequence.npz","certificate.npz")})
        except Exception as error:
            for name in ("sequence.npz","certificate.npz"):
                (directory/name).unlink(missing_ok=True)
            report.pop("sha256",None)
            report.update(status="error",failure_phase="export",error=str(error)[:4096],exception_type=type(error).__name__)
        _write_json(directory/"report.json",report); reports.append(report)
    completed=[r for r in reports if r["status"]=="completed"]
    manifest={"cases":[{"case_id":uid+"-"+r["candidate_arm"],"uid":uid,"case_dir":r["candidate_arm"],
                        "arm_role":r["candidate_arm"]} for r in completed],"expected_roles":list(ROLES),
              "common_target":common_ref,"scope":SCOPE}
    _write_json(output/"manifest.json",manifest)
    result={**base,"status":"completed" if len(completed)==3 else "incomplete","roles":list(ROLES),
            "arms":reports,"solver":{"absolute_tolerance":p["absolute_tolerance"],
                "relative_tolerance":p["relative_tolerance"],"max_iterations":p["max_iterations"]},
            "bounds":{"lower":coordinate_lower,"upper":coordinate_upper,"policy":bounds_policy},
            "elapsed_seconds":time.monotonic()-started,"recorded_at":datetime.now(timezone.utc).isoformat()}
    _write_json(output/"candidate.json",result)
    _write_deterministic_archive(output,reports,max_artifact_bytes)
    return result


def validate_candidate_artifact(root, candidate_path):
    """Local-only native acceptance: reconstruct source, shared repair and every arm."""
    root=Path(root).resolve(); candidate_path=Path(candidate_path).resolve(); candidate_path.relative_to(root)
    output=candidate_path.parent; candidate=json.loads(candidate_path.read_text())
    if candidate.get("candidate_id")!=CANDIDATE_ID or tuple(candidate.get("roles",()))!=ROLES:
        raise ValueError("Exact C04 artifact identity required")
    for key in ("native_qualified","scientific_admission","local_method_verified","semantic_coverage_verified","probabilistic_coverage_claim"):
        if candidate.get(key) is not False:
            raise ValueError("Artifact overstates evidence scope")
    if candidate.get("source_delivery_status")!="generated_unexecuted_at_authoring" or candidate.get("implementation_sha256")!=_digest(Path(__file__)):
        raise ValueError("Implementation/scope identity mismatch")
    if (candidate.get("surrogate")!="normalized_predicted_kinetic_energy_v1"
        or candidate.get("uncertainty")!="positive_diagonal_acceleration_weighted_gradient_ellipsoid_v1"):
        raise ValueError("Declared surrogate/set specialization identity mismatch")
    p=validate_parameters(candidate["target_parameters"])
    if candidate.get("solver")!={k:p[k] for k in ("absolute_tolerance","relative_tolerance","max_iterations")}:
        raise ValueError("Solver parameter aliases mismatch")
    refs=candidate["source_refs"]
    if set(refs)!={"sequence","report"}:
        raise ValueError("Exact source refs required")
    source_path=_safe_ref(root,refs["sequence"]); report_path=_safe_ref(root,refs["report"])
    if source_path.name!="sequence.npz" or report_path!=source_path.with_name("report.json"):
        raise ValueError("Colocated source sequence/report required")
    source_report=json.loads(report_path.read_text())
    if (source_report.get("status")!="completed" or source_report.get("uid")!=candidate["uid"]
        or source_report.get("seed")!=candidate["seed"]
        or source_report.get("sha256",{}).get("sequence.npz")!=refs["sequence"]["sha256"]
        or candidate.get("source_sequence_sha256")!=refs["sequence"]["sha256"]
        or candidate.get("source_report_sha256")!=refs["report"]["sha256"]):
        raise ValueError("Source receipt or identity mismatch")
    with np.load(source_path,allow_pickle=False) as saved:
        arrays={key:saved[key].copy() for key in saved.files}
    source,faces,times=validate_native_arrays(arrays)
    common_path=_safe_ref(output,candidate["common_target"])
    if common_path.name!="common-target.npz":
        raise ValueError("Exact common target path required")
    with np.load(common_path,allow_pickle=False) as saved:
        common={key:saved[key].copy() for key in saved.files}
    expected_common={"source_vertices":source,"faces":faces,"timesteps":times}
    outcomes={}; common_error=None; repair={}
    try:
        desired,repair=compute_common_geometry_repair(source,faces,times,
            arap_weight=p["arap_weight"],temporal_weight=p["temporal_weight"],iterations=p["iterations"],
            cg_tolerance=p["cg_tolerance"],cg_max_iterations=p["cg_max_iterations"])
        model=motion_surrogate(source,times,shape_floor=p["shape_floor"],shape_gain=p["shape_gain"])
        expected_common.update(desired_step=desired,**{k:np.asarray(v) for k,v in model.items()})
        outcomes=construct_all_arms(source,times,desired,model,p)
    except Exception as error:
        common_error={"error":str(error)[:4096],"exception_type":type(error).__name__}
        outcomes={role:{"status":"error",**common_error} for role in ROLES}
    if set(common)!=set(expected_common) or any(not np.array_equal(common[k],v) for k,v in expected_common.items()):
        raise ValueError("Common target/surrogate/shape failed exact recomputation")
    if candidate.get("repair")!=repair or candidate.get("common_error")!=common_error:
        raise ValueError("Common repair evidence mismatch")
    records=[]; reports=[]; bounds=candidate["bounds"]
    for role in ROLES:
        directory=output/role; report_path=directory/"report.json"; report=json.loads(report_path.read_text())
        reports.append(report)
        for key in ("candidate_id","uid","seed","source_refs","source_sequence_sha256","source_report_sha256",
                    "implementation_sha256","common_target","target_parameters","repair","common_error",
                    "surrogate","uncertainty","semantic_coverage_verified","probabilistic_coverage_claim",
                    "native_qualified","scientific_admission","local_method_verified","source_delivery_status"):
            if report.get(key)!=candidate.get(key):
                raise ValueError("Arm/candidate invariant mismatch: "+key)
        if report.get("candidate_arm")!=role or report.get("method_id")!=METHOD_IDS[role] or report.get("role_contract")!=role:
            raise ValueError("Arm identity mismatch")
        outcome=outcomes[role]
        # A frozen reject-bounds policy may turn an otherwise complete solve into an error.
        if outcome["status"]=="completed":
            try:
                expected_bounds=_bound_diagnostics(outcome["vertices"],bounds["lower"],bounds["upper"],bounds["policy"])
            except ValueError as error:
                outcome={"status":"error","error":str(error),"exception_type":type(error).__name__}
        observed_export_failure=(report.get("status")=="error" and report.get("failure_phase")=="export")
        if observed_export_failure:
            constructed=outcomes[role]
            if (constructed["status"]!="completed"
                or report.get("projection")!=constructed["projection"]
                or report.get("finite_step")!=constructed["finite_step"]):
                raise ValueError("Export failure lost completed solver/backtracking evidence")
            # The original receipt retains an observed I/O/export failure. Replaying
            # a deterministic solver need not recreate a transient disk failure.
            if (not isinstance(report.get("error"),str) or not report["error"]
                or not isinstance(report.get("exception_type"),str) or not report["exception_type"]):
                raise ValueError("Export failure requires original exception evidence")
        elif report.get("status")!=outcome["status"]:
            raise ValueError("Arm status failed native reconstruction")
        if report.get("status")=="completed" and report.get("failure_phase") is not None:
            raise ValueError("Completed arm retains contradictory failure phase")
        if report.get("status")=="error" and report.get("failure_phase") not in ("construction","export"):
            raise ValueError("Failed arm requires explicit observed phase")
        row={"role":role,"status":report["status"],"sequence":None,"certificate":None,
             "report":{"path":report_path.relative_to(root).as_posix(),"sha256":_digest(report_path)}}
        if report["status"]=="error":
            if (directory/"sequence.npz").exists() or (directory/"certificate.npz").exists() or report.get("sha256"):
                raise ValueError("Failed arm contains scoreable outputs")
            if (not observed_export_failure and
                (report.get("error")!=outcome["error"] or report.get("exception_type")!=outcome["exception_type"]
                 or report.get("rejected_trials",[])!=outcome.get("rejected_trials",[]))):
                raise ValueError("Failure evidence differs from recomputation")
        else:
            hashes={name:_digest(directory/name) for name in ("sequence.npz","certificate.npz")}
            if report.get("sha256")!=hashes:
                raise ValueError("Arm output hashes differ")
            with np.load(directory/"sequence.npz",allow_pickle=False) as saved:
                exported={key:saved[key].copy() for key in saved.files}
            validate_native_arrays(exported)
            if set(exported)!=set(arrays) or any(not np.array_equal(exported[k],outcome["vertices"] if k=="vertices" else v) for k,v in arrays.items()):
                raise ValueError("Arm output/metadata differs from exact reconstruction")
            with np.load(directory/"certificate.npz",allow_pickle=False) as saved:
                cert={key:saved[key].copy() for key in saved.files}
            expected_cert={"projected_step":outcome["projected"],"displacement":outcome["vertices"].astype(np.float64)-source.astype(np.float64)}
            if set(cert)!=set(expected_cert) or any(not np.array_equal(cert[k],v) for k,v in expected_cert.items()):
                raise ValueError("Arm certificate differs from exact reconstruction")
            if (report.get("projection")!=outcome["projection"] or report.get("finite_step")!=outcome["finite_step"]
                or report.get("export_bounds")!=expected_bounds):
                raise ValueError("Numerical certificate diagnostics mismatch")
            for key,name in (("sequence","sequence.npz"),("certificate","certificate.npz")):
                row[key]={"path":(directory/name).relative_to(root).as_posix(),"sha256":hashes[name]}
        records.append(row)
    expected_status="completed" if all(r["status"]=="completed" for r in records) else "incomplete"
    if candidate.get("arms")!=reports or candidate.get("status")!=expected_status:
        raise ValueError("Embedded reports/aggregate status mismatch")
    manifest_path=output/"manifest.json"; manifest=json.loads(manifest_path.read_text())
    expected_manifest={"cases":[{"case_id":candidate["uid"]+"-"+r["role"],"uid":candidate["uid"],"case_dir":r["role"],"arm_role":r["role"]} for r in records if r["status"]=="completed"],"expected_roles":list(ROLES),"common_target":candidate["common_target"],"scope":SCOPE}
    if manifest!=expected_manifest:
        raise ValueError("Manifest differs from exact terminal roles")
    archive_refs=_validate_artifact_archive(output,reports)
    ref=lambda path:{"path":path.relative_to(root).as_posix(),"sha256":_digest(path)}
    return {"candidate_id":CANDIDATE_ID,"uid":candidate["uid"],"seed":candidate["seed"],"roles":list(ROLES),
            "source_sequence_sha256":candidate["source_sequence_sha256"],"common_target":ref(common_path),
            "manifest":ref(manifest_path),"artifact_archive":{key:ref(value["path"]) for key,value in archive_refs.items()},
            "arms":records,"status":candidate["status"],"native_qualified":False,"scientific_admission":False,
            "local_method_verified":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sequence",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    for name in ("uid","expected-sequence-sha256","source-sequence-ref","source-report-ref"):
        parser.add_argument("--"+name,required=True)
    for name in FLOAT_PARAMETERS:
        parser.add_argument("--"+name.replace("_","-"),type=float,required=True)
    for name in INTEGER_PARAMETERS:
        parser.add_argument("--"+name.replace("_","-"),type=int,required=True)
    parser.add_argument("--coordinate-bounds",type=float,nargs=2,required=True)
    parser.add_argument("--bounds-policy",choices=("preserve_and_report","reject"),required=True)
    parser.add_argument("--max-artifact-bytes",type=int,required=True)
    args=parser.parse_args()
    if args.source_sequence.name!="sequence.npz":
        parser.error("Expected sequence.npz")
    result=export_robust_candidate(args.source_sequence.parent,args.output,uid=args.uid,
        expected_sequence_sha256=args.expected_sequence_sha256,source_sequence_ref=args.source_sequence_ref,
        source_report_ref=args.source_report_ref,parameters={k:getattr(args,k) for k in FLOAT_PARAMETERS+INTEGER_PARAMETERS},
        coordinate_lower=args.coordinate_bounds[0],coordinate_upper=args.coordinate_bounds[1],
        bounds_policy=args.bounds_policy,max_artifact_bytes=args.max_artifact_bytes)
    print(json.dumps(result,allow_nan=False))
    return 0 if result["status"]=="completed" else 1


if __name__=="__main__":
    raise SystemExit(main())
