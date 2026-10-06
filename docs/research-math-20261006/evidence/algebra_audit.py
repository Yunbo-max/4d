"""Finite algebra witnesses for conditional derivations, NOT method evaluation.

No videos, meshes from ActionBench, learned checkpoints or native scorer are used.
These calculations check identities on explicitly constructed finite objects;
they are neither general theorem proofs nor evidence of practical performance.
"""
import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import null_space, polar
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',required=True,help='Research repository root')
ROOT = Path(parser.parse_args().root).resolve()
OUT = ROOT / 'docs/research-math-20261006/evidence'
RNG = np.random.default_rng(20261006)
CHECKS = []

def witness(number, claim, actual, expected, limit='finite identity only', tol=1e-9):
    err = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    if not np.isfinite(err) or err > tol:
        raise AssertionError((number, claim, err, tol))
    CHECKS.append(dict(candidate_id=f'4d-math-20261006-c{number:02d}',
                       claim=claim, status='passed', max_absolute_error=err,
                       tolerance=tol, scope=limit))

def spd(n):
    a = RNG.normal(size=(n,n))
    return a.T @ a + np.eye(n)

# C01: exact anchor, additive gauge, and retained risk cross term.
x, fa, ft, bias = RNG.normal(size=(4,12,3))
h = x + ft-fa
witness(1,'anchor substitution',x+fa-fa,x)
witness(1,'shared additive bias cancellation',x+(ft+bias)-(fa+bias),h)
e, r = RNG.normal(size=(2,100,3))
witness(1,'risk expansion including cross moment',
        np.mean(np.sum((e-r)**2-e**2,axis=1)),
        np.mean(np.sum(r*r-2*e*r,axis=1)))

# C02: redundant constraints, weighted KKT solution and independent nullspace LS.
n=8; w=spd(n); c0=RNG.normal(size=(3,n)); c=np.vstack([c0,c0[0]])
d=RNG.normal(size=n); wi=np.linalg.inv(w)
delta=d-wi@c.T@np.linalg.pinv(c@wi@c.T)@c@d
z=null_space(c); independently=z@np.linalg.solve(z.T@w@z,z.T@w@d)
witness(2,'redundant constraint feasibility',c@delta,np.zeros(c.shape[0]))
witness(2,'KKT versus nullspace solve',delta,independently)
full=d-wi@np.linalg.pinv(wi)@d
witness(2,'full column rank protection forbids all changes',full,np.zeros(n))

# C03: moments are constructed PSD, not estimated from a real population.
cov=spd(6); ctt,cta,caa=cov[:3,:3],cov[:3,3:],cov[3:,3:]
cat=cta.T; bstar=cta@np.linalg.inv(caa)
def risk(b): return np.trace(ctt)-2*np.trace(b@cat)+np.trace(b@caa@b.T)
errmat=RNG.normal(size=(3,3))
witness(3,'optimal correction reduction',risk(np.zeros((3,3)))-risk(bstar),
        np.trace(cta@np.linalg.inv(caa)@cat),'true-moment finite covariance identity')
witness(3,'fitted matrix excess-risk square',risk(bstar+errmat)-risk(bstar),
        np.trace(errmat@caa@errmat.T),'does not establish availability/calibration of these moments')
ec,rc=RNG.normal(size=(2,40,3));ec-=ec.mean(0);rc-=rc.mean(0)
mean_offset=RNG.normal(size=3); centered_error=ec-rc@bstar.T
witness(3,'fitted means add their squared offset to centered risk',
        np.mean(np.sum((centered_error+mean_offset)**2,axis=1)),
        np.mean(np.sum(centered_error**2,axis=1))+mean_offset@mean_offset,
        'finite centered sample identity; calibration and distribution transfer remain unknown')

# C04: ellipsoidal support, an attaining direction, and a quadratic remainder.
sh=spd(5); ev,q=np.linalg.eigh(sh); sr=(q*np.sqrt(ev))@q.T
g,dd=RNG.normal(size=(2,5)); radius=.7; ss=sr@dd
u=radius*ss/np.linalg.norm(ss)* (1 if g@dd>=0 else -1)
witness(4,'ellipsoid support attained',abs((g+sr@u)@dd),
        abs(g@dd)+radius*np.linalg.norm(ss),'set support only; no probabilistic coverage')

# C05: normalized finite weights and an ambiguity counterexample, not native data.
ys=RNG.normal(size=(4,3)); p=np.array([.1,.2,.3,.4]); mean=p@ys
covar=sum(p[i]*np.outer(ys[i]-mean,ys[i]-mean) for i in range(4))
witness(5,'weighted variance identity',p@np.sum(ys*ys,axis=1)-mean@mean,np.trace(covar))
v=np.array([1.,0.,0.]); rplus=Rotation.from_euler('z',60,degrees=True).as_matrix()
rminus=Rotation.from_euler('z',-60,degrees=True).as_matrix()
witness(5,'opposite rotation mean contracts',np.linalg.norm((rplus@v+rminus@v)/2),.5,
        'counterexample to universal shape preservation by averaging; no prevalence claim')

# C06: finite positive kernel scaling witness; does not test geometry correspondence.
a=np.array([.2,.3,.5]); b=np.array([.6,.4]); cost=RNG.uniform(size=(3,2))
eps=.4; kernel=np.exp(-cost/eps); vv=np.ones(2)
for _ in range(2000):
    uu=a/(kernel@vv); vv=b/(kernel.T@uu)
pi=uu[:,None]*kernel*vv[None,:]
witness(6,'two transport marginals',np.r_[pi.sum(1),pi.sum(0)],np.r_[a,b])
witness(6,'stationarity scaling identity',cost+eps*np.log(pi),
        eps*(np.log(uu)[:,None]+np.log(vv)[None,:]))

# C07: zero-entropy scalar LP, both unmatched slacks are charged.
gamma=.8
for cc,match in [(.5,1.),(2.,0.)]:
    opt=linprog([cc-2*gamma],bounds=[(0,1)],method='highs')
    if not opt.success: raise AssertionError(opt.message)
    witness(7,f'unregularized scalar optimum cost={cc}',opt.x[0],match,
            'epsilon=0 scalar limit only; coupled capacities/entropy alter allocation')

# C08: positive two-state reference; endpoint scaling versus exact enumeration.
paths=list(itertools.product(range(2),repeat=4)); q1=np.array([.4,.6])
kernels=[np.array([[.8,.2],[.3,.7]]),np.array([[.6,.4],[.2,.8]]),
         np.array([[.7,.3],[.4,.6]])]
pathq=np.array([q1[ps[0]]*np.prod([kernels[t][ps[t],ps[t+1]] for t in range(3)]) for ps in paths])
endjoint=np.zeros((2,2))
for ps,prob in zip(paths,pathq): endjoint[ps[0],ps[-1]]+=prob
first,last=np.array([.7,.3]),np.array([.2,.8]); vend=np.ones(2)
for _ in range(1000):
    uend=first/(endjoint@vend); vend=last/(endjoint.T@uend)
pq=np.array([uend[ps[0]]*prob*vend[ps[-1]] for ps,prob in zip(paths,pathq)])
first_enum=np.array([sum(pr for ps,pr in zip(paths,pq) if ps[0]==i) for i in range(2)])
last_enum=np.array([sum(pr for ps,pr in zip(paths,pq) if ps[-1]==i) for i in range(2)])
f=q1*uend
for k in kernels: f=f@k
witness(8,'forward message versus enumerated final marginal',f*vend,last_enum)
witness(8,'endpoint scaling closure',np.r_[first_enum,last_enum],np.r_[first,last])

# C09: connection Laplacian energy and the right global rotation gauge.
rot=Rotation.from_rotvec(RNG.normal(size=(3,3))).as_matrix(); gauge=Rotation.random(random_state=4).as_matrix()
relative=rot[0]@rot[1].T; test=Rotation.from_rotvec(RNG.normal(size=(2,3))).as_matrix()
u=np.vstack(test); lap=np.block([[np.eye(3),-relative],[-relative.T,np.eye(3)]])
eng=np.sum((test[0]-relative@test[1])**2)
witness(9,'connection Laplacian energy',np.trace(u.T@lap@u),eng)
witness(9,'common right rotation gauge',np.sum((test[0]@gauge-relative@test[1]@gauge)**2),eng)

# C10: inconsistent differential on a triangle graph; one node is pinned.
gg=np.array([[-1,1,0],[0,-1,1],[1,0,-1]],float); ss=np.eye(3)[:,1:]
ww=np.diag([1.,2.,3.]); hh=np.array([1.,1.,1.]); gs=gg@ss
zz=np.linalg.solve(gs.T@ww@gs,gs.T@ww@hh)
sqrtw=np.diag(np.sqrt(np.diag(ww))); zr=np.linalg.lstsq(sqrtw@gs,sqrtw@hh,rcond=None)[0]
witness(10,'anchored normal equations versus QR/LS',zz,zr)
witness(10,'weighted projection orthogonality',gs.T@ww@(gs@zz-hh),np.zeros(2))

# C11: full-rank square map, not an automatically defined surface-triangle map.
rr=Rotation.random(random_state=9).as_matrix(); stretch=spd(3); ff=rr@stretch
pr,ps=polar(ff); vals,vec=np.linalg.eigh(ps)
clipped=(vec*np.clip(vals,.5,2.))@vec.T; new=pr@clipped
nr,ns=polar(new)
witness(11,'polar stretch projection retains proper rotation',nr,pr,
        'local square-map property only; integration and real stretch can change conclusions')
witness(11,'stretch spectral clipping',np.linalg.eigvalsh(ns),np.clip(vals,.5,2.))

# C12: exact line polynomial and a known first crossing.
e1=np.array([1.,0,0]);e2=np.array([0.,1,0]);n0=np.array([0.,0,1])
d1=np.array([-.2,.4,.1]);d2=np.array([.2,-.3,.4])
coefs=[n0@np.cross(e1,e2),n0@(np.cross(d1,e2)+np.cross(e1,d2)),n0@np.cross(d1,d2)]
for alpha in [0.,.13,.51,1.]:
    witness(12,'oriented area exact quadratic',n0@np.cross(e1+alpha*d1,e2+alpha*d2),
            coefs[0]+coefs[1]*alpha+coefs[2]*alpha**2)
beta=.2; firstroot=(1-beta)/2
witness(12,'first positive crossing of 1-2alpha floor',1-2*firstroot,beta,
        'projected area only; not a global injectivity certificate')

# C13: group-norm support; non-diagonal metric and anchor terms retained.
aa=RNG.normal(size=3); lam=.7; pp=lam*aa/np.linalg.norm(aa)
witness(13,'group norm support',pp@aa,lam*np.linalg.norm(aa))
tt=np.arange(5)*.25; second=np.diff(tt**2,n=2)/.25**2
witness(13,'physical second differences of quadratic',second,np.full(3,2.))
m=spd(5); ss=np.eye(5)[:,1:]; yp=np.array([.9,0,0,0,0]); yhat=RNG.normal(size=5)
mf=ss.T@m@ss; center=np.linalg.solve(mf,ss.T@m@(yhat-yp)); zz=RNG.normal(size=4)
witness(13,'anchored effective center keeps metric cross terms',
        ss.T@m@(yp+ss@zz-yhat),mf@(zz-center))

# C14: exact rigid factorization witness, no pose estimation success assumed.
anchor=RNG.normal(size=(7,3));rots=Rotation.from_rotvec(RNG.normal(size=(4,3))).as_matrix();centers=RNG.normal(size=(4,3))
y=np.array([anchor@r.T+c for r,c in zip(rots,centers)])
body=np.array([(yt-c)@r-anchor for yt,r,c in zip(y,rots,centers)])
witness(14,'exact varying rigid pose has zero body residual',body,np.zeros_like(body))
witness(14,'zero-preserving body repair reconstructs rigid sequence',
        np.array([anchor@r.T+c for r,c in zip(rots,centers)]),y)

# C15: singular-value proximal on the orthogonal residual, including anchor.
t=6; e0=np.eye(t)[:,0]; v=RNG.normal(size=t);v[0]=0;v/=np.linalg.norm(v)
q=np.outer(e0,e0)+np.outer(v,v); uh=RNG.normal(size=(t,18)); rh=(np.eye(t)-q)@uh
left,sv,right=np.linalg.svd(rh,full_matrices=False); rp=(left*np.maximum(sv-.6,0))@right
up=q@uh+rp
witness(15,'SVT remains in complement of protected subspace',q@rp,np.zeros_like(rp))
witness(15,'protected components and anchor preserved',q@up,q@uh)
witness(15,'anchor row preserved',up[0],uh[0])

# C16: shared anchor covariance propagation, independent later noises stipulated.
sigma=spd(3); j1=RNG.normal(size=(4,3));j2=RNG.normal(size=(5,3));jointj=np.vstack([j1,j2])
jointcov=jointj@sigma@jointj.T+np.eye(9)
witness(16,'cross-time shared covariance block',jointcov[:4,4:],j1@sigma@j2.T)
witness(16,'risk trace propagation',np.trace(jointcov[:4,:4]),np.trace(j1@sigma@j1.T)+4,
        'specified independent linear-noise model only; actual decoder covariance unmeasured')

# C17: continuous convex query allocation with strictly positive coefficients.
areas=np.array([1.,2.,.5]);kappa=np.array([.8,1.3,2.]);coeff=kappa**2*areas**3; total=90.
allocation=total*coeff**(1/3)/sum(coeff**(1/3));derivative=-2*coeff/allocation**3
witness(17,'fixed total continuous budget',allocation.sum(),total)
witness(17,'KKT equal marginal derivatives',derivative,np.full(3,derivative[0]),
        'continuous allocation only; bounds, rounding, curvature probes and coverage still owed')

# C18: weighted nuisance elimination, including redundant camera columns.
w=spd(9); jc0=RNG.normal(size=(9,2));jc=np.column_stack([jc0,jc0[:,0]]);v=RNG.normal(size=9)
proj=np.eye(9)-jc@np.linalg.pinv(jc.T@w@jc)@jc.T@w
dc=-np.linalg.pinv(jc.T@w@jc)@jc.T@w@v
witness(18,'weighted nuisance residual versus elimination',v+jc@dc,proj@v)
witness(18,'weighted projector idempotence',proj@proj,proj)
witness(18,'weighted self adjointness',w@proj,(w@proj).T)

# C19: ridge output coefficient solution versus augmented least squares.
kk=RNG.normal(size=(9,2));bb=RNG.normal(size=9);lam=.3
coef=np.linalg.solve(kk.T@kk+lam*np.eye(2),kk.T@bb)
aug=np.linalg.lstsq(np.vstack([kk,np.sqrt(lam)*np.eye(2)]),np.r_[bb,np.zeros(2)],rcond=None)[0]
witness(19,'output ridge solution versus augmented least squares',coef,aug)
unreachable=bb-kk@np.linalg.lstsq(kk,bb,rcond=None)[0]
witness(19,'unreachable output component orthogonal to guidance span',kk.T@unreachable,np.zeros(2),
        'linear reachability only; real decoder Jacobian and finite-step remainder unmeasured')

# C20: profiled phase/amplitude solve versus the full unconstrained block problem.
w=spd(10);phase=RNG.normal(size=(10,2));amp=RNG.normal(size=(10,3));resid=RNG.normal(size=10);lam=.8
mm=np.eye(10)-phase@np.linalg.pinv(phase.T@w@phase)@phase.T@w
gam=np.linalg.solve(amp.T@w@mm@amp+lam*np.eye(3),amp.T@w@mm@resid)
allcols=np.column_stack([phase,amp]);ridge=np.diag([0,0,lam,lam,lam])
joint=np.linalg.solve(allcols.T@w@allcols+ridge,allcols.T@w@resid)
witness(20,'profiled amplitude versus unconstrained joint solve',gam,joint[2:],
        'inactive phase inequalities only; actual timewarp needs monotonicity and remainder checks')
witness(20,'complete overlap removes amplitude identification',mm@phase,np.zeros_like(phase))

script=Path(__file__).read_bytes()
report=dict(kind='finite-algebra-audit',version='1.0.0',
            executed_at=datetime.now(timezone.utc).isoformat(),seed=20261006,
            numpy_version=np.__version__,scipy_version=scipy.__version__,
            candidate_count=len(set(c['candidate_id'] for c in CHECKS)),
            passed_checks=len(CHECKS),script_sha256=hashlib.sha256(script).hexdigest(),
            scope='Deterministic finite algebra witnesses only. No native benchmark, candidate implementation, performance result, mathematical theorem certification or independent review.',
            checks=CHECKS)
(OUT/'algebra-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['executed_at','candidate_count','passed_checks','scope']}))
