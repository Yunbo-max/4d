"""M02: actual multi-source decoder cycles with anchored graph consensus.

NumPy callback protocol: decode(source_time, target_times, positions, normals,
vertex_ids) -> absolute positions [len(target_times),len(vertex_ids),3].
All calls use the same frozen latent context/coordinate system. One-ring support
is explicitly queried to recompute intermediate mesh normals; it is NOT free.
Default normals are area weighted. Supply the official normal_fn/source_normals
when testing native replay parity. A smaller cycle residual is not quality proof.
"""
import numpy as np


def vertex_normals(vertices, faces):
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all()
            or faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in "iu"
            or len(faces)==0 or faces.min()<0 or faces.max()>=len(vertices)):
        raise ValueError("finite vertices and valid integer triangles required")
    tri=vertices[faces]
    areas=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    result=np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(result,faces[:,corner],areas)
    norm=np.linalg.norm(result,axis=1)
    if np.any(norm<=1e-14):
        raise ValueError("undefined normals: isolated/degenerate/cancelling vertex")
    return result/norm[:,None]


def anchored_consensus(direct, edges, *, anchor_index=0, weight=1., edge_weights=None):
    """Solve data fidelity + sum ||X_j-X_i-edge_delta||²; anchor exact."""
    direct=np.asarray(direct,dtype=float)
    if (direct.ndim!=3 or direct.shape[-1]!=3 or not np.isfinite(direct).all()
            or not 0<=anchor_index<len(direct) or not np.isfinite(weight) or weight<0):
        raise ValueError("invalid consensus input")
    edges=list(edges)
    weights=np.ones(len(edges)) if edge_weights is None else np.asarray(edge_weights,dtype=float)
    if weights.shape!=(len(edges),) or not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError("invalid per-edge weights")
    t=len(direct);matrix=np.eye(t);rhs=direct.reshape(t,-1).copy()
    for (i,j,delta),relative_weight in zip(edges,weights):
        delta=np.asarray(delta,dtype=float)
        if not (0<=i<t and 0<=j<t and i!=j) or delta.shape!=direct.shape[1:] or not np.isfinite(delta).all():
            raise ValueError("invalid graph edge")
        strength=weight*relative_weight
        matrix[i,i]+=strength;matrix[j,j]+=strength
        matrix[i,j]-=strength;matrix[j,i]-=strength
        rhs[i]-=strength*delta.ravel();rhs[j]+=strength*delta.ravel()
    free=np.array([i for i in range(t) if i!=anchor_index])
    result=direct.copy()
    result[free]=np.linalg.solve(matrix[np.ix_(free,free)],
                                rhs[free]-matrix[free,anchor_index,None]*direct[anchor_index].ravel()).reshape(
                                    len(free),*direct.shape[1:])
    result[anchor_index]=direct[anchor_index]
    return result


def probe_cycles(decode, anchor_vertices, faces, times, query_ids, *,
                 anchor_index=0, edge_weight=1., max_query_points=1_000_000,
                 normal_fn=vertex_normals, source_normals=None):
    """Direct, two-hop and return-cycle queries on fixed material vertex IDs.

    times must contain >=3 sorted real times. Only nonanchor sources are
    composed. Returns all queried trajectories plus same-query averaging and a
    no-extra-query temporal smoothing control. No GT input is accepted.
    """
    vertices=np.asarray(anchor_vertices,dtype=float);faces=np.asarray(faces)
    # Validate topology even when caller supplies a different normal convention.
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all()
            or faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in "iu"
            or len(faces)==0 or faces.min()<0 or faces.max()>=len(vertices)):
        raise ValueError("invalid mesh")
    times=np.asarray(times,dtype=float);ids=np.asarray(query_ids)
    if (times.ndim!=1 or len(times)<3 or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or not isinstance(anchor_index,(int,np.integer)) or not 0<=anchor_index<len(times)
            or ids.ndim!=1 or len(ids)==0 or ids.dtype.kind not in "iu"
            or ids.min()<0 or ids.max()>=len(vertices) or len(np.unique(ids))!=len(ids)
            or not isinstance(max_query_points,(int,np.integer)) or max_query_points<1):
        raise ValueError("invalid times/IDs/budget")
    normals=np.asarray(normal_fn(vertices,faces) if source_normals is None else source_normals,dtype=float)
    if (normals.shape!=vertices.shape or not np.isfinite(normals).all()
            or not np.allclose(np.linalg.norm(normals,axis=1),1.,rtol=0,atol=1e-4)):
        raise ValueError("invalid source normals")
    selected_faces=faces[np.any(np.isin(faces,ids),axis=1)]
    support=np.unique(selected_faces)
    if not np.isin(ids,support).all():
        raise ValueError("query vertex has no incident face")
    remap=np.full(len(vertices),-1,int);remap[support]=np.arange(len(support))
    local_faces=remap[selected_faces];local_ids=remap[ids]
    t=len(times);sources=[i for i in range(t) if i!=anchor_index]
    planned=len(support)*(t-1)+len(ids)*(t-1)*(t-1)
    if planned>max_query_points:
        raise ValueError(f"query-point budget exceeded: need {planned}, cap {max_query_points}")
    records=[]
    def call(source,targets,xyz,nrm,identities):
        result=np.asarray(decode(float(times[source]),times[targets].copy(),
                                 xyz.copy(),nrm.copy(),identities.copy()),dtype=float)
        if result.shape!=(len(targets),len(identities),3) or not np.isfinite(result).all():
            raise ValueError("decoder returned invalid absolute positions")
        records.append({"source_index":int(source),"target_indices":list(map(int,targets)),
                        "points":len(identities),"point_targets":len(targets)*len(identities)})
        return result
    support_direct=np.empty((t,len(support),3))
    support_direct[anchor_index]=vertices[support]
    support_direct[sources]=call(anchor_index,sources,vertices[support],normals[support],support)
    direct=support_direct[:,local_ids].copy()
    edges=[];predictions=[];cycles=[];compositions=[];averages=[[direct[i]] for i in range(t)]
    for source in sources:
        intermediate_normals=np.asarray(normal_fn(support_direct[source],local_faces),dtype=float)
        if (intermediate_normals.shape!=(len(support),3) or not np.isfinite(intermediate_normals).all()
                or not np.allclose(np.linalg.norm(intermediate_normals[local_ids],axis=1),1.,rtol=0,atol=1e-4)):
            raise ValueError("normal_fn returned invalid intermediate normals")
        target_ids=[j for j in range(t) if j!=source]
        values=call(source,target_ids,direct[source],intermediate_normals[local_ids],ids)
        for j,value in zip(target_ids,values):
            predictions.append({"source":int(source),"target":int(j),"positions":value})
            edges.append((source,j,value-direct[source]))
            if j==anchor_index:
                cycles.append(float(np.sqrt(np.mean((value-direct[j])**2))))
            else:
                compositions.append(float(np.sqrt(np.mean((value-direct[j])**2))))
                averages[j].append(value)
    average=np.stack([np.mean(a,axis=0) for a in averages]);average[anchor_index]=direct[anchor_index]
    consensus=anchored_consensus(direct,edges,anchor_index=anchor_index,weight=edge_weight)
    smooth_edges=[(i,i+1,np.zeros_like(direct[0])) for i in range(t-1)]
    smoothing=anchored_consensus(direct,smooth_edges,anchor_index=anchor_index,weight=edge_weight,
                                 edge_weights=1./np.diff(times)**2)
    return {"direct":direct,"composed":predictions,"consensus":consensus,
            "multi_reference_average":average,"smoothing":smoothing,
            "max_cycle_residual":max(cycles,default=0.),
            "max_composition_residual":max(compositions,default=0.),
            "query_ids":ids.copy(),"support_ids":support,"query_points":planned,
            "decoder_calls":len(records),"query_log":records,"times":times.copy(),
            "normal_convention":"caller supplied normal_fn; default area weighted",
            "smoothing_penalty":"sum squared adjacent velocity using supplied real time intervals",
            "natural_quality_claim":False}


def demo():
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    def rigid(source,targets,positions,normals,ids):
        return np.stack([positions+[t-source,0.,0.] for t in targets])
    def inconsistent(source,targets,positions,normals,ids):
        return np.stack([positions+[t-source+.05*(t-source)**2,0.,0.] for t in targets])
    good=probe_cycles(rigid,vertices,faces,[0.,2.,5.],[0,1])
    bad=probe_cycles(inconsistent,vertices,faces,[0.,2.,5.],[0,1])
    return {"method":2,"evidence":"constructed callback flows, not native quality",
            "rigid_cycle_residual":good["max_cycle_residual"],
            "inconsistent_cycle_residual":bad["max_cycle_residual"],
            "inconsistent_composition_residual":bad["max_composition_residual"],
            "query_points_per_arm":bad["query_points"],"decoder_calls_per_arm":bad["decoder_calls"],
            "anchor_exact":bool(np.array_equal(bad["consensus"][0],vertices[[0,1]])),
            "native_validation_required":True}
