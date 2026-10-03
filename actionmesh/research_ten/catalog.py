"""Method identifiers and explicit non-oracle input prerequisites."""
METHODS = {
    1: dict(module="m01_elasticity", name="Observation-supported local elasticity",
            requires=["fixed topology sequence", "aligned input-observation residuals and confidence"]),
    2: dict(module="m02_cycles", name="Cycle-consistent frozen decoding",
            requires=["frozen decoder", "Stage-I latents", "anchor topology and time/vertex IDs"]),
    3: dict(module="m03_correspondence", name="Temporal multi-hypothesis correspondence",
            requires=["actual candidate matcher outputs", "stable source IDs", "candidate costs and validity"]),
    4: dict(module="m04_occlusion", name="Reappearance-constrained occlusion repair",
            requires=["observed tracks", "visibility and two valid endpoints", "surface neighbors"]),
    5: dict(module="m05_scale", name="Stable-region scale/deformation separation",
            requires=["known source geometry", "explicit validated stable-region mask", "shared camera convention"]),
    6: dict(module="m06_selection", name="Action-gated 3D-feasibility video selection",
            requires=["same-task candidate videos", "shared frozen action/text/identity scorer outputs", "2D tracking features"]),
    7: dict(module="m07_sampling", name="Budgeted motion-residual control allocation",
            requires=["anchor coordinates", "budgeted trajectory query callback", "sparse interpolation baseline"]),
    8: dict(module="m08_contact", name="Contact-normal trajectory repair",
            requires=["nonadjacent signed contact constraints", "fixed topology", "collision/contact eligibility"]),
    9: dict(module="m09_guidance", name="Mesh/video conflict projection",
            requires=["live three-branch CFG denoiser", "known-frame mask", "fresh Stage-I rollout"]),
    10: dict(module="m10_windows", name="Bidirectional overlapping-window conditioning",
             requires=["natural long video covering >=4 windows", "aligned latent representation", "fresh conditioned rollout callback"]),
}


def load_method(identifier):
    import importlib
    if identifier not in METHODS:
        raise ValueError("method must be in 1..10")
    return importlib.import_module("research_ten." + METHODS[identifier]["module"])
