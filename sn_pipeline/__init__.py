"""
sn_pipeline - automated segmentation of the neuromelanin-bright substantia nigra (left / right)
in native T1W_3D_SPIR space.

Pipeline flow (see README.md for the full description):
    dataset audit -> subject table and header geometry -> rigid T1w -> SPIR registration
    -> CIT168 atlas registration into SPIR space -> crop, normalisation and mask preparation
    -> GMM label refinement (subjects with a manual mask) -> nnU-Net training (only if no trained model is given)
    -> nnU-Net inference and post-processing -> visual QC -> final summary
"""
