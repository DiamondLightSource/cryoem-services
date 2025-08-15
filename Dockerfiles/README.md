# Dockerfiles for services

The following services rely on the `cryoemservices_cpu` dockerfile:

- BFactor
- CLEMAlignAndMerge
- CLEMLIFToStack
- CLEMTIFFToStack
- ClusterSubmission
- DenoiseSlurm
- EMISPyB
- Extract
- ExtractClass
- IceBreaker
- Images
- NodeCreator
- ProcessRecipe
- SelectParticles
- TomoAlignSlurm

The remaining services that have dockerfiles are:

- MotionCorr: build using `motioncor2` or `motioncor_relion`
- CTFFind: build using `ctffind`
- CrYOLO: build using `cryolo`
- SelectClasses: build using `class_selection`
- PostProcess: build using `motioncor_relion`
- TomoAlign: build using `tomo_align`
- Denoise: build using `topaz`
- MembrainSeg: build using `topaz`

The `motioncor2`, `tomo_align` and `topaz` Dockerfiles have GPU support.

Some dependencies are not built in the dockerfiles,
and are instead assumed to be in a "packages" subfolder

- AreTomo2
- ctffind-4.1.14
- ctffind-5.0.2
- motioncor-1.4.0

For `cryolo`, the `gmodel_phosnet_202005_N63_c17.h5` also needs to be present in a "cryolo_models" folder
