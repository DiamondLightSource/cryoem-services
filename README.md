# cryoem-services

Services and configuration for cryo-EM pipelines.

This package consists of a number of services to process cryo-EM micrographs,
both for single particle analysis and tomography,
using a range of commonly used cryo-EM processing software.
These services can be run independently to process data,
or as part of a wider structure for performing live analysis during microscope collection.
For live analysis, this package integrates with a package
for transferring and monitoring collected data,
[Murfey](https://github.com/DiamondLightSource/python-murfey),
and a database for storing processing outcomes,
[ISPyB](https://github.com/DiamondLightSource/ispyb-database).

To run these services the software executables being called must be installed.
These do not come with this package.

# Tomography processing

The tomography processing pipeline consists of:

- Motion correction
- CTF estimation
- Tomogram alignment
- Tomogram denoising using [Topaz](http://topaz-em.readthedocs.io)
- Segmentation using [membrain-seg](https://github.com/teamtomo/membrain-seg)

The results of this processing can be opened and continued using
[Relion 5.0](https://relion.readthedocs.io).

# Single particle analysis

The single particle analysis pipeline produces a project
that can be opened and continued using
[CCP-EM doppio](https://www.ccpem.ac.uk/docs/doppio/user_guide.html)
or [Relion](https://relion.readthedocs.io).

The processing pipeline consists of:

- Motion correction
- CTF estimation
- Particle picking
- (Optionally) Ice thickness estimation
- Particle extraction and rebatching
- 2D classification using Relion
- Automated 2D class selection using Relion
- 3D classification using Relion
- 3D Refinement and post-processing
- BFactor estimation by refinement with varying particle count

# Services currently available

The following services are provided for running the pipelines:

- Utility services:
  - **ClusterSubmission**: Submits zocalo wrappers to an HPC cluster
  - **Dispatcher**: Converts recipes into messages suitable for processing services
  - **Images**: Creates thumbnail images for viewing processing outcomes
  - **ISPyB**: Inserts results into an ISPyB database
  - **NodeCreator**: Creates Relion project files for the services run
- Processing services:
  - **BFactor**: Performs the setup for 3D refinement with varying particle count
  - **CrYOLO**: Particle picking on micrographs using [crYOLO](https://cryolo.readthedocs.io)
  - **CTFFind**: CTF estimation on micrographs using [CTFFIND4](https://grigoriefflab.umassmed.edu/ctffind4)
  - **DenoiseSlurm**: Tomogram denoising, submitted to a slurm HPC cluster, using [Topaz](http://topaz-em.readthedocs.io)
  - **Extract**: Extracts picked particles from micrographs
  - **ExtractClass**: Extracts particles from a given 3D class
  - **IceBreaker**: Ice thickness estimation with [IceBreaker](https://github.com/DiamondLightSource/python-icebreaker)
  - **MembrainSeg**: Tomogram segmentation, submitted to a slurm HPC cluster, using [membrain-seg](https://github.com/teamtomo/membrain-seg)
  - **MotionCorr**: Motion correction of micrographs using [MotionCor2](http://emcore.ucsf.edu/ucsf-software) or [Relion](https://relion.readthedocs.io), optionally submitted to a slurm HPC cluster
  - **PostProcess**: Post-processing of 3D refinements using [Relion](https://relion.readthedocs.io)
  - **SelectClasses**: Runs automated 2D class selection using [Relion](https://relion.readthedocs.io) and re-batches the particles from these classes
  - **SelectParticles**: Creates files listing batches of extracted particles
  - **TomoAlign**: Tomogram reconstruction from a list of micrographs using [imod](https://bio3d.colorado.edu/imod) and [AreTomo2](https://github.com/czimaginginstitute/AreTomo2)
  - **TomoAlignSlurm**: Tomogram alignment processing submitted to a slurm HPC cluster

There are also three zocalo wrapper scripts that can be run on an HPC cluster.
These perform 2D classification, 3D classification and 3D refinement
using [Relion](https://relion.readthedocs.io).

# Running services using zocalo

The services in this package are run using
[zocalo](https://github.com/DiamondLightSource/python-zocalo)
and [python-workflows](https://github.com/DiamondLightSource/python-workflows).
To start a service run the `zocalo.service` command and specify the service name.
For example, to start a motion correction service:

```bash
$ zocalo.service -s MotionCorr
```

Once started, these services will initialise and then wait for messages to be sent to them.
Messages are sent through a message broker,
currently [RabbitMQ](http://www.rabbitmq.com) is supported using pika transport in `python-workflows`.
Individual processing stages can be run by sending a dictionary of the parameters,
but the processing pipelines are designed to run through recipes.

A recipe is a specication of a series of steps to carry out,
and how these steps interact with each other.
Recipes for the current processing pipelines are provided in the `recipes` folder.

To run a recipe in python a dictionary needs to be provided consisting of
the recipe name and the parameters expected by the recipe.
The following snippet shows an example of the setup needed.
This will send a message to a running **Dispatcher** service which
prepares the recipe for the processing services.

```python
import workflows.transport.pika_transport as pt

example_message = {
    "recipes": ["em-tomo-align"],
    "parameters": {
        "path_pattern": "micrograph_*.mrc",
        "pix_size": "1",
        ...
    },
}

transport = pt.PikaTransport()
transport.connect()
transport.send("processing_recipe", example_message)
```
