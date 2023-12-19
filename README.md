# cryoem-services
Services and configuration for cryo-EM pipelines.

This package consists of a number of services to process cryo-EM micrographs,
both for single particle analysis and tomography.
These services can be run independently to process data,
or as part of a wider structure for performing live analysis during microscope collection.
For live analysis, this package integrates with a package
for transferring and monitoring collected data,
[Murfey](https://github.com/DiamondLightSource/python-murfey),
and a database for storing processing outcomes,
[ISPyB](https://github.com/DiamondLightSource/ispyb-database).


# Tomography processing

The tomography processing pipeline consists of:
- Motion correction
- CTF estimation
- Tomogram alignment
- Tomogram denoising


# Single particle analysis

The single particle analysis pipeline produces a project
that can be opened and continued using
[CCP-EM doppio](https://www.ccpem.ac.uk/docs/doppio/user_guide.html)
or [Relion](https://relion.readthedocs.io).


# Services currently available

The following services are provided for running the pipelines:
- Utility services:
    - **Dispatcher**: Converts recipes into messages suitable for processing services
    - **Images**: Creates thumbnail images for viewing processing outcomes
    - **ISPyB**: Inserts results into an ISPyB database
- Processing services:
    - **MotionCorr**: Motion correction of micrographs
    - **CTFFind**: CTF estimation on micrographs
    - **TomoAlign**: Tomogram reconstruction from a list of micrographs


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

