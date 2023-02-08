|Apache license|

Code for the paper "Inference of Infrastructure Network Flows via Physics-Inspired Implicit Neural Networks".

HOW TO USE THE PACKAGE
-----

The package is currently not distributed anywhere.

After downloading it, you can do the following:


Option 1 (recommended): use docker
-----

A `docker compose <docker/docker-compose.yml>`_ file is provided, and can be launched and run in interactive mode as follows:

.. code-block:: bash

    $ docker compose -f docker/docker-compose.yml up -d
    $ docker exec -it <container_name> /bin/bash

where ``<container_name>`` is the name of the container created with the first command.
It can be retrieved via ``docker container ls``.
The image in the compose file corresponds to this `Dockerfile <docker/Dockerfile>`_, and contains all the main dependencies needed.
Clearly, (a reasonably up-to-date version of) `docker <https://www.docker.com/>`_ is required to be installed.

If the user wants to enable NVIDIA GPU support inside the container, replace the compose step above with ``docker compose -f docker/docker-compose.yml -f docker/docker-compose-gpu.yml up -d`` (tested only on a host with Ubuntu installed).
It is assumed the appropriate NVIDIA drivers as well as an appropriate version of `nvidia-docker2 <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html>`_ is installed.

On the first run, it is necessary to additionally install the package by executing (in the container):

.. code-block:: bash

    $ cd /content
    $ pip3 install --upgrade pip
    $ pip3 install -e .

Option 2: install on host machine
-----

1. Create virtual environment:

.. code-block:: bash

    $ cd <project-home>
    $ python3 -m venv .env

2. Activate virtual environment

.. code-block:: bash

    $ source .env/bin/activate

3. Update the pip distribution (to avoid potential issues)

.. code-block:: bash

    $ pip install --upgrade pip

4. Install package for testing in the virtual environment

.. code-block:: bash

    $ python3 setup.py develop


Together with an appropriate handling of `pytorch-geometric`, allowing to optionally specify torch and cuda versions, a pip-based installation replacing steps 3-4 has been included in the `setup.sh` script for your convenience:

.. code-block:: bash

    $ ./setup.sh -T <torch version> -C <cuda version>

---------------

RUNNING THE EXPERIMENTS
-----

After doing the steps above, do

.. code-block:: bash

    $ pip3 install pickle5
    $ cd experiments
    $ python3 experiments.py --exp <exp-name>

with `<exp_name>` being either `traffic` or `power`.

Data is taken from https://openreview.net/attachment?id=l0V53bErniB&name=supplementary_material

---------------

DEPENDENCIES
-----

The main package dependencies are pytorch_, `pytorch-geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_, qpth_, networkx_.

---------------

HOW TO CITE
-----

.. code-block:: bash

    @article{FS-FB-AKS:23,
        title={Inference of Infrastructure Network Flows via Physics-Inspired Implicit Neural Networks},
        author={Seccamonte, Francesco and Singh, Ambuj K. and Bullo, Francesco},
        journal={Under review},
        year={2023}
    }

.. _pytorch: https://pytorch.org
.. _qpth: https://locuslab.github.io/qpth/
.. _networkx: https://networkx.org

.. |Apache license| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: LICENSE
