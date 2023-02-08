Docker commands
========================

It assumes a reasonably recent version of docker is installed.

To start the service, from the project root folder run ``docker compose -f docker/docker-compose.yml up -d``.

If GPU support inside the container is required, replace the compose step above with ``docker compose -f docker/docker-compose.yml -f docker/docker-compose-gpu.yml up -d``.
It is assumed the appropriate NVIDIA drivers as well as an appropriate version of `nvidia-docker2 <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html>`_ is installed.

To launch/relaunch a container in interactive mode, run ``docker exec <container name> /bin/bash``.
