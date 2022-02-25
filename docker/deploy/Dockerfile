FROM ubuntu:20.04
LABEL maintainer="Reto Achermann <achreto@gmail.com>"

# copy the scripts
COPY docker/deploy/build.sh /build.sh
COPY docker/deploy/entrypoint.sh /entrypoint.sh

# add the files
ADD . /gosdt-src

# run the setup script
RUN /bin/bash /build.sh

# entrypoint for the docker image
ENTRYPOINT ["/entrypoint.sh"]