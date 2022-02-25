# Deploy Docker Image

This docker image creates a container with GOSDT pre-installed including
its dependencies

# Building the Image

Execute the build script to create the docker image

```
    $ bash build-deploy-image.sh
```

Set the docker image name
```
export GOSDT_IMAGE=gosdt
```

# Pulling the image

```
    docker pull achreto/gosdt
```

Set the docker image name

```
export GOSDT_IMAGE=achreto/gosdt
```

**Note: It's compiled on x86 with at least Broadwell CPU generation. So it won't run on non-x86 hardware, and there could be issues with older CPUs**

# Running GOSDT

The infrastructure with datasets etc will be placed in `/gosdt` in the image.

## Interactive Mode

This opens a shell on the container

```
    docker run -i -t ${GOSDT_IMAGE}
```

## Execute Mode

You can pass the command to be executed as another parameter:

```
    docker run -i -t ${GOSDT_IMAGE} <COMMAND>
```
