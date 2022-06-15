VERSION=1.0

docker build . \
    --memory 26GB \
    --tag "andretelfer/hohlbaum-black-mouse-dataset-pytorch:$VERSION"
