#!/bin/bash

VERSION=1.2

usage() { echo "Usage: $0 [-p (push)]" 1>&2; exit 1; }

push=0
while getopts ":p" o; do
    echo ${OPTARG}
    case "${o}" in
        p)
            push=1
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

docker build . \
    --tag "andretelfer/hohlbaum-black-mouse-dataset-pytorch:$VERSION" \
    --tag "andretelfer/hohlbaum-black-mouse-dataset-pytorch:latest"

if [ ${push} == 1 ]; then
    docker push \
        "andretelfer/hohlbaum-black-mouse-dataset-pytorch:$VERSION" \
        "andretelfer/hohlbaum-black-mouse-dataset-pytorch:latest"
fi


