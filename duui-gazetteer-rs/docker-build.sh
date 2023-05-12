export VERSION=1.5.4
export NAME=base

docker build --build-arg VERSION="v${VERSION}" -t gazetteer-rs/${NAME}:${VERSION} .
docker image tag gazetteer-rs/${NAME}:${VERSION} docker.texttechnologylab.org/gazetteer-rs/${NAME}:latest
docker image tag gazetteer-rs/${NAME}:${VERSION} docker.texttechnologylab.org/gazetteer-rs/${NAME}:${VERSION}
docker push docker.texttechnologylab.org/gazetteer-rs/${NAME}:latest
docker push docker.texttechnologylab.org/gazetteer-rs/${NAME}:${VERSION}
