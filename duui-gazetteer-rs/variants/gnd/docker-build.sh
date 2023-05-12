export VERSION=1.5.4
export NAME=gnd

rsync -ahz --info=Progress2 sirao:/resources/nlp/gazetteer_lists/filter_de.txt resources/
if [ ! -f resources/Kern-Personennamen.lex.gazetteer.v2.json.txt.gz ]; then
  rsync -ahz --info=Progress2 sirao:/resources/corpora/GND/Kern-Personennamen.lex.gazetteer.v2.json.txt resources/;
  pigz resources/Kern-Personennamen.lex.gazetteer.v2.json.txt || gzip resources/Kern-Personennamen.lex.gazetteer.v2.json.txt;
fi;

docker build --build-arg VERSION="${VERSION}" -t gazetteer-rs/${NAME}:${VERSION} .
docker image tag gazetteer-rs/${NAME}:${VERSION} docker.texttechnologylab.org/gazetteer-rs/${NAME}:latest
docker image tag gazetteer-rs/${NAME}:${VERSION} docker.texttechnologylab.org/gazetteer-rs/${NAME}:${VERSION}
docker push docker.texttechnologylab.org/gazetteer-rs/${NAME}:latest
docker push docker.texttechnologylab.org/gazetteer-rs/${NAME}:${VERSION}
