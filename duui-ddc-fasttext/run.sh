export JAVA_HOME=/usr/lib64/jvm/java-11-openjdk-11/
/usr/lib64/jvm/java-11-openjdk-11/bin/java -jar target/quarkus-app/quarkus-run.jar

# Docker
docker run --rm -p 8080:8080 --name ddc-fasttext-1 textimager-uima-ddc-fasttext-service:0.1
#docker run --rm -p 8080:8080 --name ddc-fasttext-1 141.2.89.20:5000/textimager-uima-ddc-fasttext-service:0.1
