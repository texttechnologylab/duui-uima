FROM maven:3.9-eclipse-temurin-21 AS java-build
WORKDIR /build
COPY pom.xml .
COPY src/main/java/ src/main/java/
RUN mvn -q -DskipTests package dependency:copy-dependencies \
    -DincludeScope=runtime -DoutputDirectory=/build/lib

FROM python:3.11-slim

RUN apt-get update && apt-get install --yes --no-install-recommends ffmpeg openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/main/python/ .
COPY --from=java-build /build/target/classes /app/java/classes
COPY --from=java-build /build/lib /app/java/lib

EXPOSE 9715
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9715"]
