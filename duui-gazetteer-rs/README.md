# gazetteer-rs 

DUUI Bindings for [`gazetteer-rs`](https://github.com/texttechnologylab/gazetteer-rs/).

## Available Variants

Latest version: `1.3.5`

```shell
docker.texttechnologylab.org/gazetteer-rs/biofid:latest
docker.texttechnologylab.org/gazetteer-rs/biofid-habitat:latest
docker.texttechnologylab.org/gazetteer-rs/geonames:latest
docker.texttechnologylab.org/gazetteer-rs/gnd:latest
```
## How To Use

### Use as Stand-Alone-Image
```shell
docker run docker.texttechnologylab.org/gazetteer-rs/biofid:1.3.5
```

### Run on a specific port (8001)
```shell
docker run -p 8001:80 docker.texttechnologylab.org/gazetteer-rs/biofid:1.3.5
```

### Run within DUUI

#### From the Image
```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/gazetteer-rs:1.3.5")
        .withScale(iWorkers)
        .withImageFetching()
);
```

#### With an Independent Remote Service

```java
composer.addDriver(new DUUIRemoteDriver(10000));
composer.add(
    new DUUIRemoteDriver.Component("http://localhost:8001")
);
```

## How To Build

There are multiple [variants](/variants) of the gazetteer, each backed by different lists and different target classes.
To build a custom gazetteer you will need to:

1. Set the desired target class of your found search terms in the communication layer.
2. Adopt the Dockerfile to your needs.
3. Run the build script.

### Communication Layer

```lua
-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
--  - parameters: A map of optional parameters
function serialize(inputCas, outputStream, parameters)
    -- Get data from CAS
    local doc_text = inputCas:getDocumentText();
    if parameters ~= nil then
        local max_len = parameters:get("max_len")
        local result_selection = parameters:get("result_selection")
        -- Encode data as JSON object and write to stream
        if max_len ~= nil and result_selection ~= nil then
            outputStream:write(json.encode({
                text = doc_text,
                max_len = max_len,
                result_selection = result_selection,
            }))
            return
        elseif max_len ~= nil then
            outputStream:write(json.encode({
                text = doc_text,
                max_len = max_len,
            }))
            return
        elseif result_selection ~= nil then
            outputStream:write(json.encode({
                text = doc_text,
                result_selection = result_selection,
            }))
            return
        end
    end
    -- Encode data as JSON object and write to stream
    outputStream:write(json.encode({
        text = doc_text
    }))
end

-- This "deserialize" function is called on receiving the results from the annotator that have to be transformed into a CAS object
-- Inputs:
--  - inputCas: The actual CAS object to deserialize into
--  - inputStream: Stream that is received from to the annotator, can be e.g. a string, JSON payload, ...
function deserialize(inputCas, inputStream)
    -- Get string from stream, assume UTF-8 encoding
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)

    -- Parse JSON data from string into object
    local results = json.decode(inputString)

    -- Add Taxa
    for i, match in ipairs(results) do
        local taxon = luajava.newInstance("org.texttechnologylab.annotation.type.Taxon", inputCas)
        taxon:setValue(match["match_strings"])
        taxon:setIdentifier(match["match_labels"])
        taxon:setBegin(match["begin"])
        taxon:setEnd(match["end"])
        taxon:addToIndexes()
    end

end
```

### Dockerfile

```dockerfile
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y wget unzip

# The gazetteer configuration file
COPY config.toml /app/
# The server configuration file
COPY Rocket.toml /app/
# The communication layer
COPY communication_layer.lua /app/
# Copy your lists and other data
COPY resources/ /app/resources/

# The gazetteer version. Use `docker build --build-args VERSION="vX.Y.Z" or change the default value here
ARG VERSION="v1.3.5"

# Download latest gazetteer-rs binary from GitHub
WORKDIR /app
RUN wget "https://github.com/texttechnologylab/gazetteer-rs/releases/download/${VERSION}/gazetteer"
RUN chmod +x /app/gazetteer

EXPOSE 9714

ENV ROCKET_ADRESS="0.0.0.0"
ENV ROCKET_PORT=9714

# You can optionally pass the path to the configuration file as an argument, if it is not "config.toml".
CMD ./gazetteer
```

### Build Script
With some `VARIANT_NAME` and the [latest](https://github.com/texttechnologylab/gazetteer-rs/releases/latest) gazetteer version **without** any `v*` prefix:
```shell
export VERSION=1.3.5

docker build --build-arg VERSION="v${VERSION}" -t gazetteer-rs:${VERSION}-gnd .
docker image tag gazetteer-rs:${VERSION}-gnd docker.texttechnologylab.org/gazetteer-rs:latest-VARIANT_NAME
docker push docker.texttechnologylab.org/gazetteer-rs:latest-VARIANT_NAME
docker image tag gazetteer-rs:${VERSION}-VARIANT_NAME docker.texttechnologylab.org/gazetteer-rs:${VERSION}-VARIANT_NAME
docker push docker.texttechnologylab.org/gazetteer-rs:${VERSION}-VARIANT_NAME
```
