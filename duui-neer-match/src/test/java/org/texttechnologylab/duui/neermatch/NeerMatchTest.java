package org.texttechnologylab.duui.neermatch;

import com.google.gson.*;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIPodmanDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.IDUUIInstantiatedPipelineComponent;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.*;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

public class NeerMatchTest {

    static final String NEER_MATCH_IMAGE = "docker.texttechnologylab.org/duui-neer-match:latest";

    DUUIComposer composer;
    JCas cas1;
    JCas cas2;
    String pipelineId;

    @BeforeEach
    public void setup() throws IOException, URISyntaxException, UIMAException, SAXException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        composer.addDriver(new DUUIPodmanDriver());

        cas1 = JCasFactory.createJCas();
        cas1.setDocumentLanguage("de");
        cas1.setDocumentText("Der erste Text ist dieser Text hier. Er ist anders als der zweite, aber nur etwas.");

        cas2 = JCasFactory.createJCas();
        cas2.setDocumentLanguage("de");
        cas2.setDocumentText("Der zweite Text ist jener Text hier. Er ist nur etwas anders als der erste, ein klein wenig.");

        pipelineId = UUID.randomUUID().toString();
    }

    @Test
    public void testTokens() throws Exception {

        composer.add(
                new DUUIPodmanDriver.Component("docker.texttechnologylab.org/textimager-duui-spacy:0.4.0").withImageFetching()
                        .build());
        composer.add(new DUUIPodmanDriver.Component(NEER_MATCH_IMAGE)
                .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
                .withParameter("pipeline_id", pipelineId)
                .withName("neer_match")
                .build()
        );
        composer.run(cas1);
        composer.run(cas2);

        JsonObject result = finalizeNeerMatch("token1", 0.01);
        Gson gson = new GsonBuilder()
                .setPrettyPrinting()
                .create();
        System.out.println(gson.toJson(result));
    }

    @Test
    public void testTokensImage() throws Exception {

        composer.add(
                new DUUIPodmanDriver.Component("docker.texttechnologylab.org/textimager-duui-spacy:0.4.0").withImageFetching()
                        .build());
        composer.add(new DUUIPodmanDriver.Component(NEER_MATCH_IMAGE)
                .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
                .withParameter("pipeline_id", pipelineId)
                .withName("neer_match")
                .build()
        );
        composer.run(cas1);
        composer.run(cas2);

        JsonObject result = finalizeNeerMatch("token1", 0.0);
        NeerMatchResult matchResult = parseNeerMatchResult(result);
        assertNotNull(matchResult);
        assertEquals(1, matchResult.results().length);
        MatchResult firstMatch = matchResult.results()[0];
        assertEquals(0, firstMatch.document1Id());
        assertEquals(1, firstMatch.document2Id());
        MatchedEntity[] matches = firstMatch.matches();
        assertTrue(matches.length > 0);
        BufferedImage image = drawSimilarityMatrix(matches, 20, BufferedImage.TYPE_BYTE_GRAY, NeerMatchTest::bwGradientColorMapping);
        assertNotNull(image);
        ImageIO.write(image, "png", new File("similarity_matrix.png"));
    }

    static Color bwGradientColorMapping(double similarity) {
        int grayValue = (int) (similarity * 255);
        return new Color(0x010101 * grayValue);
    }

    @Test
    public void testNamedEntities() throws Exception {

        composer.add(
                new DUUIPodmanDriver.Component("docker.texttechnologylab.org/textimager-duui-spacy:0.4.0").withImageFetching()
                        .build());
        composer.add(new DUUIPodmanDriver.Component(NEER_MATCH_IMAGE)
                .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity")
                .withParameter("pipeline_id", pipelineId)
                .withName("neer_match")
                .build()
        );
        composer.run(cas1);
        composer.run(cas2);

        JsonObject result = finalizeNeerMatch("named_entity1", 0);
        Gson gson = new GsonBuilder()
                .setPrettyPrinting()
                .create();
        System.out.println(gson.toJson(result));
    }

    String retrieveNeerMatchContainerAddress() {
        // Currently it is not really supported to send http requests to images directly, so we have to extract the address using reflections
        // if using a remote driver this is not necessary because the address is already known, but this shows that it is even possible for docker driver
        // this code is not really part of the project and should be seen as a helper function, that does nothing actually with neer match
        DUUIComposer.PipelinePart neerMatchPart = composer.get_instantiatedPipeline().stream()
                .filter(component -> Objects.equals(component.getName(), "neer_match"))
                .findFirst()
                .orElseThrow(() -> new RuntimeException("NeerMatch component not found in pipeline"));
        if (neerMatchPart.getDriver() instanceof DUUIPodmanDriver driver) {
            try {
                Field activeComponentsField = DUUIPodmanDriver.class.getDeclaredField("_active_components");
                activeComponentsField.setAccessible(true);
                //noinspection unchecked
                Map<String, IDUUIInstantiatedPipelineComponent> activeComponents = (Map<String, IDUUIInstantiatedPipelineComponent>) activeComponentsField.get(driver);
                IDUUIInstantiatedPipelineComponent neerMatchComponent = activeComponents.get(neerMatchPart.getUUID());
                if (neerMatchComponent == null) {
                    throw new RuntimeException("NeerMatch component instance not found in active components");
                }
                Class<?> instantiatedComponentClass = Class.forName("org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver$InstantiatedComponent");
                Field instancesField = instantiatedComponentClass.getDeclaredField("_instances");
                instancesField.setAccessible(true);
                //noinspection unchecked
                ConcurrentLinkedQueue<DUUIDockerDriver.ComponentInstance> instances = (ConcurrentLinkedQueue<DUUIDockerDriver.ComponentInstance>) instancesField.get(neerMatchComponent);
                if (instances.size() != 1) {
                    throw new RuntimeException("Expected exactly one instance of NeerMatch component, found " + instances.size());
                }
                DUUIDockerDriver.ComponentInstance instance = instances.peek();
                if (instance == null) {
                    throw new RuntimeException("NeerMatch component instance is null");
                }
                return instance.generateURL();
            } catch (NoSuchFieldException | IllegalAccessException | ClassNotFoundException e) {
                throw new RuntimeException(e);
            }
        } else {
            throw new RuntimeException("NeerMatch component is not running on Podman");
        }
    }

    JsonObject finalizeNeerMatch(String model, double threshold) throws IOException, InterruptedException {
        String NEER_MATCH_URL = retrieveNeerMatchContainerAddress();
        String response;
        try (HttpClient client = HttpClient.newHttpClient()) {
            JsonObject requestBody = new JsonObject();
            requestBody.addProperty("pipeline_id", pipelineId);
            JsonObject requestProperties = new JsonObject();
            requestProperties.addProperty("model", model);
            requestProperties.addProperty("threshold", threshold);
            requestBody.add("properties", requestProperties);

            response = client.send(
                    HttpRequest.newBuilder()
                            .uri(URI.create(NEER_MATCH_URL + "/v1/finalize"))
                            .header("Content-Type", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString(requestBody.toString()))
                            .build(),
                    HttpResponse.BodyHandlers.ofString()
            ).body();
        }

        return JsonParser.parseString(response).getAsJsonObject();
    }

    static NeerMatchResult parseNeerMatchResult(JsonObject json) {
        JsonArray resultsArray = json.getAsJsonArray("results");
        MatchResult[] matchResults = new MatchResult[resultsArray.size()];
        Map<Integer, Document> documents = new HashMap<>();
        BiConsumer<Integer, JsonObject> storeEntity = (Integer documentId, JsonObject entityJson) -> {
            int entityId = entityJson.get("entity_id").getAsInt();
            Document doc = documents.computeIfAbsent(documentId, Document::new);
            if (doc.entities.containsKey(entityId)) {
                return; // Entity already stored
            }
            String text = entityJson.get("text").getAsString();
            JsonObject propertiesJson = entityJson.getAsJsonObject("properties");
            Map<String, Object> properties = new HashMap<>();
            if (propertiesJson != null) {
                for (Map.Entry<String, JsonElement> entry : propertiesJson.entrySet()) {
                    if (entry.getValue().isJsonPrimitive()) {
                        JsonPrimitive primitive = entry.getValue().getAsJsonPrimitive();
                        if (primitive.isString()) {
                            properties.put(entry.getKey(), primitive.getAsString());
                        } else if (primitive.isNumber()) {
                            properties.put(entry.getKey(), primitive.getAsNumber());
                        } else if (primitive.isBoolean()) {
                            properties.put(entry.getKey(), primitive.getAsBoolean());
                        }
                    } else {
                        System.err.println("Skipping non-primitive property: " + entry.getKey());
                    }
                }
            }
            doc.entities.put(entityId, new DocumentEntity(text, entityId, properties));
        };

        for (int i = 0; i < resultsArray.size(); i++) {
            JsonObject matchResultJson = resultsArray.get(i).getAsJsonObject();
            int document1Id = matchResultJson.get("document_1_index").getAsInt();
            int document2Id = matchResultJson.get("document_2_index").getAsInt();
            JsonArray matchesArray = matchResultJson.getAsJsonArray("predictions");
            MatchedEntity[] matches = new MatchedEntity[matchesArray.size()];
            for (int j = 0; j < matchesArray.size(); j++) {
                JsonObject matchJson = matchesArray.get(j).getAsJsonObject();
                double similarity = matchJson.get("score").getAsDouble();
                JsonObject entity1Json = matchJson.getAsJsonObject("document_1_entity");
                JsonObject entity2Json = matchJson.getAsJsonObject("document_2_entity");
                storeEntity.accept(document1Id, entity1Json);
                storeEntity.accept(document2Id, entity2Json);
                int entityId1 = entity1Json.get("entity_id").getAsInt();
                int entityId2 = entity2Json.get("entity_id").getAsInt();
                matches[j] = new MatchedEntity(entityId1, entityId2, similarity);
            }
            matchResults[i] = new MatchResult(document1Id, document2Id, matches);
        }
        return new NeerMatchResult(matchResults, documents);
    }

    static BufferedImage drawSimilarityMatrix(MatchedEntity[] matches, int cellSize, int imageType, Function<Double, Color> colorMapping) {
        List<Integer> entityIds1 = Arrays.stream(matches)
                .map(m -> m.entityId1)
                .distinct()
                .sorted()
                .toList();
        List<Integer> entityIds2 = Arrays.stream(matches)
                .map(m -> m.entityId2)
                .distinct()
                .sorted()
                .toList();
        TreeMap<Integer, Integer> entityIdToIndex1 = IntStream.range(0, entityIds1.size())
                .boxed()
                .collect(Collectors.toMap(entityIds1::get, i -> i, (a, b) -> a, TreeMap::new));
        TreeMap<Integer, Integer> entityIdToIndex2 = IntStream.range(0, entityIds2.size())
                .boxed()
                .collect(Collectors.toMap(entityIds2::get, i -> i, (a, b) -> a, TreeMap::new));
        int width = entityIds2.size() * cellSize;
        int height = entityIds1.size() * cellSize;
        BufferedImage image = new BufferedImage(width, height, imageType);
        Graphics2D g = image.createGraphics();
        g.setColor(colorMapping.apply(0.0)); // Background color for zero similarity
        g.fillRect(0, 0, width, height);
        for (MatchedEntity match : matches) {
            int entityId1 = match.entityId1;
            int entityId2 = match.entityId2;
            int entityIndex1 = entityIdToIndex1.get(entityId1);
            int entityIndex2 = entityIdToIndex2.get(entityId2);
            double similarity = match.similarity;
            Color color = colorMapping.apply(similarity);
            g.setColor(color);
            int x = entityIndex2 * cellSize;
            int y = entityIndex1 * cellSize;
            g.fillRect(x, y, cellSize, cellSize);
        }
        g.dispose();
        return image;
    }

    public record DocumentEntity(String text, int entityId, Map<String, Object> properties) {
    }

    public record Document(int documentId, Map<Integer, DocumentEntity> entities) {
        public Document(int documentId) {
            this(documentId, new HashMap<>());
        }
    }

    public record MatchedEntity(int entityId1, int entityId2, double similarity) {
    }

    public record MatchResult(int document1Id, int document2Id, MatchedEntity[] matches) {
    }

    public record NeerMatchResult(MatchResult[] results, Map<Integer, Document> documents) {
    }

}
