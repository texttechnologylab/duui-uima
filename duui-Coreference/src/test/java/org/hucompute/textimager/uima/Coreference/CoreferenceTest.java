package org.hucompute.textimager.uima.Coreference;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.uima.fit.util.JCasUtil;
import org.texttechnologylab.uima.type.spacy.SpacyToken;
import org.texttechnologylab.annotation.Coreference;

import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XmlCasSerializer;

import org.junit.jupiter.api.*;

import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.*;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

public class CoreferenceTest {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:9714";

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException, SAXException, CompressorException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);

        cas = JCasFactory.createJCas();
    }

    @AfterAll
    static void afterAll() throws UnknownHostException {
        composer.shutdown();
    }

    @AfterEach
    public void afterEach() throws IOException, SAXException {
        composer.resetPipeline();

        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        XmlCasSerializer.serialize(cas.getCas(), null, stream);
        System.out.println(stream.toString(StandardCharsets.UTF_8));

        cas.reset();
    }

    public void createCas(
            String language,
            List<String> tokens,
            List<Integer> begins,
            List<Integer> ends
    ) throws UIMAException {
        validateInput(tokens, begins, ends);

        cas.setDocumentLanguage(language);

        String documentText = buildDocumentText(tokens, begins, ends);
        cas.setDocumentText(documentText);

        addTokens(tokens, begins, ends);
        addSentencesFromPunctuation(tokens, begins, ends);
    }

    private void addTokens(
            List<String> tokens,
            List<Integer> begins,
            List<Integer> ends
    ) {
        for (int i = 0; i < tokens.size(); i++) {
            SpacyToken token = new SpacyToken(cas, begins.get(i), ends.get(i));
            token.addToIndexes();
        }
    }

    private void addSentencesFromPunctuation(
            List<String> tokens,
            List<Integer> begins,
            List<Integer> ends
    ) {
        int sentenceBegin = begins.get(0);

        for (int i = 0; i < tokens.size(); i++) {
            String token = tokens.get(i);

            if (token.equals(".") || token.equals("!") || token.equals("?")) {
                int sentenceEnd = ends.get(i);

                Sentence sentence = new Sentence(cas, sentenceBegin, sentenceEnd);
                sentence.addToIndexes();

                if (i + 1 < tokens.size()) {
                    sentenceBegin = begins.get(i + 1);
                }
            }
        }
    }

    private String buildDocumentText(
            List<String> tokens,
            List<Integer> begins,
            List<Integer> ends
    ) {
        int documentLength = ends.get(ends.size() - 1);
        char[] chars = new char[documentLength];
        Arrays.fill(chars, ' ');

        for (int i = 0; i < tokens.size(); i++) {
            String token = tokens.get(i);
            int begin = begins.get(i);
            int end = ends.get(i);

            for (int j = 0; j < token.length(); j++) {
                chars[begin + j] = token.charAt(j);
            }
        }

        return new String(chars);
    }

    private void validateInput(
            List<String> tokens,
            List<Integer> begins,
            List<Integer> ends
    ) {
        assertEquals(tokens.size(), begins.size());
        assertEquals(tokens.size(), ends.size());

        for (int i = 0; i < tokens.size(); i++) {
            String token = tokens.get(i);
            int begin = begins.get(i);
            int end = ends.get(i);

            if (token.length() != end - begin) {
                throw new IllegalArgumentException(
                        "Token length does not match offsets at index " + i
                                + ": token='" + token + "'"
                                + ", begin=" + begin
                                + ", end=" + end
                );
            }

            if (i > 0 && begin < ends.get(i - 1)) {
                throw new IllegalArgumentException(
                        "Token offsets overlap at index " + i
                );
            }
        }
    }

    @Test
    public void EnTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter(
                                "selection",
                                "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
                        )
        );

        List<String> tokens = Arrays.asList(
                "Anna", "bought", "a", "cactus", ".",
                "The", "plant", "needed", "sunlight", ".",
                "She", "put", "a", "vase", "on", "the", "table", ".",
                "The", "vase", "was", "old", ",", "but", "it", "was", "beautiful", ".",
                "The", "cactus", "grew", "quickly", "because", "it", "got", "enough", "light", "."
        );

        List<Integer> begins = Arrays.asList(
                0, 5, 12, 14, 20,
                22, 26, 32, 39, 47,
                49, 53, 57, 59, 64, 67, 71, 76,
                78, 82, 87, 91, 94, 96, 100, 103, 107, 116,
                118, 122, 129, 134, 142, 150, 153, 157, 164, 169
        );

        List<Integer> ends = Arrays.asList(
                4, 11, 13, 20, 21,
                25, 31, 38, 47, 48,
                52, 56, 58, 63, 66, 70, 76, 77,
                81, 86, 90, 94, 95, 99, 102, 106, 116, 117,
                121, 128, 133, 141, 149, 152, 156, 163, 169, 170
        );

        createCas("en", tokens, begins, ends);

        System.out.println("Input document:");
        System.out.println(cas.getDocumentText());
        SpacyToken h = JCasUtil.selectAt(cas, SpacyToken.class, 0, 4).iterator().next();
        composer.run(cas);

        Collection<Coreference> coreferences = JCasUtil.select(cas, Coreference.class);
        Map<String, ArrayList<Object>> result = extractCoreferenceResult();
        for (Coreference coreference : coreferences) {
            String token = coreference.getCoveredText();
            int begin = coreference.getBegin();
            int end = coreference.getEnd();

            result.get("token").add(token);
            result.get("begin").add(begin);
            result.get("end").add(end);

            if (coreference.getLink() != null) {
                String token_resolve = coreference.getLink().getCoveredText();
                int begin_resolve = coreference.getLink().getBegin();
                int end_resolve = coreference.getLink().getEnd();
                System.out.println("Coreference: '" + token + "' (begin=" + begin + ", end=" + end + ")" + " -> '" + token_resolve + "' (begin=" + begin_resolve + ", end=" + end_resolve + ")");
            }

        }
    }

    private Map<String, ArrayList<Object>> extractCoreferenceResult() {
        Map<String, ArrayList<Object>> result = new LinkedHashMap<>();

        result.put("begin", new ArrayList<>());
        result.put("end", new ArrayList<>());
        result.put("begin_resolve", new ArrayList<>());
        result.put("end_resolve", new ArrayList<>());
        result.put("token", new ArrayList<>());
        result.put("token_resolve", new ArrayList<>());
        return result;
    }
}