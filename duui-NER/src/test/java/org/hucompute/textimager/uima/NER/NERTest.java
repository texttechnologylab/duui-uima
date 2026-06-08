package org.hucompute.textimager.uima.NER;

import de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
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

import static org.junit.jupiter.api.Assertions.*;

public class NERTest {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:8000";

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

    public void createCasFromSentences(String language, List<String> sentences) {
        cas.setDocumentLanguage(language);

        StringBuilder documentText = new StringBuilder();
        int offset = 0;

        for (int i = 0; i < sentences.size(); i++) {
            String sentenceText = sentences.get(i);

            if (i > 0) {
                documentText.append("\n");
                offset += 1;
            }

            int begin = offset;
            int end = begin + sentenceText.length();

            documentText.append(sentenceText);

            Sentence sentence = new Sentence(cas, begin, end);
            sentence.addToIndexes();

            offset = end;
        }

        cas.setDocumentText(documentText.toString());
    }

    @Test
    public void DeBioFidNERTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter(
                                "selection",
                                "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
                        )
                        .withParameter("threshold", "0.5")
                        .withParameter("batch_size", "8")
                        .withParameter("labels", "person,organization,location,date,event,product,taxon,other")
        );

        List<String> textes = Arrays.asList(
                "Dr. Anna Weber untersuchte für BioFID eine Streuobstwiese bei Frankfurt am Main und einen Buchenwald im Taunus.",
                "Auf der Wiese fand sie Apis mellifera, Bombus terrestris, Papilio machaon und Vanessa atalanta.",
                "Der Schwalbenschwanz flog über Daucus carota und setzte sich kurz auf eine Blüte.",
                "Er wurde fotografiert und später im Protokoll erwähnt.",
                "Im Wald standen Fagus sylvatica, Quercus robur und Acer pseudoplatanus.",
                "Eine alte Buche trug Flechten wie Xanthoria parietina und Parmelia sulcata.",
                "Sie war teilweise abgestorben, bot aber vielen Insekten Lebensraum.",
                "Unter ihrer Rinde fanden die Forschenden Spuren von Lucanus cervus.",
                "An einem Bach beobachtete das Team Salamandra salamandra und Rana temporaria.",
                "Der Feuersalamander kroch langsam über den feuchten Weg.",
                "Er verschwand unter einem Stein, nachdem Dr. Weber ihn fotografiert hatte.",
                "Später verglich sie die Funde aus Frankfurt am Main, dem Taunus und dem Bodensee.",
                "In ihrem Bericht wurden wissenschaftliche Namen, Ortsnamen und Koreferenzen markiert.",
                "Die Pipeline sollte erkennen, dass Fagus sylvatica, Papilio machaon, Salamandra salamandra und Lucanus cervus Taxa sind, während Frankfurt am Main, Taunus und Bodensee geographische Namen sind."
        );

        createCasFromSentences("de", textes);

        System.out.println("Input document:");
        System.out.println(cas.getDocumentText());

        composer.run(cas);

        Collection<NamedEntity> namedEntities = JCasUtil.select(cas, NamedEntity.class);
        Map<String, ArrayList<Object>> result = extractNERResult();

        for (NamedEntity namedEntity : namedEntities) {
            String coveredText = namedEntity.getCoveredText();
            int begin = namedEntity.getBegin();
            int end = namedEntity.getEnd();
            String value = namedEntity.getValue();
            String typeName = namedEntity.getType().getName();

            result.get("token").add(coveredText);
            result.get("begin").add(begin);
            result.get("end").add(end);
            result.get("value").add(value);
            result.get("type").add(typeName);

            System.out.println(
                    "NER: '" + coveredText + "'"
                            + " (begin=" + begin + ", end=" + end + ")"
                            + " value='" + value + "'"
                            + " type='" + typeName + "'"
            );

            assertTrue(begin >= 0, "NER begin offset must be non-negative");
            assertTrue(end > begin, "NER end offset must be greater than begin offset");
            assertFalse(coveredText.isBlank(), "NER covered text must not be blank");
        }

        assertFalse(namedEntities.isEmpty(), "The DUUI NER component should create at least one NamedEntity annotation.");
    }

    private Map<String, ArrayList<Object>> extractNERResult() {
        Map<String, ArrayList<Object>> result = new LinkedHashMap<>();

        result.put("begin", new ArrayList<>());
        result.put("end", new ArrayList<>());
        result.put("token", new ArrayList<>());
        result.put("value", new ArrayList<>());
        result.put("type", new ArrayList<>());

        return result;
    }
}