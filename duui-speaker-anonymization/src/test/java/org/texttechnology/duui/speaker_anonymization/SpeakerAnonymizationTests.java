package org.texttechnology.duui.speaker_anonymization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;

import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.InvalidXMLException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

/**
 * Integration tests for duui-speaker-anonymization.
 * Requires the component running on http://127.0.0.1:9714.
 */
public class SpeakerAnonymizationTests {

    private static final String URL = System.getProperty(
            "duui.speaker.url",
            System.getenv().getOrDefault("DUUI_SPEAKER_ANONYMIZATION_URL", "http://127.0.0.1:9714"));
    private static final String WHISPERX_URL = System.getProperty(
            "duui.whisperx.url",
            System.getenv().getOrDefault("DUUI_WHISPERX_URL", "http://127.0.0.1:9710"));

    @TempDir
    Path outputDirectory;

    private DUUIComposer composer;
    private JCas cas;

    @BeforeEach
    void setUp() throws Exception {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver, new DUUIUIMADriver());
        composer.addDriver(new DUUIDockerDriver());
        cas = JCasFactory.createJCas();
    }

    @AfterEach
    void tearDown() throws Exception {
        composer.shutdown();
        cas.reset();
    }

    private void addComponent(String language)
            throws InvalidXMLException, IOException, URISyntaxException, SAXException, CompressorException {
        composer.add(
            new DUUIRemoteDriver.Component(URL)
                    .withParameter("language", language)
                    .withTargetView("transcript")
                    .build().withTimeout(600)
        );
        composer.add(
            new DUUIRemoteDriver.Component(WHISPERX_URL)
                    .withParameter("language", language)
                    .withSourceView("transcript")
                    .withTargetView("anonymized_transcript")
                    .build().withTimeout(600)
        );
    }

    private void runAnonymizationTest(String language, String audioPath) throws Exception {
        addComponent(language);

        Path inputPath = Path.of(audioPath);
        byte[] audioBytes = Files.readAllBytes(inputPath);
        assertTrue(audioBytes.length > 44, "Input fixture must contain WAV audio data");

        String audioB64 = Base64.getEncoder().encodeToString(audioBytes);
        cas.setSofaDataString(audioB64, "audio/wav");

        composer.run(cas);

        System.out.println("WhisperX: " + cas.getView("anonymized_transcript").getDocumentText());

        JCas anonView = cas.getView("transcript");
        String anonB64 = anonView.getSofaDataString();
        assertNotNull(anonB64);
        assertFalse(anonB64.isBlank(), "Anonymized audio must not be empty");

        byte[] anonAudio = Base64.getDecoder().decode(anonB64);
        assertTrue(anonAudio.length > 44, "Anonymized WAV must contain audio data");
        assertEquals("RIFF", new String(anonAudio, 0, 4, StandardCharsets.US_ASCII));
        assertEquals("WAVE", new String(anonAudio, 8, 4, StandardCharsets.US_ASCII));

        Path outputPath = outputDirectory.resolve("anonymized_" + inputPath.getFileName());
        Files.write(outputPath, anonAudio);
    }

    @Test
    void testAudioDe() throws Exception {
        runAnonymizationTest("de", "src/test/resources/Katze.wav");
    }

    @Test
    void testAudioEn() throws Exception {
        runAnonymizationTest("en", "src/test/resources/zoo.wav");
    }
}
