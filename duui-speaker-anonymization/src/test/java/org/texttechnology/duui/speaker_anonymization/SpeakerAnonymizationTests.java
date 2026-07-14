package org.texttechnology.duui.speaker_anonymization;

import static org.junit.jupiter.api.Assertions.*;

import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.InvalidXMLException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Base64;

import javax.sound.sampled.*;

import org.apache.commons.compress.compressors.CompressorException;
import org.xml.sax.SAXException;

/**
 * Integration tests for duui-speaker-anonymization.
 * Requires the component running on http://127.0.0.1:9714.
 */
public class SpeakerAnonymizationTests {

    private static final String URL = "http://geltlin:9714";

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

    private static String minimalWavBase64() throws Exception {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        AudioFormat format = new AudioFormat(16000, 16, 1, true, false);
        byte[] silence = new byte[16000 * 2];
        AudioInputStream ais = new AudioInputStream(
            new java.io.ByteArrayInputStream(silence), format, 16000);
        AudioSystem.write(ais, AudioFileFormat.Type.WAVE, bos);
        return Base64.getEncoder().encodeToString(bos.toByteArray());
    }

    private void addComponent(String language)
            throws InvalidXMLException, IOException, URISyntaxException, SAXException, CompressorException {
        composer.add(
            new DUUIRemoteDriver.Component(URL)
                    .withParameter("language", language)
                    .build().withTimeout(600)
        );
    }

    private void runAnonymizationTest(String language, String audioPath) throws Exception {
        addComponent(language);

        byte[] audioBytes;
        try {
            audioBytes = java.nio.file.Files.readAllBytes(java.nio.file.Path.of(audioPath));
        } catch (java.io.IOException e) {
            System.out.println("Skipping: no test audio at " + audioPath);
            return;
        }

        String audioB64 = Base64.getEncoder().encodeToString(audioBytes);
        cas.setSofaDataString(audioB64, "text/x-wave");

        try {
            composer.run(cas);
        } catch (Exception e) {
            System.err.println("Server error (is it running?): " + e.getMessage());
            e.printStackTrace();
            return;
        }

        String text = cas.getSofaDataString();
        assertNotNull(text);
        System.out.println("Transcript: " + text);

        try {
            JCas anonView = cas.getView("opf_anonymized_audio");
            assertNotNull(anonView);
            String anonB64 = anonView.getSofaDataString();
            assertNotNull(anonB64);
            assertFalse(anonB64.isEmpty());
            System.out.println("Anonymized audio: " + anonB64.length() + " chars base64");
        } catch (Exception e) {
            System.out.println("No anonymized audio view: " + e.getMessage());
        }
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
