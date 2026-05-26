package org.texttechnologylab.duui.whisperx;

import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.type.DiarizedAudioToken;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.nio.file.Files;

public class WhisperXTest {
    private static final String HF_TOKEN = "";

    @Test
    public void testWhisperXDe() throws Exception {
        String language = "de";
        Path audioPath = Paths.get("/home/staff_homes/dbaumart/dev/git/experiments-transcription/temp/Interpretation_T1_short_audio.wav");

        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentLanguage(language);

        byte[] audioBytes = Files.readAllBytes(audioPath);
        String audioBase64 = Base64.getEncoder().encodeToString(audioBytes);

        jCas.setSofaDataString(audioBase64, "audio/wav");

        DUUIComposer composer = new DUUIComposer()
                .withWorkers(1)
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);

        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:8000")
                        .withScale(1)
                        .withTargetView("transcript")
                        .withParameter("language", language)
                        .withParameter("model", "large-v3")
                        .withParameter("hf_token", HF_TOKEN)
                        .build()
                        .withTimeout(1000000000L)
        );

        composer.run(jCas);

        for (DiarizedAudioToken audioToken : JCasUtil.select(jCas.getView("transcript"), DiarizedAudioToken.class)) {
            System.out.println(audioToken.getSpeakerId() + ": " + audioToken.getTimeStart() + " " + audioToken.getTimeEnd() + " " + audioToken.getCoveredText());
        }
    }
}
