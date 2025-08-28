package org.hucompute.textimager.uima.duui.crisper_whisper;

import net.bramp.ffmpeg.FFmpeg;
import net.bramp.ffmpeg.FFmpegExecutor;
import net.bramp.ffmpeg.builder.FFmpegBuilder;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.type.AudioToken;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;

public class TestCrisperWhisper {
    @Test
    public void testTranscribeSingle() throws Exception {
        JCas jCas = JCasFactory.createJCas();

        Path audioPath = Paths.get("/storage/projects/chatgpt-creativity/data/experiments2/Z209_2024-12-16_14-00/Audio Record/audioTim93006026249.m4a");
//        Path audioPath = Paths.get("output.wav");

        Path tmpAudioPath = Files.createTempFile("duui_mm_audio", ".wav");
        System.out.println("Converting audio to wav: " + tmpAudioPath);

        FFmpegBuilder builder = new FFmpegBuilder()
                .setInput(audioPath.toString())
                .addOutput(tmpAudioPath.toString())
                .setFormat("wav")
                .setAudioChannels(1)
                .setAudioSampleRate(16000)
                .done();

        FFmpeg fFmpeg = new FFmpeg("/usr/bin/ffmpeg");
        FFmpegExecutor executor = new FFmpegExecutor(fFmpeg);
        executor.createJob(builder).run();


        byte[] audioBytes = Files.readAllBytes(tmpAudioPath);
        String audioBase64 = Base64.getEncoder().encodeToString(audioBytes);

        jCas.setSofaDataString(audioBase64, "audio/wav");

        DUUIComposer composer = new DUUIComposer()
                .withWorkers(1)
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();
        composer.addDriver(dockerDriver);
        DUUIUIMADriver uimaDriver = new DUUIUIMADriver();
        composer.addDriver(uimaDriver);
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);

        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:8000")
                        .withScale(1)
                        .withTargetView("transcript")
                        .withParameter("language", "de")
                        .build()
                        .withTimeout(1000000000L)
        );

        composer.run(jCas, "whisper");

        System.out.println(jCas.getView("transcript").getDocumentText());
        for (AudioToken token : JCasUtil.select(jCas.getView("transcript"), AudioToken.class)) {
            System.out.println(token.getTimeStart() + " - " + token.getTimeEnd() + " | " + token.getBegin() + " - " + token.getEnd() + ": " + token.getCoveredText());
        }
    }
}
