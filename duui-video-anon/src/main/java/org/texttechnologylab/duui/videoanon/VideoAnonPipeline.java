package org.texttechnologylab.duui.videoanon;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;

import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.type.Video;

/** Runs one remote DUUI stage; Python handles the local media stages. */
public final class VideoAnonPipeline {
    private VideoAnonPipeline() { }

    private static String setting(String name, String defaultValue) {
        String value = System.getenv(name);
        return value == null || value.isBlank() ? defaultValue : value;
    }

    private static String requiredUrl(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must be set");
        }
        return value;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 3 || !List.of("face", "speaker").contains(args[0])) {
            throw new IllegalArgumentException(
                    "Usage: VideoAnonPipeline face|speaker input.mp4|input.wav output.mp4|output.wav");
        }
        Path input = Path.of(args[1]);
        Path output = Path.of(args[2]);
        if (args[0].equals("face")) {
            runFace(input, output);
        } else {
            runSpeaker(input, output);
        }
    }

    private static void runFace(Path input, Path output) throws Exception {
        String mode = setting("DUUI_FACE_MODE", "redact");
        if (!List.of("redact", "single_align", "multiple_align").contains(mode)) {
            throw new IllegalArgumentException("DUUI_FACE_MODE must be redact, single_align, or multiple_align");
        }
        String token = setting("HF_TOKEN", "");
        if (!mode.equals("redact") && token.isBlank()) {
            throw new IllegalArgumentException("HF_TOKEN is required for generative face anonymization");
        }

        String faceUrl = requiredUrl("DUUI_FACE_ANON_URL");
        DUUIComposer composer = newComposer();
        composer.add(new DUUIRemoteDriver.Component(faceUrl)
                .withName("face-anonymization")
                .withParameter("anon_type", mode)
                .withParameter("redact_type", setting("DUUI_REDACT_TYPE", "black"))
                .withParameter("hf_token", token)
                .withParameter("sampling_mode", "uniform")
                .withParameter("frame_interval", "1")
                .withTargetView("face_anonymized")
                .build().withTimeout(3600));
        try {
            JCas cas = JCasFactory.createJCas();
            cas.setDocumentText("video");
            cas.setDocumentLanguage(setting("DUUI_LANGUAGE", "en"));
            Video video = new Video(cas, 0, 5);
            video.setSrc(Base64.getEncoder().encodeToString(Files.readAllBytes(input)));
            video.setMimetype(input.getFileName().toString().toLowerCase().endsWith(".webm")
                    ? "video/webm" : "video/mp4");
            video.addToIndexes();

            composer.run(cas);
            List<Video> videos = List.copyOf(JCasUtil.select(cas.getView("face_anonymized"), Video.class));
            if (videos.size() != 1 || videos.get(0).getSrc() == null
                    || videos.get(0).getSrc().isBlank()) {
                throw new IllegalStateException("Face stage produced no video");
            }
            Files.write(output, Base64.getDecoder().decode(videos.get(0).getSrc()));
        } finally {
            composer.shutdown();
        }
    }

    private static void runSpeaker(Path input, Path output) throws Exception {
        String speakerUrl = requiredUrl("DUUI_SPEAKER_ANON_URL");
        DUUIComposer composer = newComposer();
        composer.add(new DUUIRemoteDriver.Component(speakerUrl)
                .withName("speaker-anonymization")
                .withParameter("language", setting("DUUI_LANGUAGE", "en"))
                .withTargetView("anonymized_audio")
                .build().withTimeout(3600));
        try {
            JCas cas = JCasFactory.createJCas();
            cas.setDocumentLanguage(setting("DUUI_LANGUAGE", "en"));
            cas.setSofaDataString(Base64.getEncoder().encodeToString(Files.readAllBytes(input)), "audio/wav");
            composer.run(cas);

            String audio = null;
            for (String viewName : List.of("anonymized_audio", "opf_anonymized_audio")) {
                try {
                    audio = cas.getView(viewName).getSofaDataString();
                } catch (Exception ignored) {
                    // Older speaker images write to a fixed output view.
                }
                if (audio != null) {
                    break;
                }
            }
            if (audio == null) {
                throw new IllegalStateException("Speaker stage produced no audio view");
            }
            Files.write(output, Base64.getDecoder().decode(audio));
        } finally {
            composer.shutdown();
        }
    }

    private static DUUIComposer newComposer() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIRemoteDriver());
        return composer;
    }

}
