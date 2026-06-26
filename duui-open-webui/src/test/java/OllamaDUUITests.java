import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.type.Image;
import org.texttechnologylab.annotation.type.Video;
import org.texttechnologylab.annotation.type.Audio;
import org.texttechnologylab.type.llm.prompt.Prompt;
import org.texttechnologylab.type.llm.prompt.Message;
import org.texttechnologylab.type.llm.prompt.Result;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.dkpro.core.io.xmi.XmiWriter;
import org.xml.sax.SAXException;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.util.*;
import java.util.Base64;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
import static org.junit.jupiter.api.Assertions.*;

public class OllamaDUUITests {

    static DUUIComposer composer;
    static JCas cas;
    static JCas inputView;

    // Update this URL to your Ollama/OpenWebUI wrapper endpoint
    static String url = "http://127.0.0.1:9977";
    static String sOutputPath = "src/test/results";

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException, SAXException, CompressorException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIUIMADriver uima_driver = new DUUIUIMADriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver, uima_driver);

        cas = JCasFactory.createJCas();
        inputView = cas.createView("UserRequestView");
    }

    @AfterAll
    static void afterAll() throws UnknownHostException {
        composer.shutdown();
    }

    @AfterEach
    public void afterEach() throws IOException, SAXException {
        composer.resetPipeline();
        cas.reset();
    }

    // Helper method to create CAS with prompts
    public void createCas(String language, List<String> prompts) throws UIMAException {
        cas.setDocumentLanguage(language);
        StringBuilder sb = new StringBuilder();

        for (String messageText : prompts) {
            Prompt prompt = new Prompt(cas);
            prompt.setArgs("{}");

            Message message = new Message(cas);
            message.setRole("user");
            message.setContent(messageText);
            message.addToIndexes();

            FSArray messages = new FSArray(cas, 1);
            messages.set(0, message);
            prompt.setMessages(messages);
            prompt.addToIndexes();

            sb.append(messageText).append(" ");
        }

        inputView.setDocumentText(sb.toString().trim());
//        cas.setDocumentText(sb.toString().trim());
    }

    // Helper method to create CAS with images
    private void    createCasWithImages(String language, List<String> prompts, List<String> imagePaths) throws UIMAException {
        createCas(language, prompts);

        for (String path : imagePaths) {
            Image img = new Image(cas);
            img.setSrc(convertFileToBase64(path));
            img.addToIndexes();
        }
    }

    // Helper method to create CAS with audio
    private void createCasWithAudio(String language, List<String> prompts, List<String> audioPaths) throws UIMAException {
        createCas(language, prompts);
        for (String path : audioPaths) {
            Audio audio = new Audio(cas);
            audio.setSrc(readFileAsBase64(path));
            audio.setMimetype("audio/wav");
            audio.addToIndexes();
        }
    }

    // Helper method to create CAS with video
    public void createCasWithVideo(String language, List<String> prompts, String videoBase64) throws UIMAException {
        cas.setDocumentLanguage(language);

        for (String messageText : prompts) {
            Prompt prompt = new Prompt(cas);
            prompt.setArgs("{}");

            Message message = new Message(cas);
            message.setRole("user");
            message.setContent(messageText);
            message.addToIndexes();

            FSArray messages = new FSArray(cas, 1);
            messages.set(0, message);
            prompt.setMessages(messages);
            prompt.addToIndexes();
        }

        Video videoWrapper = new Video(cas);
        videoWrapper.setMimetype("video/mp4");
        videoWrapper.setSrc(videoBase64);
        videoWrapper.addToIndexes();
    }

    // Helper method to convert file to Base64
    private static String convertFileToBase64(String filePath) {
        try {
            File file = new File(filePath);
            FileInputStream fis = new FileInputStream(file);
            byte[] bytes = fis.readAllBytes();
            fis.close();
            return Base64.getEncoder().encodeToString(bytes);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    // Helper method to read file as Base64
    private String readFileAsBase64(String filePath) {
        try {
            byte[] fileBytes = java.nio.file.Files.readAllBytes(new File(filePath).toPath());
            return Base64.getEncoder().encodeToString(fileBytes);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    // Helper method to save Base64 to image
    private static void saveBase64ToImage(String base64String, String outputPath) {
        try {
            byte[] decodedBytes = Base64.getDecoder().decode(base64String);
            java.nio.file.Files.write(new File(outputPath).toPath(), decodedBytes);
            System.out.println("Image saved as: " + outputPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Helper method to save Base64 to video
    private static void saveBase64ToVideo(String base64String, String outputPath) {
        try {
            byte[] decodedBytes = Base64.getDecoder().decode(base64String);
            java.nio.file.Files.write(new File(outputPath).toPath(), decodedBytes);
            System.out.println("Video saved as: " + outputPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Helper method to verify no images
    private void verifyNoImages() {
        Collection<Image> allImages = JCasUtil.select(cas, Image.class);
        assertTrue(allImages.isEmpty(), "No images should be generated for text-only mode.");
    }

    // Test for text-only mode
    @Test
    public void testTextOnly() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", "llama3.2:latest")
                        .withParameter("mode", "text")
                        .withParameter("language", "en")
                        .withParameter("ollama_host", "https://llm.texttechnologylab.org")
//                        .withParameter("ollama_port", "8080")
                        .withParameter("ollama_auth_token", "")
                        .withParameter("system_prompt", "")
                        .build().withTimeout(1000)
        );

        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, sOutputPath,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());

        List<String> prompts = Arrays.asList(
                "Who is the current president of the USA?",
                "Is Frankfurt the capital of EU finance?"
        );

        createCas("en", prompts);
        composer.run(cas);

        verifyNoImages();

        // Print results
        for (Result result : JCasUtil.select(cas, Result.class)) {
            System.out.println(result.getMeta());
        }
    }

    // Test for image-only mode
    @Test
    public void testImageOnly() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", "anduin.qwen2.5vl:3b")
                        .withParameter("mode", "image")
                        .withParameter("language", "en")
                        .withParameter("ollama_host", "localhost") // https:/llm.example
//                        .withParameter("ollama_port", "8080")
                        .withParameter("ollama_auth_token", "")
                        .withParameter("system_prompt", "")
                        .build().withTimeout(1000)
        );

        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, sOutputPath,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());

        List<String> prompts = Collections.singletonList("What is shown in this image?");
        List<String> imagePaths = Arrays.asList(
//                "src/test/resources/images/fridge.jpg",
                "src/test/resources/images/cars.jpg"
        );

        createCasWithImages("en", prompts, imagePaths);
        composer.run(cas);

        int idx = 0;
        for (Image img : JCasUtil.select(cas, Image.class)) {
            saveBase64ToImage(img.getSrc(), "src/test/results/images/output_image_" + idx++ + ".png");
        }

        // Print results
        for (Result result : JCasUtil.select(cas, Result.class)) {
            System.out.println(result.getMeta());
        }
    }

   }
