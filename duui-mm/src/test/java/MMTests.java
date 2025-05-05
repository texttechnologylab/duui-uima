import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.*;
import org.dkpro.core.io.xmi.XmiWriter;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.*;

import java.util.*;
import java.util.Base64;
import java.io.*;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

import org.texttechnologylab.annotation.type.Image;
import org.texttechnologylab.type.llm.prompt.Prompt;
import org.texttechnologylab.type.llm.prompt.Message;

import static org.junit.Assert.assertEquals;

public class MMTests {

    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:8000";
    static String model = "microsoft/Phi-4-multimodal-instruct";
    static String sOutputPath = "src/test/results";

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException, SAXException, CompressorException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIUIMADriver uima_driver = new DUUIUIMADriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver, uima_driver);
//        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
//        composer.addDriver(docker_driver);

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

    public void createCas(String language, List<String> prompts) throws UIMAException {
        cas.setDocumentLanguage(language);
        StringBuilder sb = new StringBuilder();

        for (String messageText : prompts) {
            Prompt prompt = new Prompt(cas);

            // Set required `args` field as an empty JSON string
            prompt.setArgs("{}");

            // Create a Message object
            Message message = new Message(cas);
            message.setRole("user"); // Or whatever default role
            message.setContent(messageText);
            message.addToIndexes();

            // Link message to prompt
            prompt.setMessages(new FSArray(cas, 1));
            prompt.setMessages(0, message);
            prompt.addToIndexes();

            sb.append(messageText).append(" ");
        }

        cas.setDocumentText(sb.toString().trim());
    }


    private void createCasWithImages(String language, List<String> prompts, List<String> imagePaths) throws UIMAException {
        createCas(language, prompts);

        for (String path : imagePaths) {
            Image img = new Image(cas);
            img.setSrc(convertImageToBase64(path));
            img.addToIndexes();
        }
    }

//    // Placeholder for audio
//    private void createCasWithAudio(String language, List<String> prompts, List<String> audioPaths) throws UIMAException {
//        createCas(language, prompts);
//        for (String path : audioPaths) {
//            org.texttechnologylab.annotation.type.Audio audio = new org.texttechnologylab.annotation.type.Audio(cas);
//            audio.setSrc(readFileAsBase64(path));
//            audio.addToIndexes();
//        }
//    }

    public void createCasWithVideo(String language, List<String> prompts, String videoBase64) throws UIMAException {
        cas.setDocumentLanguage(language);

        // Add prompt
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

        // Add video as base64 image with dummy MIME
        Image videoWrapper = new Image(cas);
        videoWrapper.setMimetype("video/mp4");
        videoWrapper.setSrc(videoBase64);
        videoWrapper.addToIndexes();
    }

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



    // Load audio/video or binary as Base64
    private String readFileAsBase64(String filePath) {
        try {
            byte[] fileBytes = java.nio.file.Files.readAllBytes(new File(filePath).toPath());
            return Base64.getEncoder().encodeToString(fileBytes);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }


    // Helper method to save Base64 string back to an image file
    private static void saveBase64ToImage(String base64String, String outputPath) {
        try {
            // Decode the Base64 string into a byte array
            byte[] decodedBytes = Base64.getDecoder().decode(base64String);

            // Create an image from the byte array
            InputStream inputStream = new ByteArrayInputStream(decodedBytes);
            BufferedImage image = ImageIO.read(inputStream);

            // Save the image to the specified output file
            File outputFile = new File(outputPath);
            ImageIO.write(image, "png", outputFile);

            System.out.println("Image saved as: " + outputPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    // Helper method to convert image file to Base64 string
    private static String convertImageToBase64(String imagePath) {
        try {
            File file = new File(imagePath);
            BufferedImage bufferedImage = ImageIO.read(file);

            // Convert the image to byte array
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            ImageIO.write(bufferedImage, "png", byteArrayOutputStream);
            byte[] imageBytes = byteArrayOutputStream.toByteArray();

            // Encode the byte array to Base64
            return Base64.getEncoder().encodeToString(imageBytes);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private void verifyNoImages() {
        Collection<Image> allImages = JCasUtil.select(cas, Image.class);
        if (allImages.isEmpty()) {
            System.out.println("Test Passed: No images generated.");
        } else {
            System.out.println("Test Failed: Unexpected image outputs.");
        }
    }



    @Test
        public void testTextOnly() throws Exception {
            composer.add(
                    new DUUIRemoteDriver.Component(url)
                            .withParameter("model_name", model)
                            .withParameter("mode", "text")
                            .build().withTimeout(1000)

            );

            List<String> prompts = Arrays.asList(
                    "Who is the current president of the USA?",
                    "Is Frankfurt the capital of EU finance?"
            );

            createCas("en", prompts);
            composer.run(cas);

            verifyNoImages(); // Text-only should produce no image outputs
        }

    @Test
    public void testImageOnly() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("mode", "image")
                        .build().withTimeout(1000)
        );

        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, sOutputPath,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());

        List<String> prompts = Arrays.asList(
                "What is shown in this image?"
        );
        List<String> imagePaths = Arrays.asList(
                "src/test/resources/images/fridge.jpg",
                "src/test/resources/images/cars.jpg"
        );

        createCasWithImages("en", prompts, imagePaths);
        composer.run(cas);

        int idx = 0;
        for (Image img : JCasUtil.select(cas, Image.class)) {
            saveBase64ToImage(img.getSrc(), "src/test/results/images/output_image_" + idx++ + ".png");
        }
    }


    @Test
    public void testImageOnlyTwoPrompts() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("mode", "image")
                        .build().withTimeout(1000)
        );

        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, sOutputPath,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());

        List<String> prompts = Arrays.asList(
                "What is shown in this image?",
                "how many cars are there?"
        );
        List<String> imagePaths = Arrays.asList(
                "src/test/resources/images/fridge.jpg",
                "src/test/resources/images/cars.jpg"
        );

        createCasWithImages("en", prompts, imagePaths);
        composer.run(cas);

        int idx = 0;
        for (Image img : JCasUtil.select(cas, Image.class)) {
            saveBase64ToImage(img.getSrc(), "src/test/results/images/output_image_" + idx++ + ".png");
        }
    }

    @Test
    public void testFramesOnly() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("mode", "frames")
                        .build().withTimeout(1000)
        );

        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, sOutputPath,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());

        List<String> prompts = Collections.singletonList("Who drunk the water from the cup?");

        List<String> framePaths = Arrays.asList(
                "src/test/resources/frames/1.png",
                "src/test/resources/frames/2.png",
                "src/test/resources/frames/3.png",
                "src/test/resources/frames/4.png"

        );

        createCasWithImages("en", prompts, framePaths);
        composer.run(cas);

        int idx = 0;
        for (Image img : JCasUtil.select(cas, Image.class)) {
            saveBase64ToImage(img.getSrc(), "src/test/results/frames/output_frame_" + idx++ + ".png");
        }
    }

//    @Test
//    public void testVideo() throws Exception {
//
//        composer.add(
//                new DUUIRemoteDriver.Component(url)
//                        .withParameter("model_name", model)
//                        .withParameter("mode", "video")
//                        .build().withTimeout(1000)
//        );
//
//        // Optional: Add UIMA XmiWriter for output storage
//        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
//                XmiWriter.PARAM_TARGET_LOCATION, "src/test/results/video/",
//                XmiWriter.PARAM_PRETTY_PRINT, true,
//                XmiWriter.PARAM_OVERWRITE, true,
//                XmiWriter.PARAM_VERSION, "1.1"
//        )).build());
//
//        // Load video and convert to base64
//        String videoPath = "src/test/resources/videos/sample_video.mp4";
//        String videoBase64 = convertFileToBase64(videoPath);
//
//        List<String> prompts = Collections.singletonList("Describe what happens in the video.");
//
//        // Create CAS with 1 video and 1 prompt
//        createCasWithVideo("en", prompts, videoBase64);
//
//        // Run pipeline
//        composer.run(cas);
//
//        // Optionally verify the outputs
//        int idx = 0;
//        for (Image img : JCasUtil.select(cas, Image.class)) {
//            saveBase64ToImage(img.getSrc(), "src/test/results/video/frame_output_" + idx++ + ".png");
//        }
//    }
}

