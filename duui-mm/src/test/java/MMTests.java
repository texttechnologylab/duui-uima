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

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException, SAXException, CompressorException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);
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

        @Test
        public void SimpleTest() throws Exception {

            composer.add(
                    new DUUIRemoteDriver.Component(url)
                            .withParameter("model_name", model)
                            .withParameter("selection", "org.texttechnologylab.annotation.type.Image")
            );
            List<String> prompts = Arrays.asList(
                    "who's the current president of the USA?",
                    "Is frankfurt the capital of EU finance?"
            );

            createCas("en", prompts);

            composer.run(cas);

            Collection<Image> all_images = JCasUtil.select(cas, Image.class);

            Collection<String> all_images_base64 = new ArrayList<String>();

            for (Image image: all_images){
//            System.out.println(image.getCoveredText());
                all_images_base64.add(image.getSrc());
            }

//         Convert all Base64 strings back to images and save them as output_[idx].png
            int idx = 0;
            for (String base64 : all_images_base64) {
                saveBase64ToImage(base64, "output_" + idx + ".png");
                idx++;
            }

            // expected values
            ArrayList<String> expected_images = new ArrayList<String>();


//        assert all_images_base64.equals(expected_images);
            if (all_images_base64.equals(expected_images)) {
                System.out.println("Test Passed: All images match.");
            } else {
                System.out.println("Test Failed: Images do not match.");
            }
        }
    }

