package imageToText;

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

import static org.junit.Assert.assertEquals;

public class ImagetoTextTests {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:8000";
    static String model = "OFA-Sys/small-stable-diffusion-v0";
//    static String model = "pol_emo_mDeBERTa";

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException, SAXException, CompressorException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIUIMADriver uima_driver = new DUUIUIMADriver()
                .withDebug(false);
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

    public void createCas(String language, List<String> images) throws UIMAException {
        cas.setDocumentLanguage(language);

        StringBuilder sb = new StringBuilder();
        for (String base64image : images) {
            Image image_ = new Image(cas, sb.length(), sb.length()+base64image.length());
            image_.setSrc(base64image);
            image_.addToIndexes();
            sb.append(base64image).append(" ");
        }
        System.out.println(cas.getDocumentText());

//        cas.setDocumentText(sb.toString());
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
        String model = "microsoft/kosmos-2-patch14-224";
        String sOutputPath = "/home/staff_homes/aabusale/duui-uima/duui_image_to_text/tests/";
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("prompt", "<grounding>An image of")
        );


        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, sOutputPath,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());



        // convert it to a list of base64 strings
        List<String> base64_images = new ArrayList<String>();
        base64_images.add(convertImageToBase64("/home/staff_homes/aabusale/duui-uima/duui_image_to_text/tests/german_election_25.jpg"));

//        // add another dummy text
//        base64_images.add("Hello this is a dummy text for testing");

        createCas("en", base64_images);

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
        // Reading the image files and converting them to base64
//        expected_images.add(convertImageToBase64("/home/staff_homes/aabusale/duui-uima/duui-text-to-image/src/main/python/original_0.png"));
//        expected_images.add(convertImageToBase64("/home/staff_homes/aabusale/duui-uima/duui-text-to-image/src/main/python/original_1.png"));
        // add the base64 string of the image to the expected_emotions list

        // compare the expected_images list with the all_images_base64 list
        // Compare the expected_images list with the all_images_base64 list
//        assert all_images_base64.equals(expected_images);
        if (all_images_base64.equals(expected_images)) {
            System.out.println("Test Passed: All images match.");
        } else {
            System.out.println("Test Failed: Images do not match.");
        }
    }

}
