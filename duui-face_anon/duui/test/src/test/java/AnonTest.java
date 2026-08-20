import org.apache.uima.UIMAException;
import org.apache.uima.cas.SerialFormat;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.CasIOUtils;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.apache.uima.fit.factory.JCasFactory;
import org.texttechnologylab.annotation.type.Image;
import org.texttechnologylab.annotation.type.Video;
import org.xml.sax.SAXException;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.UnknownHostException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Base64;
import java.util.List;

public class AnonTest {

    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:9714";
    static String inputPath = "resources/input";
    static String videoInputPath = "resources/input/videos/kids_video.mp4";
    static String sOutputPath = "target/test-output";
    static String hf_token;

    @BeforeAll
    static void beforeAll() throws Exception {
        hf_token = System.getenv("HF_TOKEN");
        if (hf_token == null) {
            throw new IllegalStateException("HF_TOKEN environment variable is not set");
        }
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIUIMADriver uima_driver = new DUUIUIMADriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver, uima_driver);
        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        composer.addDriver(docker_driver);

        cas = JCasFactory.createJCas();
    }

    @AfterAll
    static void afterAll() throws UnknownHostException {
        composer.shutdown();
    }

    @AfterEach
    public void afterEach() throws Exception {
        composer.resetPipeline();


        cas.reset();

    }


    // Helper method to convert image file to Base64 string
    private static List<String> convertImageToBase64(String imagePath) {
        try {
            File file = new File(imagePath);
            BufferedImage bufferedImage = ImageIO.read(file);
            int height = bufferedImage.getHeight();
            int width = bufferedImage.getWidth();

            // Convert the image to byte array
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            ImageIO.write(bufferedImage, "png", byteArrayOutputStream);
            byte[] imageBytes = byteArrayOutputStream.toByteArray();

            // Encode the byte array to Base64

            return List.of(Base64.getEncoder().encodeToString(imageBytes), Integer.toString(height), Integer.toString(width));
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
    // Helper method to save Base64 string back to an image file
    private static void saveBase64ToImage(String base64String, String name) throws IOException {
        LocalDateTime currentDateTime = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH-mm-ss");

        // Decode the Base64 string into a byte array
        byte[] decodedBytes = Base64.getDecoder().decode(base64String);

        // Create an image from the byte array
        BufferedImage image;
        try (InputStream inputStream = new ByteArrayInputStream(decodedBytes)) {
            image = ImageIO.read(inputStream);
            if (image == null) {
                throw new IOException("Output is not a supported image");
            }
        }

        // Save generated test artifacts outside the input resources directory.
        File outputFile = new File(sOutputPath + "/" + currentDateTime.format(formatter) + "_" + name + ".png");
        File outputDirectory = outputFile.getParentFile();
        if (!outputDirectory.exists() && !outputDirectory.mkdirs()) {
            throw new IOException("Could not create output directory: " + outputDirectory);
        }
        if (!ImageIO.write(image, "png", outputFile)) {
            throw new IOException("No PNG writer is available");
        }

        System.out.println("Image saved as: " + outputFile.getAbsolutePath());
    }


    public void createCas() throws UIMAException{
        cas.setDocumentLanguage("eng");
        cas.setDocumentText("Place holder text");
        File inputFolder = new File(inputPath);
        System.out.println("Reading in images from: " + inputFolder);
        for (File file: inputFolder.listFiles()){
            if (! file.isDirectory()){
                Image image = new Image(cas);
                List<String> imageInfo = convertImageToBase64(file.getPath());
                System.out.println(imageInfo.get(1));
                System.out.println(imageInfo.get(2));
                image.setSrc(imageInfo.get(0));
                image.setMimetype("img/");
                image.setHeight(Integer.parseInt(imageInfo.get(1)));
                image.setWidth(Integer.parseInt(imageInfo.get(2)));

                image.addToIndexes();
            }
        }
        cas.createView("output");

    }

    public void readImagesInCas(String name) throws Exception {
        int i = 0;
        JCas view = cas.getView("output");
        for (Image image : JCasUtil.select(view, Image.class)){
            saveBase64ToImage(image.getSrc(), name + String.valueOf(i));
            String result = "Image %d is has specified: \n Height: %d; Width: %d".formatted(i, image.getHeight(), image.getWidth());
            System.out.println(result);
        }

    }

    private void createVideoCas() throws IOException, UIMAException {
        cas.setDocumentLanguage("eng");
        cas.setDocumentText("video");

        Video video = new Video(cas, 1, 4);
        video.setSrc(Base64.getEncoder().encodeToString(
                Files.readAllBytes(Path.of(videoInputPath))
        ));
        video.setFps(30.0);
        video.setLength(1.0);
        video.addToIndexes();

        cas.createView("output");
    }

    private static void saveBase64ToVideo(String base64String, String name) throws IOException {
        Path outputDirectory = Path.of(sOutputPath);
        Files.createDirectories(outputDirectory);
        Files.write(outputDirectory.resolve(name + ".mp4"), Base64.getDecoder().decode(base64String));
    }

//    /*
//    to visualize the cas to check if it was initialized correctly
//     */
//    public static void simpleTempExporter(JCas cas, String filename) throws IOException{
//        File outputFile = new File(sOutputPath + "/"+ filename + ".xmi");
//        outputFile.getParentFile().mkdirs();
//        CasIOUtils.save(cas.getCas(), new FileOutputStream(outputFile), SerialFormat.XMI_PRETTY);
//    }

    @Test
    public void testSingleFaceSimple() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("anon_type", "single_align")
                        .withParameter("vis_input", "true")
                        .withParameter("hf_token", hf_token)
                        .withTargetView("output")
                        .build().withTimeout(1000)

        );

        createCas();
        composer.run(cas);
        readImagesInCas("single face");

    }



    @Test
    public void testMultipleAlignSimple() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("anon_type", "multiple_align")
                        .withParameter("vis_input", "true")
                        .withParameter("hf_token", hf_token)

                        // to write to
                        .withTargetView("output")
                        .build().withTimeout(1000)
        );

        createCas();
        composer.run(cas);
        readImagesInCas("multiple");

    }

    @Test
    public void testSwapSimple() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("anon_type", "swap")
                        .withParameter("vis_input", "true")
                        .withParameter("hf_token", hf_token)
                        .withTargetView("output")
                        .build().withTimeout(1000)
        );

        createCas(); // needs exactly 2 images loaded in the CAS
        composer.run(cas);
        readImagesInCas("simple");

    }

    @Test
    public void testRedactBlur() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("anon_type", "redact")
                        .withParameter("redact_type", "blur")
                        .withParameter("blur", "51")
                        .withParameter("hf_token", hf_token)
                        .withTargetView("output")
                        .build().withTimeout(1000)
        );

        createCas();
        composer.run(cas);
        readImagesInCas("bur");

    }

    @Test
    public void testRedactPixel() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("anon_type", "redact")
                        .withParameter("redact_type", "pixel")
                        .withParameter("pixel", "10")
                        .withParameter("hf_token", hf_token)
                        .withTargetView("output")
                        .build().withTimeout(1000)
        );

        createCas();
        composer.run(cas);
        readImagesInCas("pixel");

    }
    @Test
    public void testBlackOut()throws Exception{
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("anon_type", "redact")
                        .withParameter("redact_type", "black")
                        .withParameter("pixel", "10")
                        .withParameter("hf_token", hf_token)
                        .withTargetView("output")
                        .build().withTimeout(1000)
        );

        createCas();
        composer.run(cas);
        readImagesInCas("blackout");

    }

    @Test
    public void testVideoRedactBlur() throws Exception {
        int frameInterval = 5;
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("anon_type", "redact")
                        .withParameter("redact_type", "blur")
                        .withParameter("blur", "51")
                        .withParameter("frame_interval", Integer.toString(frameInterval))
                        .withTargetView("output")
                        .build().withTimeout(1000)
        );

        createVideoCas();
        composer.run(cas);

        List<Video> outputVideos = List.copyOf(
                JCasUtil.select(cas.getView("output"), Video.class)
        );
        Assertions.assertEquals(1, outputVideos.size());

        Video output = outputVideos.get(0);
        Assertions.assertFalse(output.getSrc().isBlank());
        Assertions.assertTrue(output.getFps() > 0.0);
        Assertions.assertTrue(output.getFps() < 30.0);
        Assertions.assertTrue(output.getLength() > 0.0);
        Assertions.assertEquals(1, output.getBegin());
        Assertions.assertEquals(4, output.getEnd());
        saveBase64ToVideo(output.getSrc(), "video-redact-blur");
    }
//    @Test
//    public void testMissingHfTokenFails() throws Exception {
//        composer.add(
//                new DUUIRemoteDriver.Component(url)
//                        .withParameter("anon_type", "single_align")
//                        .build().withTimeout(1000)
//        );
//
//        createCas();
//        composer.run(cas);
//        // todo does it contain the errors? and how do i see them
//        readImagesInCas("noHF");
//
//    }


    @Test
    public void testNoImages()throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("anon_type", "single_align")
                        .withParameter("hf_token", hf_token)
                        .withTargetView("output")
                        .build().withTimeout(1000)
        );

        cas = JCasFactory.createJCas();
        cas.setDocumentText("placeholder");

        composer.run(cas);
        readImagesInCas("noImg");

    }


    @Test
    public void testSwap() throws Exception{
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("anon_type", "swap")
                        .withParameter("hf_token", hf_token)
                        .withTargetView("output")
                        .build().withTimeout(1000)
        );

        createCas();
        composer.run(cas);
        readImagesInCas("testSwap");

    }
}
