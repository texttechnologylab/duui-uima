package org.hucompute.textimager.uima.transformers.emotion;

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

import static org.junit.Assert.assertEquals;

public class TextToImageTest {
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

    public void createCas(String language, List<String> sentences) throws UIMAException {
        cas.setDocumentLanguage(language);

        StringBuilder sb = new StringBuilder();
        for (String sentence : sentences) {
            Sentence sentenceAnnotation = new Sentence(cas, sb.length(), sb.length()+sentence.length());
            sentenceAnnotation.addToIndexes();
            sb.append(sentence).append(" ");
        }

        cas.setDocumentText(sb.toString());
    }

    // Helper method to convert image file to Base64 string
    private static String convertImageToBase64(String imagePath) {
        try {
            File file = new File(imagePath);
            BufferedImage bufferedImage = ImageIO.read(file);

            // Convert the image to byte array
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            ImageIO.write(bufferedImage, "PNG", byteArrayOutputStream);
            byte[] imageBytes = byteArrayOutputStream.toByteArray();

            // Encode the byte array to Base64
            return Base64.getEncoder().encodeToString(imageBytes);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    @Test
    public void EnglishTest() throws Exception {
//        composer.add(new DUUIDockerDriver.
//                Component("docker.texttechnologylab.org/textimager-duui-transformers-topic:0.0.1")
//                .withParameter("model_name", model)
//                .withParameter("selection", "text,de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
//                .withScale(1)
//                .withImageFetching());
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );
        List<String> sentences = Arrays.asList(
                "toryboard sketch of a zombie basketball player dunking with both hands, action shot, motion blur, hero.",
                "frankfurt as a future city. I love this place."
        );

        createCas("en", sentences);

        composer.run(cas);

        Collection<Image> all_images = JCasUtil.select(cas, Image.class);

        Collection<String> all_images_base64 = new ArrayList<String>();

        for (Image image: all_images){
//            System.out.println(image.getCoveredText());
            all_images_base64.add(image.getSrc());
        }

        // expected values
        ArrayList<String> expected_images = new ArrayList<String>();
        // Reading the image files and converting them to base64
        expected_images.add(convertImageToBase64("/home/staff_homes/aabusale/duui-uima/duui-text-to-image/src/main/python/first.png"));
        expected_images.add(convertImageToBase64("/home/staff_homes/aabusale/duui-uima/duui-text-to-image/src/main/python/sec.png"));
        // add the base64 string of the image to the expected_emotions list

        // compare the expected_images list with the all_images_base64 list
        // Compare the expected_images list with the all_images_base64 list
        assert all_images_base64.equals(expected_images);
//        if (all_images_base64.equals(expected_images)) {
//            System.out.println("Test Passed: All images match.");
//        } else {
//            System.out.println("Test Failed: Images do not match.");
//        }
    }

//    @Test
//    public void GermanTest() throws Exception {
////        composer.add(new DUUIDockerDriver.
////                Component("docker.texttechnologylab.org/textimager-duui-transformers-topic:0.0.1")
////                .withParameter("model_name", model)
////                .withParameter("selection", "text,de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
////                .withScale(1)
////                .withImageFetching());
//        composer.add(
//                new DUUIRemoteDriver.Component(url)
//                        .withParameter("model_name", model)
//                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
//        );
//        List<String> sentences = Arrays.asList(
//                "I hasse dich. ich bin richtig wütend.",
//                "Ich bin sehr glücklich hier zu sein. Ich liebe diesen Ort."
//        );
//
//        createCas("de", sentences);
//
//        composer.run(cas);
//
//        Collection<Emotion> all_emotions = JCasUtil.select(cas, Emotion.class);
////        System.out.println(topics.size());
//        ArrayList<Map<String, Float>> expected = new ArrayList<Map<String, Float>>();
//        for (Emotion emotion: all_emotions){
//            System.out.println(emotion.getCoveredText());
//            Map<String, Float> emotions = new HashMap<String, Float>();
//            FSArray<AnnotationComment> emotions_all = emotion.getEmotions();
//            for (AnnotationComment comment_i: emotions_all){
//                emotions.put(comment_i.getKey(), Float.parseFloat(comment_i.getValue()));
//                System.out.println("key:"+comment_i.getKey()+"; Value:"+comment_i.getValue());
//            }
//            expected.add(emotions);
//        }
//
//        // expected values
//        ArrayList<String> expected_emotions = new ArrayList<String>();
//        expected_emotions.add("anger");
//        expected_emotions.add("joy");
//        for (Map<String, Float> emotion: expected){
//            // highest value
//            String key = Collections.max(emotion.entrySet(), Map.Entry.comparingByValue()).getKey();
//            Assertions.assertEquals(expected_emotions.get(expected.indexOf(emotion)), key);
//        }
//    }
//
//
//    @Test
//    public void TurkishTest() throws Exception {
////        composer.add(new DUUIDockerDriver.
////                Component("docker.texttechnologylab.org/textimager-duui-transformers-topic:0.0.1")
////                .withParameter("model_name", model)
////                .withParameter("selection", "text,de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
////                .withScale(1)
////                .withImageFetching());
//        composer.add(
//                new DUUIRemoteDriver.Component(url)
//                        .withParameter("model_name", model)
//                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
//        );
//        List<String> sentences = Arrays.asList(
//                "Seni nefret ediyorum. Çok sinirliyim.",
//                "Burada olmaktan çok mutluyum. Bu yeri seviyorum."
//        );
//
//        createCas("tr", sentences);
//
//        composer.run(cas);
//
//        Collection<Emotion> all_emotions = JCasUtil.select(cas, Emotion.class);
////        System.out.println(topics.size());
//        ArrayList<Map<String, Float>> expected = new ArrayList<Map<String, Float>>();
//        for (Emotion emotion: all_emotions){
//            System.out.println(emotion.getCoveredText());
//            Map<String, Float> emotions = new HashMap<String, Float>();
//            FSArray<AnnotationComment> emotions_all = emotion.getEmotions();
//            for (AnnotationComment comment_i: emotions_all){
//                emotions.put(comment_i.getKey(), Float.parseFloat(comment_i.getValue()));
//                System.out.println("key:"+comment_i.getKey()+"; Value:"+comment_i.getValue());
//            }
//            expected.add(emotions);
//        }
//
//        // expected values
//        ArrayList<String> expected_emotions = new ArrayList<String>();
//        expected_emotions.add("anger");
//        expected_emotions.add("joy");
//        for (Map<String, Float> emotion: expected){
//            // highest value
//            String key = Collections.max(emotion.entrySet(), Map.Entry.comparingByValue()).getKey();
//            Assertions.assertEquals(expected_emotions.get(expected.indexOf(emotion)), key);
//        }
//    }
}
