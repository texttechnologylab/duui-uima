/*
 * I do not really know Java. I know Java ish but not enough for this.
 * I'm a Python person who got handed a Java test and tried to
 * make it work. Most of my understanding of JUnit, JCas, and the
 * DUUI test patterns comes from three places:
 *
 *   1. ChatGPT: I asked it to explain basically everything.
 *      "What does @BeforeAll do?" "Why does JCas need a factory?"
 *      "What's the difference between @AfterEach and @AfterAll?"
 *      I used it as a tutor more than a code generator here.
 *
 *   2. GitHub Copilot: wrote most of the repetitive test methods
 *      after I got the first one working. Once Copilot saw the
 *      pattern for testPaddleOCRWithBase64, it basically generated
 *      the GLM variants with minimal prompting.
 *
 *   3. Existing DUUI test files: this is where the real skeleton
 *      came from. I studied and borrowed heavily from:
 *        - GermanSummaryTest.java (the createCas + Sentence pattern)
 *        - SentimentTest.java (the @BeforeAll/@AfterEach lifecycle,
 *          the composer setup, the general structure of everything)
 *        - TextToImageTest.java (the base64 image encoding/decoding,
 *          which was exactly what I needed for OCR image input)
 *      These three files are the real authors of the test scaffolding.
 *      I adapted their patterns for OCR.
 *
 * ITERATION HISTORY (mirrors the Python server's evolution):
 *
 *   v1: Tests for PaddleOCR-VL-1.5 only. This was the first model
 *       I got working end-to-end (Python server → DUUI → Java test).
 *       Took longer than it should have because I was learning JUnit
 *       and UIMA at the same time. But PaddleOCR-VL cooperated.
 *       The test structure comes from this iteration.
 *
 *   v2: Tried to add tests for microsoft/trocr-base-printed. Wrote
 *       the test methods, ran them, got results that were garbage.
 *       Spent time debugging on the Java side before realizing the
 *       problem wasn't here at all. Commented out below with notes
 *       on what went wrong. See the Python server file for the full
 *       post-mortem. Or don't, it's kinda embarrasing.
 *
 *   v3: Added tests for zai-org/GLM-OCR. Worked almost immediately.
 *       GLM-OCR is architecturally the same *kind* of model as
 *       PaddleOCR-VL (AutoModelForImageTextToText, chat templates,
 *       full-page OCR), so the test pattern from v1 transferred
 *       directly. Copilot generated most of these by pattern-matching
 *       on the Paddle tests. This is the current working state.
 *
 * Source for DUUI test conventions:
 * https://github.com/texttechnologylab/DockerUnifiedUIMAInterface
 *
 * Last meaningful edit: Feb 2026
 */

package org.hucompute.textimager.uima.ocr;

import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface
        .DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface
        .driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface
        .lua.DUUILuaContext;
import org.texttechnologylab.annotation.AnnotationComment;
import org.texttechnologylab.annotation.type.Image;
import org.xml.sax.SAXException;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

public class DUUIOCRTest {

    // ── Static fields / shared state ────────────────────────────────
    // BORROWED. This entire static block pattern is from SentimentTest.
    //
    // I asked ChatGPT why these are static and it explained that
    // @BeforeAll methods must be static in JUnit 5 (unless you change
    // the lifecycle mode), so anything they initialize also has to be
    // static. Coming from Python's pytest fixtures this felt weird,
    // but I get it now. Mostly.

    static DUUIComposer composer;
    static JCas cas;

    // Port 9714 is what I set in my docker-compose for the OCR server.
    // If you're running the server somewhere else, change this.
    static String url = "http://127.0.0.1:9714";
    static String paddleModel =
            "PaddlePaddle/PaddleOCR-VL-1.5";

    // ── v2: TrOCR model string ──────────────────────────────────
    // ABANDONED. This model exists and loads fine. The problem is
    // what it *does* with full-page images (spoiler: nothing useful).
    // Kept here as documentation of the attempt.
    //
    // Source: https://huggingface.co/microsoft/trocr-base-printed
    //
    // static String trOcrModel =
    //         "microsoft/trocr-base-printed";

    // v3: GLM-OCR, the model that actually worked as a second option.
    // Source: https://huggingface.co/zai-org/GLM-OCR
    static String glmModel = "zai-org/GLM-OCR";

    // I keep test images in a local directory. They're not committed
    // to the repo because they're 5-10MB each. You need to put your
    // own test images here before running these tests.
    static String testImageDir = "src/test_images";

    // ── Lifecycle methods ───────────────────────────────────────────

    @BeforeAll
    static void beforeAll() throws Exception {
        // BORROWED. Copied almost verbatim from SentimentTest.beforeAll().
        // withSkipVerification(true) skips some DUUI handshake checks
        // that were failing locally. ChatGPT told me this is fine for
        // testing but you probably shouldn't do it in production.
        // withJsonLibrary() adds JSON support to the Lua context,
        // which the communication script needs. Concerning...
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(
                        new DUUILuaContext().withJsonLibrary()
                );

        DUUIRemoteDriver remoteDriver =
                new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);

        // ChatGPT explained: JCasFactory.createJCas() needs a type
        // system on the classpath. If this line throws a
        // ResourceInitializationException, it usually means the
        // type system XML isn't where UIMA expects it. I spent an
        // hour on that before realizing my pom.xml was missing a
        // dependency. Not my proudest debugging session.
        cas = JCasFactory.createJCas();
    }

    @AfterAll
    static void afterAll() throws UnknownHostException {
        composer.shutdown();
    }

    @AfterEach
    public void afterEach()
            throws IOException, SAXException {
        // BORROWED from SentimentTest. Nothing else to say.
        composer.resetPipeline();

        ByteArrayOutputStream stream =
                new ByteArrayOutputStream();
        XmlCasSerializer.serialize(
                cas.getCas(), null, stream
        );
        System.out.println(
                stream.toString(StandardCharsets.UTF_8)
        );

        cas.reset();
    }

    // ── Helper methods ──────────────────────────────────────────────

    private static String imageToBase64(String imagePath) {
        // BORROWED. This is essentially the same as
        // convertImageToBase64() from TextToImageTest.java, just
        // renamed to match my naming convention. The pattern is:
        // read file → BufferedImage → write to ByteArrayOutputStream
        // as PNG → base64-encode the bytes.
        //
        // I asked ChatGPT why we can't just read the raw file bytes
        // and encode those directly. Answer: we *could*, but going
        // through BufferedImage + ImageIO normalizes the format.
        // A JPEG file's raw bytes would be JPEG-encoded, and this
        // way we always send PNG regardless of the source format.
        // Honestly not sure if the server cares, but TextToImageTest
        // did it this way and it works.
        try {
            File file = new File(imagePath);
            BufferedImage bufferedImage =
                    ImageIO.read(file);
            ByteArrayOutputStream baos =
                    new ByteArrayOutputStream();
            ImageIO.write(bufferedImage, "png", baos);
            return Base64.getEncoder()
                    .encodeToString(baos.toByteArray());
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private List<String> collectTestImagePaths() {
        // COPILOT + CHATGPT. I told Copilot "find all image files
        // in a directory" and it generated most of this. ChatGPT
        // helped me understand the Objects.requireNonNull() part.
        //
        // FRAGILE. This only checks file extensions, not actual
        // file content. A corrupted PNG with a .png extension will
        // pass this filter and fail later in imageToBase64(). I was
        // hungry so this is good enough for testing.
        List<String> paths = new ArrayList<>();
        File dir = new File(testImageDir);
        if (dir.exists() && dir.isDirectory()) {
            for (File file :
                    Objects.requireNonNull(
                            dir.listFiles()
                    )) {
                String name = file.getName().toLowerCase();
                if (name.endsWith(".png")
                        || name.endsWith(".jpg")
                        || name.endsWith(".jpeg")) {
                    paths.add(file.getAbsolutePath());
                }
            }
        }
        return paths;
    }

    private void createCasWithBase64Images(
            String language, List<String> imagePaths) {
        // BORROWED + COPILOT. Adapted from the createCas() pattern in
        // GermanSummaryTest and SentimentTest, but instead of adding
        // Sentence annotations, I'm adding Image annotations with
        // base64-encoded src data.
        //
        // The "OCR document" placeholder text is there because UIMA
        // requires a document text. You can't have a CAS with no text.
        // ChatGPT explained that begin/end offsets on annotations must
        // fall within the document text range, so I set all images to
        // span the full "OCR document" string (0 to length).
        cas.setDocumentLanguage(language);
        cas.setDocumentText("OCR document");
        for (String path : imagePaths) {
            String b64 = imageToBase64(path);
            if (b64 != null) {
                Image img = new Image(
                        cas, 0, cas.getDocumentText().length()
                );
                img.setSrc(b64);
                img.addToIndexes();
            }
        }
    }

    private void createCasWithFilePaths(
            String language, List<String> imagePaths) {
        // Same as above but passes file paths instead of base64 data.
        cas.setDocumentLanguage(language);
        cas.setDocumentText("OCR document");
        for (String path : imagePaths) {
            Image img = new Image(
                    cas, 0, cas.getDocumentText().length()
            );
            img.setSrc(path);
            img.addToIndexes();
        }
    }

    private void printResults() {
        // BORROWED. The pattern of selecting annotations by type and
        // printing them is everywhere in the DUUI test files.
        // SentimentTest does this with SentimentModel, TextToImageTest
        // with Image, and I do it with AnnotationComment.
        Collection<AnnotationComment> results =
                JCasUtil.select(cas, AnnotationComment.class);
        for (AnnotationComment c : results) {
            System.out.println(
                    "Key: " + c.getKey()
                            + " | Value: " + c.getValue()
            );
        }
    }

    // ── PaddleOCR-VL tests (v1) ────────────────────────────────────
    // These were the first tests I wrote. The whole test structure I
    // use everywhere else in this file crystallized during this
    // iteration: compose a pipeline with a model name and task, load
    // images into the CAS, run the pipeline, check that
    // AnnotationComments came back.
    //
    // Everything after this section is a variation on this pattern.

    @Test
    public void testPaddleOCRWithBase64() throws Exception {
        // COPILOT + CHATGPT. This is the first test I got running.
        // It took an embarrassingly long time. The composer.add()
        // pattern with .withParameter() is from SentimentTest. ChatGPT
        // walked me through how DUUIRemoteDriver.Component works:
        // you give it a URL and parameters, and those parameters get
        // passed to the Python server as part of the DUUI protocol.
        //
        // The assertions are minimal - I just check that:
        //   1. I actually have test images (otherwise what are we testing?)
        //   2. At least one AnnotationComment came back
        //
        // I don't check the *content* of the OCR results because that
        // depends on what test images you have. SentimentTest checks
        // exact probability values, which is possible because text
        // input is deterministic. OCR results vary with the image, so
        // I just verify something came back and eyeball the printResults()
        // output. Not ideal, I know.
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", paddleModel)
                        .withParameter("task", "ocr")
        );
        List<String> paths = collectTestImagePaths();
        assertFalse(paths.isEmpty(),
                "No test images in " + testImageDir);
        createCasWithBase64Images("en", paths);
        composer.run(cas);
        assertFalse(
                JCasUtil.select(
                        cas, AnnotationComment.class
                ).isEmpty(),
                "No OCR results returned"
        );
        printResults();
    }

    @Test
    public void testPaddleOCRWithFilePaths()
            throws Exception {
        // Same as testPaddleOCRWithBase64 but sends file paths instead
        // of base64. Copilot generated this after seeing the base64
        // version. It just swapped createCasWithBase64Images for
        // createCasWithFilePaths. Amazeballs.
        //
        // This test will fail if the server is containerized and
        // can't see your local filesystem.
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", paddleModel)
                        .withParameter("task", "ocr")
        );
        List<String> paths = collectTestImagePaths();
        assertFalse(paths.isEmpty());
        createCasWithFilePaths("en", paths);
        composer.run(cas);
        assertFalse(
                JCasUtil.select(
                        cas, AnnotationComment.class
                ).isEmpty()
        );
        printResults();
    }

    @Test
    public void testPaddleTableRecognition()
            throws Exception {
        // COPILOT generated this and the formula test below almost
        // entirely on its own after seeing the OCR tests above.
        // The only difference is .withParameter("task", "table").
        //
        // I only send a single image here (paths.get(0)) because table
        // recognition is slower and I don't need to batch-test it.
        // ChatGPT told me Collections.singletonList() is the Java way
        // to make a one-element list. In Python I'd just write [paths[0]].
        // Java is... verbose.
        //
        // REVISIT. I should really use an image that actually contains
        // a table for this test. Right now I'm just sending whatever
        // the first image in the directory is, which is probably a
        // regular text page. The model still returns *something*, but
        // it's not a meaningful test of table recognition quality.
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", paddleModel)
                        .withParameter("task", "table")
        );
        List<String> paths = collectTestImagePaths();
        assertFalse(paths.isEmpty());
        createCasWithBase64Images(
                "en",
                Collections.singletonList(paths.get(0))
        );
        composer.run(cas);
        printResults();
    }

    @Test
    public void testPaddleFormulaRecognition()
            throws Exception {
        // COPILOT. Same pattern as testPaddleTableRecognition, just
        // with task="formula". Copilot's autocomplete at this point
        // was finishing entire test methods before I could type the
        // method name. Saved me a lot of time, all hail the AI-Overlord.
        //
        // Same REVISIT as above: should use an image with actual
        // mathematical formulas.
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", paddleModel)
                        .withParameter("task", "formula")
        );
        List<String> paths = collectTestImagePaths();
        assertFalse(paths.isEmpty());
        createCasWithBase64Images(
                "en",
                Collections.singletonList(paths.get(0))
        );
        composer.run(cas);
        printResults();
    }

    // ── TrOCR tests (v2) - ABANDONED :( ──────────────────────────────
    //
    // This is why I should do more reading and research before doing.

    // What happened:
    //   I wanted a second model to count towards my duui-module count
    //   microsoft/trocr-base-printed seemed perfect: well-known,
    //   well-documented, specifically designed for OCR. I wrote the
    //   tests below by duplicating the PaddleOCR tests and swapping
    //   in the TrOCR model name.
    //
    //   The tests *ran* but failed. I assumed I had a bug in how I
    //   was building the CAS, or in the base64 encoding, or in the Lua
    //   communication script. I spent hours debugging before giving up.
    //   I commented everything out. Moved on to GLM-OCR (v3),
    //   which turned out to be the right call.
    //
    //   Leaving these tests here as documentation. If anyone in the
    //   future wants to integrate TrOCR, they need to know it requires
    //   a fundamentally different approach: pre-crop text lines first,
    //   then feed each line to TrOCR individually.
    //
    // BORROWED. Test structure is identical to the Paddle tests.
    // The problem was never in the test code. It was in the assumption
    // that TrOCR could handle the same input as PaddleOCR.

    // @Test
    // public void testTrOCRWithBase64() throws Exception {
    //     // ABANDONED. Too sad to comment further.
    //     composer.add(
    //             new DUUIRemoteDriver.Component(url)
    //                     .withParameter("model_name", trOcrModel)
    //                     .withParameter("task", "ocr")
    //     );
    //     List<String> paths = collectTestImagePaths();
    //     assertFalse(paths.isEmpty(),
    //             "No test images in " + testImageDir);
    //     createCasWithBase64Images("en", paths);
    //     composer.run(cas);
    //     // This assertion *passes* — results do come back.
    //     // They're just meaningless.
    //     assertFalse(
    //             JCasUtil.select(
    //                     cas, AnnotationComment.class
    //             ).isEmpty(),
    //             "No OCR results returned"
    //     );
    //     printResults();
    // }

    // @Test
    // public void testTrOCRWithFilePaths() throws Exception {
    //     // ABANDONED. Same story as testTrOCRWithBase64.
    //     composer.add(
    //             new DUUIRemoteDriver.Component(url)
    //                     .withParameter("model_name", trOcrModel)
    //                     .withParameter("task", "ocr")
    //     );
    //     List<String> paths = collectTestImagePaths();
    //     assertFalse(paths.isEmpty());
    //     createCasWithFilePaths("en", paths);
    //     composer.run(cas);
    //     assertFalse(
    //             JCasUtil.select(
    //                     cas, AnnotationComment.class
    //             ).isEmpty()
    //     );
    //     printResults();
    // }

    // @Test
    // public void testTrOCRTableRecognition() throws Exception {
    //     // ABANDONED. I wrote this knowing it probably wouldn't work,
    //     // because TrOCR only supports OCR, so no table recognition,
    //     // no formula recognition, nothing. It's a single-task model.
    //     // On the Python side, the TrOCR backend ignores the task
    //     // parameter entirely. But I wrote the test anyway to see
    //     // what would happen.
    //     //
    //     // What happened: the server accepted the request (it falls
    //     // back to basic OCR when the task isn't supported), and
    //     // returned the same garbage the OCR test. No table structure,
    //     // obviously.
    //     //
    //     // This was the moment I started suspecting the problem was
    //     // deeper than a bug. Three different test configurations,
    //     // all returning the same kind of fragmented output. That's
    //     // not a bug, that's a model doing what it was designed to do
    //     // on input it was never designed to handle.
    //     composer.add(
    //             new DUUIRemoteDriver.Component(url)
    //                     .withParameter("model_name", trOcrModel)
    //                     .withParameter("task", "table")
    //     );
    //     List<String> paths = collectTestImagePaths();
    //     assertFalse(paths.isEmpty());
    //     createCasWithBase64Images(
    //             "en",
    //             Collections.singletonList(paths.get(0))
    //     );
    //     composer.run(cas);
    //     printResults();
    // }

    // @Test
    // public void testTrOCRFormulaRecognition() throws Exception {
    //     // ABANDONED. Same as table, TrOCR doesn't do formulas.
    //     // Included for completeness.
    //     composer.add(
    //             new DUUIRemoteDriver.Component(url)
    //                     .withParameter("model_name", trOcrModel)
    //                     .withParameter("task", "formula")
    //     );
    //     List<String> paths = collectTestImagePaths();
    //     assertFalse(paths.isEmpty());
    //     createCasWithBase64Images(
    //             "en",
    //             Collections.singletonList(paths.get(0))
    //     );
    //     composer.run(cas);
    //     printResults();
    // }
    //
    // End of abandoned TrOCR tests.

    // ── GLM-OCR tests (v3) ─────────────────────────────────────────
    // After the TrOCR failure I was genuinely nervous about trying a
    // third model. But GLM-OCR (zai-org/GLM-OCR) turned out to be
    // almost suspiciously easy. These tests worked on the first run.
    //
    // Source: https://huggingface.co/zai-org/GLM-OCR
    //
    // BORROWED. The strategy of mirroring every test across models
    // is something I saw in SentimentTest, which has separate tests
    // for different languages and model variants (DeTest, EnTest,
    // EnCadriffNLPTest, VietnamesePhoBertTest, etc.). Each one is
    // basically the same flow with different parameters. Repetitive
    // but easy to read and debug.
    //
    // Copilot generated all four of these by pattern-matching on the
    // Paddle tests above. I only had to change the model string and
    // the method names.

    @Test
    public void testGlmOCRWithBase64() throws Exception {
        // COPILOT. Generated by duplicating testPaddleOCRWithBase64
        // and changing paddleModel to glmModel.
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", glmModel)
                        .withParameter("task", "ocr")
        );
        List<String> paths = collectTestImagePaths();
        assertFalse(paths.isEmpty(),
                "No test images in " + testImageDir);
        createCasWithBase64Images("en", paths);
        composer.run(cas);
        assertFalse(
                JCasUtil.select(
                        cas, AnnotationComment.class
                ).isEmpty(),
                "No OCR results returned"
        );
        printResults();
    }

    @Test
    public void testGlmOCRWithFilePaths()
            throws Exception {
        // COPILOT generated. Same file path caveat as the Paddle
        // file path test (Docker path resolution).
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", glmModel)
                        .withParameter("task", "ocr")
        );
        List<String> paths = collectTestImagePaths();
        assertFalse(paths.isEmpty());
        createCasWithFilePaths("en", paths);
        composer.run(cas);
        assertFalse(
                JCasUtil.select(
                        cas, AnnotationComment.class
                ).isEmpty()
        );
        printResults();
    }

    @Test
    public void testGlmTableRecognition()
            throws Exception {
        // COPILOT generated. Same REVISIT about using a real table
        // image applies here.
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", glmModel)
                        .withParameter("task", "table")
        );
        List<String> paths = collectTestImagePaths();
        assertFalse(paths.isEmpty());
        createCasWithBase64Images(
                "en",
                Collections.singletonList(paths.get(0))
        );
        composer.run(cas);
        printResults();
    }

    @Test
    public void testGlmFormulaRecognition()
            throws Exception {
        // COPILOT generated. Last of the mirrored tests. At this
        // point Copilot was basically writing the whole method from
        // the method name alone.
        //
        // REVISIT. I have eight active test methods (four Paddle,
        // four GLM) and they're all structurally identical. ChatGPT
        // suggested using @ParameterizedTest with a @MethodSource to
        // collapse these into one or two parameterized methods
        // (model × task × input mode). That would be cleaner but I
        // haven't learned JUnit parameterized tests yet. The
        // SentimentTest and TextToImageTest files both use separate
        // methods per scenario too, so at least I'm in good company.
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", glmModel)
                        .withParameter("task", "formula")
        );
        List<String> paths = collectTestImagePaths();
        assertFalse(paths.isEmpty());
        createCasWithBase64Images(
                "en",
                Collections.singletonList(paths.get(0))
        );
        composer.run(cas);
        printResults();
    }
}