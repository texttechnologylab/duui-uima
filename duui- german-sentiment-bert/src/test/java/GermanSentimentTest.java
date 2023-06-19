import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.CasIOUtils;
import org.dkpro.core.io.xmi.XmiWriter;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.AsyncCollectionReader;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.SentimentBert;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class GermanSentimentTest {

    @Test
    public void annotateSentencesForTestJCas() throws Exception {

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withWorkers(1);         // wir geben dem Composer eine Anzahl an Threads mit.

        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIUIMADriver uima_driver = new DUUIUIMADriver()
                .withDebug(true);

        composer.addDriver(docker_driver, uima_driver);

        composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/textimager-duui-spacy-single-de_core_news_sm:0.1.4")
                .withScale(1)
                .build());

        // Hinzufügen einer UIMA-Componente zum schreiben der Ergebnisse
        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, "./testdata/xmi",
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());


        JCas jc = JCasFactory.createText("Ich liebe Erdbeeren sehr. Ich hasse Oliven, da sie mir nicht schecken.");

        DocumentMetaData dmd = DocumentMetaData.create(jc);
        File pFile = new File( "/test_jCas");
        dmd.setDocumentId(pFile.getName());
        dmd.setDocumentTitle(pFile.getName());
        dmd.setDocumentUri(pFile.getAbsolutePath());
        dmd.addToIndexes();

        composer.run(jc, "test");
    }

    @Test
    public void test() throws Exception {

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withWorkers(1);         // wir geben dem Composer eine Anzahl an Threads mit.


        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver(1000);
        composer.addDriver(docker_driver, remoteDriver);


//        composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/german-sentiment-bert:latest")
//                .withScale(1)
//                .build());
//
        composer.add(new DUUIRemoteDriver.Component("http://localhost:9716")
                .withScale(1)
                .build());

        JCas jCas = JCasFactory.createText("Dies ist ein schöner Test. Leider weiß ich nicht weiter.", "de");
        //CasIOUtils.load(new File("/home/gabrami/Downloads/xmiExample/18001.xmi").toURI().toURL(), jCas.getCas());

        new Sentence(jCas, 0, jCas.getDocumentText().indexOf(".")).addToIndexes();
        new Sentence(jCas, jCas.getDocumentText().indexOf("."), jCas.getDocumentText().lastIndexOf(".")).addToIndexes();

        System.out.println(JCasUtil.select(jCas, Sentence.class).size());

        composer.run(jCas, "test");

        for(SentimentBert sentimentBert: JCasUtil.select(jCas, SentimentBert.class)){
            System.out.println(sentimentBert);
        }

    }

    @Test
    public void testWithAsyncReaderAndXmiRider() throws Exception {

        // Input- und Output-Pfade
        String sInputPath = "./testdata/xmi";
        String sOutputPath = "./testdata/xmi_output";
        String sSuffix = "xmi";

        int workers = 1;

        // Asynchroner reader für die Input-Dateien
        AsyncCollectionReader pCorpusReader = new AsyncCollectionReader(sInputPath, sSuffix, 1, false);
        new File(sOutputPath).mkdir();

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withWorkers(workers);         // wir geben dem Composer eine Anzahl an Threads mit.

        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        DUUIUIMADriver uima_driver = new DUUIUIMADriver()
                .withDebug(false);
        composer.addDriver(docker_driver, remoteDriver, uima_driver);

        composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/german-sentiment-bert:latest")
                .withScale(workers)
                .build());

        /*composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
                .withScale(1)
                .build());*/

        // Hinzufügen einer UIMA-Componente zum schreiben der Ergebnisse
        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, sOutputPath,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());

        long startTime = System.nanoTime();
        composer.run(pCorpusReader, "test");
        long endTime = System.nanoTime();
        System.out.println("Time passed: " + formatNanoSeconds(endTime-startTime));

    }

    @Test
    public void analyseParlamentData() throws Exception {

        // Input- und Output-Pfade
        String sInputPath = "/home/max/uni/testdata/input/ThirdReich";
        String sOutputPath = "/home/max/uni/testdata/output/ThirdReich/SentimentBert";
        String sSuffix = "xmi.gz";

        int workers = 2;

        // Asynchroner reader für die Input-Dateien
        AsyncCollectionReader pCorpusReader = new AsyncCollectionReader(sInputPath, sSuffix, 1, false);
        new File(sOutputPath).mkdir();

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withWorkers(workers);         // wir geben dem Composer eine Anzahl an Threads mit.

        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        DUUIUIMADriver uima_driver = new DUUIUIMADriver()
                .withDebug(false);
        composer.addDriver(docker_driver, remoteDriver, uima_driver);

        composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/german-sentiment-bert:latest")
                .withScale(workers)
                .build());

        /*composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
                .withScale(1)
                .build());*/

        // Hinzufügen einer UIMA-Componente zum schreiben der Ergebnisse
        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, sOutputPath,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());

        long startTime = System.nanoTime();
        composer.run(pCorpusReader, "test");
        long endTime = System.nanoTime();
        System.out.println("Time passed: " + formatNanoSeconds(endTime-startTime));

    }

    @Test
    public void print_analysed_parlament_data() throws Exception {
        String sOutputPath = "/home/max/uni/testdata/output/ThirdReich";
        Files.walk(Paths.get(sOutputPath)).forEach((Path path) -> {
            try{
                if (Files.isRegularFile(path)){
                    File file = path.toFile();
                    JCas jCas = JCasFactory.createJCas();
                    CasIOUtils.load(file.toURI().toURL(), jCas.getCas());

                    Map<Integer, Integer> sentimentCounter = new HashMap<>();
                    sentimentCounter.put(0, 0);
                    sentimentCounter.put(1, 0);
                    sentimentCounter.put(-1, 0);
                    for(SentimentBert sentimentBert: JCasUtil.select(jCas, SentimentBert.class)){
                        sentimentCounter.put(sentimentBert.getSentiment(), sentimentCounter.get(sentimentBert.getSentiment())+1);
                    }
                    System.out.println(file.getName());
                    for (int sentiment : sentimentCounter.keySet()) {
                        System.out.println(sentiment + ":" + sentimentCounter.get(sentiment));
                    }
                }
            } catch (Exception e){
                e.printStackTrace();
                throw new RuntimeException();
            }

        });
    }

    public static String formatNanoSeconds(long nanoSeconds){
        double runTime = Math.round(((double) nanoSeconds) / 10_000_000.0) / 100.0;
        return runTime + "s";
    }


}
