import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.texttechnologylab.annotation.Language;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.dkpro.core.io.xmi.XmiWriter;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.AsyncCollectionReader;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import java.io.File;
import java.util.Collection;
import java.util.stream.Collectors;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class LanguageDetectionTest {

    @Test
    public void test() throws Exception {

        int iWorkers = 1;
        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withWorkers(iWorkers);         // wir geben dem Composer eine Anzahl an Threads mit.


        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver(1000);
        composer.addDriver(docker_driver, remoteDriver);

        boolean useDockerImage = true;
        if (useDockerImage){
             composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/xlm-roberta-base-language-detection:latest")
                     .withParameter("annotationClassPath", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
                     .withScale(iWorkers)
                     .withParameter("top_k", "1")
                     .build());
        }else{
            composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
                .withScale(iWorkers)
                .withParameter("annotationClassPath", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
                .withParameter("top_k", "1")
                .build());
        }

        // Create basic test jCas.
        String exampleText = "The weather today is quite pleasant, with clear skies and a gentle breeze.\n" +
                "Me gusta aprender nuevos idiomas porque es una forma fascinante de explorar diferentes culturas.\n" +
                "Les livres sont une source inestimable de connaissances et d'évasion dans des mondes fantastiques.\n" +
                "Die Welt der Technologie entwickelt sich rasant weiter, und wir sind Zeugen faszinierender Innovationen.\n" +
                "La cucina italiana è famosa per la sua deliziosa pasta e i formaggi prelibati.\n"+
                "我喜欢在春天漫步，欣赏盛开的花朵。";
        JCas jCas = JCasFactory.createText(exampleText, "de");

        new Sentence(jCas, 0, 75).addToIndexes();
        new Sentence(jCas, 75, 172).addToIndexes();
        new Sentence(jCas, 172, 271).addToIndexes();
        new Sentence(jCas, 271, 376).addToIndexes();
        new Sentence(jCas, 376, 455).addToIndexes();
        new Sentence(jCas, 455, 472).addToIndexes();

        composer.run(jCas, "test");


        // Print Result
        System.out.println("\n\n");
        Collection<Language> languages = JCasUtil.select(jCas, Language.class).stream().sorted(
                (l1, l2) -> l1.getBegin()-l2.getBegin()
        ).collect(Collectors.toList());

        for(Language language: languages){
            System.out.println(language.getBegin() + " - " + language.getEnd() + ": " + language.getValue() + ": " + language.getScore());
        }
        System.out.println("\n\n");

    }

    @Test
    public void analyseParlamentData() throws Exception {

        // Input- und Output-Pfade
        String sInputPath = "/home/max/uni/testdata/input/ThirdReich";
        String sOutputPath = "/home/max/uni/testdata/output/ThirdReich/LanguageDetection";
        String sSuffix = "xmi.gz";

        int iWorkers = 1;

        // Asynchroner reader für die Input-Dateien
        AsyncCollectionReader pCorpusReader = new AsyncCollectionReader(sInputPath, sSuffix, 1, false);
        new File(sOutputPath).mkdir();

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withWorkers(iWorkers);         // wir geben dem Composer eine Anzahl an Threads mit.

        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIUIMADriver uima_driver = new DUUIUIMADriver()
                .withDebug(false);
        DUUIRemoteDriver duuiRemoteDriver = new DUUIRemoteDriver();
        composer.addDriver(docker_driver, uima_driver, duuiRemoteDriver);

        boolean useDockerImage = false;
        if (useDockerImage){
            composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/xlm-roberta-base-language-detection:latest")
                    .withScale(iWorkers)
                    .build());
        }else{
            composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
                    .withScale(iWorkers)
                    .withParameter("annotationClassPath", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
                    .build());
        }

        // Hinzufügen einer UIMA-Componente zum schreiben der Ergebnisse
        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, sOutputPath,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());

        long startTime = System.nanoTime();
        composer.run(pCorpusReader, "test");

        // Print the past time in seconds
        long endTime = System.nanoTime();
        System.out.println("Time passed: " + formatNanoSeconds(endTime-startTime));
    }

    public static String formatNanoSeconds(long nanoSeconds){
        double runTime = Math.round(((double) nanoSeconds) / 10_000_000.0) / 100.0;
        return runTime + "s";
    }


}
