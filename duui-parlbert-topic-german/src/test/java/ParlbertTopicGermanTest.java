import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.dkpro.core.io.xmi.XmiWriter;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.AsyncCollectionReader;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.hucompute.textimager.uima.type.category.CategoryCoveredTagged;

import java.io.File;
import java.util.Collection;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class ParlbertTopicGermanTest {

    public static final String dijkstraExampleText = "Der Algorithmus von Dijkstra (nach seinem Erfinder Edsger W. Dijkstra) ist ein Algorithmus aus der Klasse der Greedy-Algorithmen[1] und löst das Problem der kürzesten Pfade für einen gegebenen Startknoten. " +
            "Er berechnet somit einen kürzesten Pfad zwischen dem gegebenen Startknoten und einem der (oder allen) übrigen Knoten in einem kantengewichteten Graphen (sofern dieser keine Negativkanten enthält)." +
            "Für unzusammenhängende ungerichtete Graphen ist der Abstand zu denjenigen Knoten unendlich, zu denen kein Pfad vom Startknoten aus existiert. Dasselbe gilt auch für gerichtete nicht stark zusammenhängende Graphen. Dabei wird der Abstand synonym auch als Entfernung, Kosten oder Gewicht bezeichnet.";

    public static final String labels = "Tiere,Pflanzen,Technologie,Geschichte,Kunst,Musik,Politik,Bildung,Sport,Gesundheit,Reisen,Essen und Trinken,Filme,Literatur,Umwelt,Wissenschaft,Mode,Philosophie,Psychologie,Wirtschaft";

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
             composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/parlbert-topic-german:latest")
                    .withScale(iWorkers)
                    .build());
        }else{
            composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
                .withScale(iWorkers)
                .build());
        }

        // Create basic test jCas.
        JCas jCas = JCasFactory.createText(dijkstraExampleText, "de");

        new Sentence(jCas, 0, 206).addToIndexes();
        new Sentence(jCas, 206, 402).addToIndexes();
        new Sentence(jCas, 402, 544).addToIndexes();
        new Sentence(jCas, 544, 616).addToIndexes();
        new Sentence(jCas, 616, 699).addToIndexes();

        composer.run(jCas, "test");

        // Print Result
        Collection<CategoryCoveredTagged> categoryCoveredTaggeds = JCasUtil.select(jCas, CategoryCoveredTagged.class).stream().sorted((c1, c2) -> c1.getBegin()-c2.getBegin()).collect(Collectors.toList());
        for(CategoryCoveredTagged categoryCoveredTagged: categoryCoveredTaggeds){
            System.out.println(categoryCoveredTagged.getBegin() + " - " + categoryCoveredTagged.getEnd() + " " + categoryCoveredTagged.getValue() + ": " + categoryCoveredTagged.getScore());
        }

    }

    @Test
    public void analyseParlamentData() throws Exception {

        // Input- und Output-Pfade
        String sInputPath = "/home/max/uni/testdata/input/ThirdReich";
        String sOutputPath = "/home/max/uni/testdata/output/ThirdReich/ParlbertTopicGerman";
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
            composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/parlbert-topic-german:latest")
                    .withScale(iWorkers)
                    .build());
        }else{
            composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
                    .withScale(iWorkers)
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
