package org.texttechnologylab.textimager.uima.rust;

import de.tudarmstadt.ukp.dkpro.core.api.ner.type.Location;
import org.apache.uima.cas.CASException;
import org.apache.uima.cas.SelectFSs;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.geonames.GeoNamesEntity;

import java.util.ArrayList;
import java.util.List;

public class TestGeoNamesFst {
    private static JCas getJCas() throws ResourceInitializationException, CASException {
        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentText(
                "Frankfurt am Main ist eine Stadt in Deutschland. Der Große Feldberg ist der höchste Berg des im Südwesten Hessens gelegenen Mittelgebirges Taunus und des gesamten Rheinischen Schiefergebirges."
        );
        jCas.setDocumentLanguage("de");

        ArrayList<Location> locations = new ArrayList<>();
        locations.add(new Location(jCas, 0, 17));
        locations.add(new Location(jCas, 36, 47));
        locations.add(new Location(jCas, 53, 67));
        locations.add(new Location(jCas, 106, 113));
        locations.add(new Location(jCas, 139, 145));
        locations.add(new Location(jCas, 163, 191));
        locations.forEach(ne -> ne.addToIndexes(jCas));
        return jCas;
    }

    @Test
    public void test_find() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary()
                )
                .withSkipVerification(true);

        composer.addDriver(new DUUIRemoteDriver(10000));
        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714")
                        .withName("duui-geonames-fst")
        );

        JCas jCas = getJCas();
        composer.run(jCas);
        composer.shutdown();

        List<GeoNamesEntity> annotations = jCas.select(GeoNamesEntity.class).toList();
        assert annotations.size() == 3;
        for (GeoNamesEntity gn : annotations) {
            StringBuffer stringBuffer = new StringBuffer();
            gn.prettyPrint(0, 2, stringBuffer, true);
            System.out.print(stringBuffer);
            System.out.println("\n  text: \"" + gn.getCoveredText() + "\"\n");
        }
    }

    @Test
    public void test_find_explicit() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary()
                )
                .withSkipVerification(true);

        composer.addDriver(new DUUIRemoteDriver(10000));
        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714")
                        .withName("duui-geonames-fst")
                        .withParameter("mode", "find")
                        .withParameter("result_selection", "first")
        );

        JCas jCas = getJCas();
        composer.run(jCas);
        composer.shutdown();

        List<GeoNamesEntity> annotations = jCas.select(GeoNamesEntity.class).toList();
        assert annotations.size() == 3;
        for (GeoNamesEntity gn : annotations) {
            StringBuffer stringBuffer = new StringBuffer();
            gn.prettyPrint(0, 2, stringBuffer, true);
            System.out.print(stringBuffer);
            System.out.println("\n  text: \"" + gn.getCoveredText() + "\"\n");
        }
    }

    /**
     * We can also search for matches that have (up to) a certain edit distance from query to match name.
     * This enables us to match
     */
    @Test
    public void test_levenshtein() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary()
                )
                .withSkipVerification(true);

        composer.addDriver(new DUUIRemoteDriver(10000));
        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714")
                        .withName("duui-geonames-fst")
                        .withParameter("mode", "levenshtein")
                        .withParameter("max_dist", "2")
                        .withParameter("result_selection", "first")
        );

        JCas jCas = getJCas();
        composer.run(jCas);
        composer.shutdown();

        List<GeoNamesEntity> annotations = jCas.select(GeoNamesEntity.class).toList();
        assert annotations.size() == 6;
        for (GeoNamesEntity gn : annotations) {
            StringBuffer stringBuffer = new StringBuffer();
            gn.prettyPrint(0, 2, stringBuffer, true);
            System.out.print(stringBuffer);
            System.out.println("\n  coveredText: \"" + gn.getCoveredText() + "\"\n");
        }
    }

    /**
     * Setting a lower state limit (default: 10000) restricts the matches proportional to the length of the query.
     */
    @Test
    public void test_levenshtein_state_limit() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary()
                )
                .withSkipVerification(true);

        composer.addDriver(new DUUIRemoteDriver(10000));
        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714")
                        .withName("duui-geonames-fst")
                        .withParameter("mode", "levenshtein")
                        .withParameter("max_dist", "2")
                        .withParameter("state_limit", "1000")
                        .withParameter("result_selection", "first")
        );

        JCas jCas = getJCas();
        composer.run(jCas);
        composer.shutdown();

        List<GeoNamesEntity> annotations = jCas.select(GeoNamesEntity.class).toList();
        assert annotations.size() == 1;
        for (GeoNamesEntity gn : annotations) {
            StringBuffer stringBuffer = new StringBuffer();
            gn.prettyPrint(0, 2, stringBuffer, true);
            System.out.print(stringBuffer);
            System.out.println("\n  coveredText: \"" + gn.getCoveredText() + "\"\n");
        }
    }
}
