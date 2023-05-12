package org.texttechnologylab.duui.io;

import org.apache.uima.UIMAException;
import org.apache.uima.collection.CollectionReaderDescription;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.dkpro.core.io.xmi.XmiWriter;
import org.hucompute.textimager.uima.spacy.SpaCyMultiTagger3;
import org.texttechnologylab.duui.io.borland.BorlandEdge;
import org.texttechnologylab.duui.io.borland.BorlandMapping;
import org.texttechnologylab.duui.io.borland.BorlandNode;
import org.texttechnologylab.duui.io.borland.BorlandWriter;
import org.texttechnologylab.duui.io.elasticsearch.ElasticsearchReader;
import org.texttechnologylab.duui.io.json.JsonReader;

import java.io.IOException;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
import static org.apache.uima.fit.factory.CollectionReaderFactory.createReaderDescription;

public class TextImagerTest {

    /**
     * main function to test the reader and writer and compare the result
     *
     * @param args
     * @throws UIMAException
     * @throws IOException
     */
    public static void main(String[] args) throws UIMAException, IOException {

        testWriter();
        //testReader();
        //System.out.println(org.texttechnologylab.duui.io.CompareFiles.compare("src/main/resources/18001.xmi.gz.xmi", "src/main/resources/elasticsearchDocker/18001.xmi.gz.org.texttechnologylab.duui.io.json.xmi"));
    }

    /**
     * Function to test the writer
     *
     * @throws UIMAException
     * @throws IOException
     */
    public static void testWriter() throws UIMAException, IOException {
        AggregateBuilder pipeline = new AggregateBuilder();

        CollectionReaderDescription reader = null;

        reader = createReaderDescription(
                JsonReader.class,
                JsonReader.PARAM_SOURCE_LOCATION, "src/main/resources/test/Bundestag/18/",
                JsonReader.PARAM_PATTERNS, "*.json",
                JsonReader.PARAM_LANGUAGE, "de"
        );


        pipeline.add(createEngineDescription(SpaCyMultiTagger3.class,
                SpaCyMultiTagger3.PARAM_REST_ENDPOINT, "http://huaxal.hucompute.org:8103",
                SpaCyMultiTagger3.PARAM_MAX_TEXT_WINDOW, 90000));

        /*
        JSONObject mapping = new JSONObject();
        JSONArray nodes = new JSONArray();
        JSONArray edges = new JSONArray();
        JSONArray groups = new JSONArray();
         */

        /*
        // GraphML: NamedEntity & Token
        nodes.put(new JSONObject().put("typeID", "0").put("class", "NamedEntity").put("label", "value").put("isUnique", true));
        nodes.put(new JSONObject().put("typeID", "1").put("class", "Token").put("label", "text"));
        edges.put(new JSONObject().put("source", "0").put("target", "1"));
        */

        /*
        // GraphML: NamedEntity & Token & POS
        nodes.put(new JSONObject().put("typeID", "0").put("class", "NamedEntity").put("label", "value"));
        nodes.put(new JSONObject().put("typeID", "1").put("class", "Token").put("label", "text"));
        nodes.put(new JSONObject().put("typeID", "2").put("class", "POS").put("label", "PosValue").put("isUnique", true));
        edges.put(new JSONObject().put("source", "0").put("target", "1"));
        edges.put(new JSONObject().put("source", "1").put("target", "2"));
        groups.put(new JSONObject().put("typeID", "3").put("nodes", new JSONArray().put("1").put("0")));
        groups.put(new JSONObject().put("typeID", "4").put("nodes", new JSONArray().put("2").put("3").put("1")));
         */

        /*
        // GraphML:  Dependencies
        nodes.put(new JSONObject().put("typeID", "1").put("class", "Dependency").put("label", "Dependent"));
        nodes.put(new JSONObject().put("typeID", "2").put("class", "Dependency").put("label", "Governor"));
        edges.put(new JSONObject().put("class", "Dependency").put("label", "DependencyType").put("source", "1").put("target", "2").put("directed", true));
        */

        /*
        // Borland: NamedEntity & Token & POS
        nodes.put(new JSONObject().put("typeID", "0").put("class", "NamedEntity").put("isUnique", true).put("attributes", new JSONArray().put(new JSONObject().put("value", "value").put("type", "String"))));
        nodes.put(new JSONObject().put("typeID", "1").put("class", "Token").put("attributes", new JSONArray().put(new JSONObject().put("value", "text").put("type", "String"))));
        nodes.put(new JSONObject().put("typeID", "2").put("class", "POS").put("isUnique", true).put("attributes", new JSONArray().put(new JSONObject().put("value", "PosValue").put("type", "String")).put(new JSONObject().put("value", "coarseValue").put("type", "String"))));
        edges.put(new JSONObject().put("source", "0").put("target", "1"));
        edges.put(new JSONObject().put("source", "1").put("target", "2"));
         */

        /*
        // Borland: Dependencies
        nodes.put(new JSONObject().put("typeID", "1").put("class", "Dependency").put("attributes", new JSONArray().put(new JSONObject().put("value", "Dependent").put("type", "String"))));
        nodes.put(new JSONObject().put("typeID", "2").put("class", "Dependency").put("attributes", new JSONArray().put(new JSONObject().put("value", "Governor").put("type", "String"))));
        edges.put(new JSONObject().put("class", "Dependency").put("attributes", new JSONArray().put(new JSONObject().put("value", "DependencyType").put("type", "String"))).put("source", "1").put("target", "2"));
         */

        /*
        mapping.put("nodes", nodes);
        mapping.put("edges", edges);
        //mapping.put("groups", groups);
         */

        /*
        // Borland: Dependencies
        BorlandNode node1 = new BorlandNode("1", "Dependency");
        node1.addAttribute("Governor", "String");

        BorlandNode node2 = new BorlandNode("2", "Dependency");
        node2.addAttribute("Dependent", "String");

        BorlandEdge edge = new BorlandEdge("1", "2");

        BorlandMapping mapping = new BorlandMapping();
        mapping.addNode(node1);
        mapping.addNode(node2);
        mapping.addEdge(edge);
        */

        // Borland: NamedEntity & Token & POS
        BorlandNode node1 = new BorlandNode("1", "NamedEntity");
        node1.addAttribute("value", "String");
        node1.setIsUnique(true);

        BorlandNode node2 = new BorlandNode("2", "Token");
        node2.addAttribute("text", "String");

        BorlandNode node3 = new BorlandNode("3", "POS");
        node3.addAttribute("PosValue", "String");
        node3.addAttribute("coarseValue", "String");
        node3.setIsUnique(true);

        BorlandEdge edge1 = new BorlandEdge("1", "2");
        BorlandEdge edge2 = new BorlandEdge("2", "3");

        BorlandMapping mapping = new BorlandMapping();
        mapping.addNode(node1);
        mapping.addNode(node2);
        mapping.addNode(node3);
        mapping.addEdge(edge1);
        mapping.addEdge(edge2);


        pipeline.add(createEngineDescription(BorlandWriter.class,
                BorlandWriter.PARAM_MAPPING, mapping.getMapping(),
                BorlandWriter.PARAM_TARGET_LOCATION, "src/main/resources/borland_NE/",
                BorlandWriter.PARAM_OVERWRITE, true
        ));



        /*
        // GraphML: NamedEntity & Token
        GraphMLNode node1 = new GraphMLNode("1", "NamedEntity");
        node1.setLabel("value");
        node1.setIsUnique(true);

        GraphMLNode node2 = new GraphMLNode("2", "Token");
        node2.setLabel("text");

        GraphMLEdge edge = new GraphMLEdge("1", "2");

        GraphMLGroup group = new GraphMLGroup("3");
        group.addNodeToGroup("1");
        group.addNodeToGroup("2");

        GraphMLMapping mapping = new GraphMLMapping();
        mapping.addNode(node1);
        mapping.addNode(node2);
        mapping.addEdge(edge);
        mapping.addGroup(group);

        pipeline.add(createEngineDescription(GraphMLWriter.class,
                GraphMLWriter.PARAM_MAPPING, mapping.getMapping(),
                GraphMLWriter.PARAM_TARGET_LOCATION, "src/main/resources/graphMLMapping_NE/",
                GraphMLWriter.PARAM_OVERWRITE, true
        ));
         */

        SimplePipeline.runPipeline(reader, pipeline.createAggregateDescription());

    }

    /**
     * Function to test the reader
     *
     * @throws UIMAException
     * @throws IOException
     */
    public static void testReader() throws UIMAException, IOException {
        AggregateBuilder pipeline = new AggregateBuilder();

        CollectionReaderDescription reader = null;

        reader = createReaderDescription(ElasticsearchReader.class,
                ElasticsearchReader.PARAM_CONTAINER_NAME, "elasticsearchContainer",
                ElasticsearchReader.PARAM_INDEX, "data",
                ElasticsearchReader.PARAM_PASSWORD, "xGK6s7yjJkN6saI6Fzre");

        pipeline.add(createEngineDescription(SpaCyMultiTagger3.class,
                SpaCyMultiTagger3.PARAM_REST_ENDPOINT, "http://huaxal.hucompute.org:8103",
                SpaCyMultiTagger3.PARAM_MAX_TEXT_WINDOW, 90000));

        pipeline.add(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, "src/main/resources/elasticsearchDocker",
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true
        ));

        SimplePipeline.runPipeline(reader, pipeline.createAggregateDescription());

    }
}
