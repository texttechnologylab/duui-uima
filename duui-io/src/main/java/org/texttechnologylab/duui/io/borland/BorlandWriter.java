package org.texttechnologylab.duui.io.borland;

import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.TOP;
import org.apache.uima.json.JsonCasSerializer;
import org.dkpro.core.api.io.JCasFileWriter_ImplBase;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.StringWriter;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;

/**
 * BorlandWriter
 * <p>
 * Converts the given data into the borland format
 *
 * @author Anna-Lena Buccoli
 */
public class BorlandWriter extends JCasFileWriter_ImplBase {

    /**
     * Mapping to define the graph
     */
    public static final String PARAM_MAPPING = "mapping";
    @ConfigurationParameter(name = PARAM_MAPPING)
    private String mapping;

    /**
     * Whether the data from multiple documents should be shown in one graph
     */
    public static final String PARAM_MULTIPLE_DOCUMENTS = "multipleDocuments";
    @ConfigurationParameter(name = PARAM_MULTIPLE_DOCUMENTS, mandatory = false, defaultValue = "false")
    private boolean multipleDocuments;

    FileOutputStream out = null;
    OutputStreamWriter doc = null;
    HashMap<String, HashMap<String, ArrayList<String>>> nodeData = new HashMap<>();
    ArrayList<Pair<HashMap<String, ArrayList<String>>, ArrayList<String>>> edgeData = new ArrayList<>();

    /**
     * Reads the information from a JCas and call the function to create a graph
     *
     * @param aJCas contains the data
     * @throws AnalysisEngineProcessException
     */
    @Override
    public void process(JCas aJCas) throws AnalysisEngineProcessException {

        // Create file
        if (!multipleDocuments) {
            AtomicReference<String> title = new AtomicReference<>("");
            JCasUtil.select(aJCas, DocumentMetaData.class).forEach(i -> {
                title.set(i.getDocumentTitle());
            });
            FileOutputStream out = null;
            OutputStreamWriter doc = null;
            try {
                doc = new OutputStreamWriter(getOutputStream(title.toString(), ".bf"));
                out = new FileOutputStream(getTargetLocation() + title + ".bf");
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            try {
                out = new FileOutputStream(getTargetLocation(), false);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        // Read and check Mapping
        JSONObject map = new JSONObject(mapping);
        map = checkInput(map);

        // Read data
        JsonCasSerializer xcs = new JsonCasSerializer();
        xcs.setOmit0Values(false);
        xcs.setJsonContext(JsonCasSerializer.JsonContextFormat.omitContext);
        StringWriter sw = new StringWriter();
        try {
            xcs.serialize(aJCas.getCas(), sw);
        } catch (IOException e) {
            e.printStackTrace();
        }

        JSONObject jsonData = new JSONObject(sw.toString());
        String sofa = jsonData.getJSONObject("_views").getJSONObject("_InitialView").getJSONArray("DocumentMetaData").getJSONObject(0).get("sofa").toString();
        String docText = jsonData.getJSONObject("_referenced_fss").getJSONObject(sofa).getString("sofaString");
        Collection<TOP> allElements = JCasUtil.selectAll(aJCas);

        StringBuilder nodeAttributes = new StringBuilder();
        StringBuilder edgeAttributes = new StringBuilder();
        StringBuilder nodedata = new StringBuilder();
        StringBuilder edgedata = new StringBuilder();
        if (!multipleDocuments) {
            nodeData = new HashMap<>();
            edgeData = new ArrayList<>();
        }

        // Read node and edge attribute definitions
        for (Object i : map.getJSONArray("nodes")) {
            for (Object j : ((JSONObject) i).getJSONArray("attributes")) {
                nodeAttributes.append("[" + ((JSONObject) j).getString("value") + "¤" + ((JSONObject) j).getString("type") + "];");
            }
        }

        for (Object i : map.getJSONArray("edges")) {
            for (Object j : ((JSONObject) i).getJSONArray("attributes")) {
                edgeAttributes.append("[" + ((JSONObject) j).getString("value") + "¤" + ((JSONObject) j).getString("type") + "];");
            }

            // Call the functions to select the data for the nodes and edges
            Pair<HashSet<Pair<HashMap<String, ArrayList<String>>, ArrayList<String>>>, HashMap<String, HashMap<String, ArrayList<String>>>> edgeNodeData = getData((JSONObject) i, jsonData, docText, map, allElements);
            edgeData.addAll(edgeNodeData.getLeft());
            nodeData.putAll(edgeNodeData.getRight());
        }

        // Create the nodes and edges
        for (String i : nodeData.keySet()) {
            nodedata.append(createNode(i, nodeData.get(i)));
        }

        for (Pair<HashMap<String, ArrayList<String>>, ArrayList<String>> i : edgeData) {
            edgedata.append(createEdge(i.getRight().get(0), i.getRight().get(1), i.getLeft()));
        }

        // Call the function to write the graph into a file
        createGraph(out, nodeAttributes.toString(), edgeAttributes.toString(), nodedata.toString(), edgedata.toString());

    }


    /**
     * Check the input and add missing values
     *
     * @param map input mapping
     * @return checked and complete mapping
     */
    public JSONObject checkInput(JSONObject map) {

        try {
            map.getJSONArray("nodes");
            map.getJSONArray("edges");
        } catch (JSONException e) {
            e.printStackTrace();
        }

        for (Object i : map.getJSONArray("nodes")) {
            JSONObject thisNode = (JSONObject) i;
            try {
                thisNode.get("class");
                thisNode.get("typeID");
            } catch (JSONException e) {
                e.printStackTrace();
            }
            try {
                thisNode.get("attributes");
            } catch (JSONException e) {
                thisNode.put("attributes", new JSONArray());
            }
            for (Object j : thisNode.getJSONArray("attributes")) {
                JSONObject thisAttribute = (JSONObject) j;
                try {
                    thisAttribute.get("value");
                    thisAttribute.get("type");
                } catch (JSONException e) {
                    e.printStackTrace();
                }
                try {
                    thisAttribute.get("function");
                } catch (JSONException e) {
                    thisAttribute.put("function", false);
                }
            }
            try {
                thisNode.get("isUnique");
            } catch (JSONException e) {
                thisNode.put("isUnique", false);
            }
        }

        for (Object i : map.getJSONArray("edges")) {
            JSONObject thisEdge = (JSONObject) i;
            try {
                thisEdge.get("source");
                thisEdge.get("target");
            } catch (JSONException e) {
                e.printStackTrace();
            }
            try {
                thisEdge.get("class");
            } catch (JSONException e) {
                thisEdge.put("class", "");
            }
            try {
                thisEdge.get("attributes");
            } catch (JSONException e) {
                thisEdge.put("attributes", new JSONArray());
            }
            for (Object j : thisEdge.getJSONArray("attributes")) {
                JSONObject thisAttribute = (JSONObject) j;
                try {
                    thisAttribute.get("value");
                    thisAttribute.get("type");
                } catch (JSONException e) {
                    e.printStackTrace();
                }
                try {
                    thisAttribute.get("function");
                } catch (JSONException e) {
                    thisAttribute.put("function", false);
                }
            }
        }

        return map;
    }


    /**
     * Get the ID of all objects of the class
     *
     * @param dataClass class of the node
     * @param jsonData  JSONObject with the given data
     * @return list with all IDs of the given class
     */
    public List<Object> getObjID(String dataClass, JSONObject jsonData) {

        List<Object> objId = new LinkedList<>();

        if (dataClass.equals("POS")) {
            for (String i : jsonData.getJSONObject("_views").getJSONObject("_InitialView").toMap().keySet())
                if (i.contains("POS")) {
                    objId.addAll(jsonData.getJSONObject("_views").getJSONObject("_InitialView").getJSONArray(i).toList());
                }
        } else if (dataClass.equals("Dependency")) {
            objId.addAll(jsonData.getJSONObject("_views").getJSONObject("_InitialView").getJSONArray("Dependency").toList());
            objId.addAll(jsonData.getJSONObject("_views").getJSONObject("_InitialView").getJSONArray("ROOT").toList());
        } else {
            objId = jsonData.getJSONObject("_views").getJSONObject("_InitialView").getJSONArray(dataClass).toList();
        }

        return objId;
    }


    /**
     * Get the attributes and id of the value
     *
     * @param attributes  all defined attributes
     * @param isUnique    whether each node label should be unique
     * @param jsonData    JSONObject with the given data
     * @param id          ID of the current node
     * @param docText     text of the document
     * @param classValue  class of the node
     * @param allElements all elements as list of TOP elements
     * @return HashMap with all attributes and id of the current node
     */
    public Pair<HashMap<String, ArrayList<String>>, String> getAttributesAndID(JSONArray attributes, boolean isUnique, JSONObject jsonData, Object id, String docText, String classValue, Collection<TOP> allElements) {

        HashMap<String, ArrayList<String>> attributeLabel = new HashMap<>();
        ArrayList<String> nodeIdList = new ArrayList<>();
        String nodeId = "";
        String tokId = "";
        String labelValue = "";

        if (attributes.length() == 0) {
            if (classValue.equals("Dependency") || classValue.equals("Root")) {
                tokId = jsonData.getJSONObject("_referenced_fss").getJSONObject(id.toString()).get(labelValue).toString();
                int begin = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(tokId).get("begin").toString());
                int end = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(tokId).get("end").toString());
                nodeIdList.add(docText.substring(begin, end));
            } else {
                int begin = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(id.toString()).get("begin").toString());
                int end = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(id.toString()).get("end").toString());
                nodeIdList.add(docText.substring(begin, end));
            }
        }

        for (Object i : attributes) {
            ArrayList<String> label = new ArrayList<>();
            labelValue = ((JSONObject) i).getString("value");
            if (labelValue.equals("text")) {
                int begin = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(id.toString()).get("begin").toString());
                int end = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(id.toString()).get("end").toString());
                label.add(docText.substring(begin, end));
                nodeIdList.add(docText.substring(begin, end));
            } else if (labelValue.equals("Governor") || labelValue.equals("Dependent")) {
                tokId = jsonData.getJSONObject("_referenced_fss").getJSONObject(id.toString()).get(labelValue).toString();
                int begin = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(tokId).get("begin").toString());
                int end = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(tokId).get("end").toString());
                label.add(docText.substring(begin, end));
                nodeIdList.add(docText.substring(begin, end));
            } else if (jsonData.getJSONObject("_referenced_fss").getJSONObject(id.toString()).has(labelValue)) {
                label.add(jsonData.getJSONObject("_referenced_fss").getJSONObject(id.toString()).get(labelValue).toString());
                nodeIdList.add(jsonData.getJSONObject("_referenced_fss").getJSONObject(id.toString()).get(labelValue).toString());
            }
            if (((JSONObject) i).getBoolean("function")) {
                String finalClass = classValue;
                String finalBeginTok = jsonData.getJSONObject("_referenced_fss").getJSONObject(id.toString()).get("begin").toString();
                TOP elem = (TOP) CollectionUtils.find(allElements, o -> o.toString().contains(finalClass) && o.toString().contains(finalBeginTok));
                try {
                    label = Attributes.getAttribute(classValue, ((JSONObject) i).getString("value"), elem);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            if (!label.isEmpty()) {
                attributeLabel.put(((JSONObject) i).get("value").toString(), label);
            }
        }

        // Assigns an ID by checking whether the ID should be unique according to the ID of the element or for the same label
        if (isUnique) {
            nodeId = nodeIdList.get(0);
        } else {
            if (labelValue.equals("Governor") || labelValue.equals("Dependent")) {
                nodeId = tokId;
            } else {
                nodeId = id.toString();
            }
            if (multipleDocuments) {
                String docId = jsonData.getJSONObject("_views").getJSONObject("_InitialView").getJSONArray("DocumentMetaData").getJSONObject(0).get("documentId").toString();
                nodeId += "_" + docId;
            }
        }

        return Pair.of(attributeLabel, nodeId);
    }


    /**
     * Select the data to create the edges and nodes
     *
     * @param edgeConfig  definition of the edges
     * @param jsonData    JSONObject with the given data
     * @param docText     text of the document
     * @param map         mapping for graph
     * @param allElements all elements as list of TOP elements
     * @return Pair with edge and node data
     */
    public Pair<HashSet<Pair<HashMap<String, ArrayList<String>>, ArrayList<String>>>, HashMap<String, HashMap<String, ArrayList<String>>>> getData(JSONObject edgeConfig, JSONObject jsonData, String docText, JSONObject map, Collection<TOP> allElements) {

        String node1Class = "";
        String node2Class = "";
        JSONArray node1Attributes = null;
        JSONArray node2Attributes = null;
        boolean isUnique1 = false;
        boolean isUnique2 = false;
        List<Object> objId1;
        List<Object> objId2;
        List<Object> objIdEdge = null;
        String sourceId = "";
        String targetId = "";
        HashMap<String, ArrayList<String>> label = new HashMap<>();
        Integer beginTok;
        Integer endTok;
        HashMap<String, ArrayList<String>> nodeLabel;
        boolean exist;
        Pair<HashMap<String, ArrayList<String>>, String> labelAndID;
        HashSet<Pair<HashMap<String, ArrayList<String>>, ArrayList<String>>> edgeData = new HashSet<>();
        HashMap<String, HashMap<String, ArrayList<String>>> nodeData = new HashMap<>();

        // Read the edge definition from the mapping
        String edgeClass = edgeConfig.getString("class");
        JSONArray edgeAttributes = edgeConfig.getJSONArray("attributes");

        // If an edge label is defined, get the IDs of the specified elements
        if (!edgeClass.equals("")) {
            objIdEdge = getObjID(edgeClass, jsonData);
        }

        // Read the node definition from the mapping for source and target of the edge
        String source = edgeConfig.getString("source");
        String target = edgeConfig.getString("target");

        for (Object i : map.getJSONArray("nodes")) {
            JSONObject thisNode = (JSONObject) i;
            if (thisNode.getString("typeID").equals(source)) {
                node1Class = thisNode.getString("class");
                isUnique1 = thisNode.getBoolean("isUnique");
                node1Attributes = thisNode.getJSONArray("attributes");
            }
            if (thisNode.getString("typeID").equals(target)) {
                node2Class = thisNode.getString("class");
                isUnique2 = thisNode.getBoolean("isUnique");
                node2Attributes = thisNode.getJSONArray("attributes");
            }
        }

        // Get the IDs of the source and target nodes
        objId1 = getObjID(node1Class, jsonData);
        objId2 = getObjID(node2Class, jsonData);

        // Find sourceID and targetID for the edge and the data of the corresponding nodes
        for (Object i : objId1) {

            // Select and add the data of a source node
            exist = false;
            beginTok = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(i.toString()).get("begin").toString());
            endTok = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(i.toString()).get("end").toString());
            labelAndID = getAttributesAndID(node1Attributes, isUnique1, jsonData, i, docText, node1Class, allElements);
            nodeLabel = labelAndID.getLeft();
            sourceId = labelAndID.getRight();
            nodeData.put(sourceId, nodeLabel);

            for (Object j : objId2) {
                exist = false;

                // Find the corresponding target nodes
                if (beginTok <= Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(j.toString()).get("begin").toString())
                        && endTok >= Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(j.toString()).get("end").toString())) {

                    // Select and add the data of a target node
                    labelAndID = getAttributesAndID(node2Attributes, isUnique2, jsonData, j, docText, node2Class, allElements);
                    nodeLabel = labelAndID.getLeft();
                    targetId = labelAndID.getRight();
                    nodeData.put(targetId, nodeLabel);
                    exist = true;
                }

                // If defined, find the edge label for the edge between current source and target nodes
                if (exist && objIdEdge != null) {
                    for (Object l : objIdEdge) {
                        if (beginTok <= Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(l.toString()).get("begin").toString())
                                && endTok >= Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(l.toString()).get("end").toString())) {
                            label = getAttributesAndID(edgeAttributes, true, jsonData, l, docText, edgeClass, allElements).getLeft();
                            exist = true;
                        }
                    }
                }

                // Add the edge data
                if (exist) {
                    ArrayList<String> edge = new ArrayList<>();
                    edge.add(sourceId);
                    edge.add(targetId);
                    edgeData.add(Pair.of(label, edge));
                }
            }
        }

        return Pair.of(edgeData, nodeData);
    }


    /**
     * Create nodes
     *
     * @param nodeId     ID of the current node
     * @param attributes attributes of the current node
     * @return node as String
     */
    public String createNode(String nodeId, HashMap<String, ArrayList<String>> attributes) {

        String data = "";
        StringBuilder node = new StringBuilder();
        node.append(nodeId + "¤");
        for (String i : attributes.keySet()) {
            data = attributes.get(i).toString().replace(", ", "¤").replace("[", "¤").replace("]", "");
            node.append("[" + i + data + "¤]¤");
        }
        node.append("\n");

        return node.toString();
    }


    /**
     * Create Edges
     *
     * @param sourceId   sourceID for the current edge
     * @param targetId   targetID for the current edge
     * @param attributes attributes of the current edge
     * @return edge as String
     */
    public String createEdge(String sourceId, String targetId, HashMap<String, ArrayList<String>> attributes) {

        String data = "";
        StringBuilder edge = new StringBuilder();
        edge.append(sourceId + "¤" + targetId + "¤1.0¤");
        for (String i : attributes.keySet()) {
            data = attributes.get(i).toString().replace(", ", "¤").replace("[", "¤").replace("]", "");
            edge.append("[" + i + data + "¤]¤");
        }
        edge.append("\n");

        return edge.toString();
    }


    /**
     * Function to save the graph
     *
     * @param out            defines where the output should be written
     * @param nodeAttributes node definition
     * @param edgeAttributes edge definition
     * @param nodeData       node data
     * @param edgeData       edge data
     */
    public void createGraph(FileOutputStream out, String nodeAttributes, String edgeAttributes, String nodeData, String edgeData) {

        try {

            String data = "directed\nSimilarityGraph\nVertex Attributes:" + nodeAttributes +
                    "\nEdge Attributes:" + edgeAttributes +
                    "\nProbabilityMassOfGraph: 0\n" +
                    "Vertices:\n" + nodeData +
                    "Edges:\n" + edgeData;

            out.write(data.getBytes());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
