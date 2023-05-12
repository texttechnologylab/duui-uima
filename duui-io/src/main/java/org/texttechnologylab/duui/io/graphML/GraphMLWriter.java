package org.texttechnologylab.duui.io.graphML;

import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
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
 * GraphML Writer with Mapping
 *
 * @author Anna-Lena Buccoli
 */
public class GraphMLWriter extends JCasFileWriter_ImplBase {

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

    int groupID = 0;
    FileOutputStream out = null;
    OutputStreamWriter doc = null;
    HashMap<String, ArrayList<String>> nodeData = new HashMap<>();
    ArrayList<ArrayList<String>> edgeData = new ArrayList<>();

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
                title.set(i.getDocumentId());
            });
            FileOutputStream out = null;
            OutputStreamWriter doc = null;
            try {
                doc = new OutputStreamWriter(getOutputStream(title.toString(), ".graphml"));
                out = new FileOutputStream(getTargetLocation() + title + ".graphml");
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

        StringBuilder data = new StringBuilder();
        if (!multipleDocuments) {
            nodeData = new HashMap<>();
            edgeData = new ArrayList<>();
        }

        // Call function to select all data for the nodes and edges
        for (Object i : map.getJSONArray("edges")) {
            Pair<HashSet<ArrayList<String>>, HashMap<String, ArrayList<String>>> edgeNodeData = getData((JSONObject) i, jsonData, docText, map, allElements);
            edgeData.addAll(edgeNodeData.getLeft());
            nodeData.putAll(edgeNodeData.getRight());
        }

        // If no groupings are defined, nodes and edges are created
        // If groupings are defined, the function to select all data for them is called and then the groupings, nodes and edges are created
        if (map.getJSONArray("groups").toString().equals("[]")) {
            for (String i : nodeData.keySet()) {
                data.append(createNode(i, nodeData.get(i).get(0), nodeData.get(i).get(1)));
            }

            for (ArrayList<String> i : edgeData) {
                data.append(createEdge(i.get(0), i.get(1), i.get(2), i.get(3)));
            }
        } else {
            Triple<HashMap<String, HashSet<String>>, HashMap<String, ArrayList<String>>, TreeMap<String, ArrayList<String>>> groups = groupGraph(nodeData, edgeData, map.getJSONArray("groups"));
            TreeMap<String, HashSet<String>> groupData = new TreeMap<>(groups.getLeft());
            nodeData = groups.getMiddle();
            TreeMap<String, ArrayList<String>> groupIDList = groups.getRight();
            ArrayList<String> groupId = new ArrayList<>();

            for (Object j : map.getJSONArray("groups")) {
                groupId.add(((JSONObject) j).get("typeID").toString());
            }

            for (String j : groupIDList.descendingKeySet()) {
                for (String k : groupIDList.get(j)) {
                    if (groupData.get(k).size() > 1) {
                        data.append(createGroup(k, groupData, groupId, nodeData));
                    }
                }
            }

            for (String i : nodeData.keySet()) {
                if (!groupId.contains(nodeData.get(i).get(3))) {
                    data.append(createNode(i, nodeData.get(i).get(0), nodeData.get(i).get(1)));
                }
            }

            for (ArrayList<String> i : edgeData) {
                data.append(createEdge(i.get(0), i.get(1), i.get(2), i.get(3)));
            }

        }

        // Call the function to write the graph into a file
        createGraph(out, data.toString());
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
        try {
            map.getJSONArray("groups");
        } catch (JSONException e) {
            map.put("groups", new JSONArray());
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
                thisNode.get("label");
            } catch (JSONException e) {
                thisNode.put("label", "");
            }
            try {
                thisNode.get("labelFunction");
            } catch (JSONException e) {
                thisNode.put("labelFunction", false);
            }
            try {
                thisNode.get("isUnique");
            } catch (JSONException e) {
                thisNode.put("isUnique", false);
            }
            try {
                thisNode.get("color");
            } catch (JSONException e) {
                thisNode.put("color", "");
            }
            try {
                thisNode.get("colorFunction");
            } catch (JSONException e) {
                thisNode.put("colorFunction", false);
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
                thisEdge.get("label");
            } catch (JSONException e) {
                thisEdge.put("label", "");
            }
            try {
                thisEdge.get("labelFunction");
            } catch (JSONException e) {
                thisEdge.put("labelFunction", false);
            }
            try {
                thisEdge.get("directed");
            } catch (JSONException e) {
                thisEdge.put("directed", false);
            }
        }

        for (Object i : map.getJSONArray("groups")) {
            JSONObject thisGroup = (JSONObject) i;
            try {
                thisGroup.get("typeID");
                thisGroup.get("nodes");
            } catch (JSONException e) {
                e.printStackTrace();
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
     * Get the label and id of the value
     *
     * @param labelValue name of the defined label
     * @param isUnique   whether each node label should be unique
     * @param jsonData   JSONObject with the given data
     * @param id         ID of the current node
     * @param docText    text of the document
     * @return label and ID of the current node
     */
    public Pair<String, String> getLabelAndID(String labelValue, boolean isUnique, JSONObject jsonData, Object id, String docText) {

        String label = "";
        String nodeId = "";
        String tokId = "";

        if (labelValue.equals("text")) {
            int begin = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(id.toString()).get("begin").toString());
            int end = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(id.toString()).get("end").toString());
            label = docText.substring(begin, end);
        } else if (labelValue.equals("Governor") || labelValue.equals("Dependent")) {
            tokId = jsonData.getJSONObject("_referenced_fss").getJSONObject(id.toString()).get(labelValue).toString();
            int begin = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(tokId).get("begin").toString());
            int end = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(tokId).get("end").toString());
            label = docText.substring(begin, end);
        } else if (jsonData.getJSONObject("_referenced_fss").getJSONObject(id.toString()).has(labelValue)) {
            label = jsonData.getJSONObject("_referenced_fss").getJSONObject(id.toString()).get(labelValue).toString();
        }

        // Assigns an ID by checking whether the ID should be unique according to the ID of the element or for the same label
        if (isUnique) {
            nodeId = label;
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

        return Pair.of(label, nodeId);
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
    public Pair<HashSet<ArrayList<String>>, HashMap<String, ArrayList<String>>> getData(JSONObject edgeConfig, JSONObject jsonData, String docText, JSONObject map, Collection<TOP> allElements) {

        String node1Class = "";
        String node2Class = "";
        String node1Label = "";
        String node2Label = "";
        String node1Color = "";
        String node2Color = "";
        boolean node1LabelFunction = false;
        boolean node2LabelFunction = false;
        boolean isUnique1 = false;
        boolean isUnique2 = false;
        boolean color1Function = false;
        boolean color2Function = false;
        List<Object> objId1;
        List<Object> objId2;
        List<Object> objIdEdge = null;
        String sourceId = "";
        String targetId = "";
        String label = "";
        Integer beginTok;
        Integer endTok;
        String nodeLabel;
        String thisNodeColor;
        boolean exist;
        TOP element = null;
        Pair<String, String> labelAndID;
        HashSet<ArrayList<String>> edgeData = new HashSet<>();
        HashMap<String, ArrayList<String>> nodeData = new HashMap();

        // Read the edge definition from the mapping
        String edgeClass = edgeConfig.getString("class");
        String edgeLabel = edgeConfig.getString("label");
        boolean edgeLabelFunction = edgeConfig.getBoolean("labelFunction");

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
                node1Label = thisNode.getString("label");
                node1LabelFunction = thisNode.getBoolean("labelFunction");
                isUnique1 = thisNode.getBoolean("isUnique");
                node1Color = thisNode.getString("color");
                color1Function = thisNode.getBoolean("colorFunction");

            }
            if (thisNode.getString("typeID").equals(target)) {
                node2Class = thisNode.getString("class");
                node2Label = thisNode.getString("label");
                node2LabelFunction = thisNode.getBoolean("labelFunction");
                isUnique2 = thisNode.getBoolean("isUnique");
                node2Color = thisNode.getString("color");
                color2Function = thisNode.getBoolean("colorFunction");
            }
        }

        // Get the IDs of the source and target nodes
        objId1 = getObjID(node1Class, jsonData);
        objId2 = getObjID(node2Class, jsonData);

        // Find sourceID and targetID for the edge and the data of the corresponding nodes
        for (Object i : objId1) {

            // Select the data of a source node
            ArrayList<String> data1 = new ArrayList<>();
            exist = false;
            labelAndID = getLabelAndID(node1Label, isUnique1, jsonData, i, docText);
            nodeLabel = labelAndID.getLeft();
            sourceId = labelAndID.getRight();
            beginTok = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(i.toString()).get("begin").toString());
            endTok = Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(i.toString()).get("end").toString());

            if (node1Color.equals("")) {
                thisNodeColor = "#FFFFFF";
            } else {
                thisNodeColor = node1Color;
            }

            if (color1Function || node1LabelFunction) {
                String finalNode1Class = node1Class;
                String finalBeginTok = beginTok.toString();
                element = (TOP) CollectionUtils.find(allElements, o -> o.toString().contains(finalNode1Class) && o.toString().contains(finalBeginTok));

                if (node1LabelFunction) {
                    try {
                        nodeLabel = Attributes.getLabel(node1Class, nodeLabel, element);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                if (color1Function) {
                    try {
                        thisNodeColor = Attributes.getColor(node1Class, nodeLabel, element);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }

            // Add the node data
            data1.add(nodeLabel);
            data1.add(thisNodeColor);
            data1.add(source);
            data1.add(null);
            data1.add(null);
            nodeData.put(sourceId, data1);

            for (Object j : objId2) {
                exist = false;

                // Find the corresponding target nodes
                if (((beginTok <= Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(j.toString()).get("begin").toString()))
                        && (endTok >= Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(j.toString()).get("end").toString())))) {

                    // Select the data of a target node
                    ArrayList<String> data2 = new ArrayList<>();
                    labelAndID = getLabelAndID(node2Label, isUnique2, jsonData, j, docText);
                    nodeLabel = labelAndID.getLeft();
                    targetId = labelAndID.getRight();

                    if (node2Color.equals("")) {
                        thisNodeColor = "#FFFFFF";
                    } else {
                        thisNodeColor = node2Color;
                    }

                    if (color2Function || node2LabelFunction) {
                        String finalNode2Class = node2Class;
                        String finalBeginTok = beginTok.toString();
                        element = (TOP) CollectionUtils.find(allElements, o -> o.toString().contains(finalNode2Class) && o.toString().contains(finalBeginTok));

                        if (node2LabelFunction) {
                            try {
                                nodeLabel = Attributes.getLabel(node1Class, nodeLabel, element);
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                        if (color2Function) {
                            try {
                                thisNodeColor = Attributes.getColor(node2Class, nodeLabel, element);
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    }

                    // Add the node data
                    data2.add(nodeLabel);
                    data2.add(thisNodeColor);
                    data2.add(target);
                    data2.add(null);
                    data2.add(null);
                    nodeData.put(targetId, data2);
                    exist = true;
                }

                // If defined, find the edge label for the edge between current source and target nodes
                if (exist && objIdEdge != null) {
                    for (Object l : objIdEdge) {
                        if ((beginTok <= Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(l.toString()).get("begin").toString()))
                                && (endTok >= Integer.parseInt(jsonData.getJSONObject("_referenced_fss").getJSONObject(l.toString()).get("end").toString()))) {
                            label = getLabelAndID(edgeLabel, true, jsonData, l, docText).getLeft();

                            if (edgeLabelFunction) {
                                String finalEdgeClass = edgeClass;
                                String finalBeginTok = beginTok.toString();
                                element = (TOP) CollectionUtils.find(allElements, o -> o.toString().contains(finalEdgeClass) && o.toString().contains(finalBeginTok));
                                try {
                                    label = Attributes.getLabel(edgeClass, label, element);
                                } catch (Exception e) {
                                    e.printStackTrace();
                                }
                            }
                        }
                    }
                }

                // Add the edge data
                if (exist) {
                    ArrayList<String> edge = new ArrayList<>();
                    edge.add(label);
                    edge.add(sourceId);
                    edge.add(targetId);
                    edge.add(edgeConfig.get("directed").toString());
                    edgeData.add(edge);
                }
            }
        }

        return Pair.of(edgeData, nodeData);
    }


    /**
     * Group data according to the given group definition
     *
     * @param nodeData data of the nodes
     * @param edgeData data of the edges
     * @param groups   given group definition
     * @return Triple with data of the groups, updated node data and IDs of the groups
     */
    public Triple<HashMap<String, HashSet<String>>, HashMap<String, ArrayList<String>>, TreeMap<String, ArrayList<String>>> groupGraph(HashMap<String, ArrayList<String>> nodeData, ArrayList<ArrayList<String>> edgeData, JSONArray groups) {

        int firstNodeIndex;
        int secondNodeIndex;
        HashMap<String, HashSet<String>> groupData = new HashMap<>();
        TreeMap<String, ArrayList<String>> groupIDList = new TreeMap<>();

        for (Object i : groups) {
            ArrayList<HashSet<String>> allGroups = new ArrayList<>();
            ArrayList<String> groupTypeIDs = new ArrayList<>();

            // Look up all node type IDs which should be grouped
            for (Object j : ((JSONObject) i).getJSONArray("nodes")) {
                groupTypeIDs.add(j.toString());
            }

            // Check for each edge whether the incident nodes should be grouped
            for (ArrayList<String> j : edgeData) {
                String firstNode = j.get(1);
                String secondNode = j.get(2);

                // Check whether the nodes should be grouped by checking their typeIDs
                if (((groupTypeIDs.contains(nodeData.get(firstNode).get(2))) || (groupTypeIDs.contains(nodeData.get(firstNode).get(3))))
                        && ((groupTypeIDs.contains(nodeData.get(secondNode).get(2))) || (groupTypeIDs.contains(nodeData.get(secondNode).get(3))))) {

                    if (groupTypeIDs.contains(nodeData.get(firstNode).get(3))) {
                        firstNode = nodeData.get(firstNode).get(4);
                    }
                    if (groupTypeIDs.contains(nodeData.get(secondNode).get(3))) {
                        secondNode = nodeData.get(secondNode).get(4);
                    }

                    firstNodeIndex = -1;
                    secondNodeIndex = -1;

                    for (Set<String> k : allGroups) {
                        if (k.contains(firstNode)) {
                            firstNodeIndex = allGroups.indexOf(k);
                        }
                        if (k.contains(secondNode)) {
                            secondNodeIndex = allGroups.indexOf(k);
                        }
                    }

                    // Insert the nodeIDs to the list of all groups
                    if (firstNodeIndex == -1 && secondNodeIndex == -1) {
                        HashSet<String> nodeIDs = new HashSet<>();
                        nodeIDs.add(firstNode);
                        nodeIDs.add(secondNode);
                        allGroups.add(nodeIDs);
                    } else if (firstNodeIndex >= 0 && secondNodeIndex == -1) {
                        allGroups.get(firstNodeIndex).add(secondNode);
                    } else if (firstNodeIndex == -1) {
                        allGroups.get(secondNodeIndex).add(firstNode);
                    } else {
                        if (firstNodeIndex != secondNodeIndex) {
                            allGroups.get(firstNodeIndex).addAll(allGroups.get(secondNodeIndex));
                            allGroups.remove(allGroups.get(secondNodeIndex));
                        }
                    }
                }
            }

            ArrayList<String> groupIDs = new ArrayList<>();

            // Define a groupID and add the groups to the list of all nodes and to the list of all groups
            for (HashSet<String> l : allGroups) {
                String groupId = "g" + ((JSONObject) i).get("typeID").toString() + groupID;
                groupID++;
                groupIDs.add(groupId);
                for (String n : l) {
                    nodeData.get(n).set(3, ((JSONObject) i).get("typeID").toString());
                    nodeData.get(n).set(4, groupId);
                }

                groupData.put(groupId, l);
                ArrayList<String> data = new ArrayList<>();
                data.add(null);
                data.add(null);
                data.add(((JSONObject) i).get("typeID").toString());
                data.add(((JSONObject) i).get("typeID").toString());
                data.add(groupId);

                nodeData.put(groupId, data);
            }
            groupIDList.put(((JSONObject) i).get("typeID").toString(), groupIDs);

        }

        return Triple.of(groupData, nodeData, groupIDList);
    }


    /**
     * Create groups
     *
     * @param groupId     ID of current group
     * @param groupData   data of all groups
     * @param groupTypeId typeIDs of the grouped data
     * @param nodeData    data of all nodes
     * @return groups as String
     */
    public String createGroup(String groupId, TreeMap<String, HashSet<String>> groupData, ArrayList<String> groupTypeId, HashMap<String, ArrayList<String>> nodeData) {

        StringBuilder data = new StringBuilder();

        for (String i : groupData.get(groupId)) {
            if (groupTypeId.contains(nodeData.get(i).get(2))) {
                if (groupData.get(i).size() > 1) {
                    data.append(createGroup(i, groupData, groupTypeId, nodeData));
                }
            } else {
                data.append(createNode(i, nodeData.get(i).get(0), nodeData.get(i).get(1)));
            }
        }

        String group = "<node id=\"" + groupId + "\">\n" +
                "<graph id=\"" + groupId + "\" edgedefault=\"undirected\">\n"
                + data.toString()
                + "</graph></node>\n";

        return group;
    }


    /**
     * Create nodes
     *
     * @param nodeId ID of the current node
     * @param label  label of the current node
     * @param color  color of the current node
     * @return node as String
     */
    public String createNode(String nodeId, String label, String color) {

        String node = "<node id=\"" + nodeId + "\"><data key=\"n\"><y:ShapeNode>" +
                "<y:Geometry x=\"0\" y=\"0\" width=\"100\" height=\"25\"/>" +
                "<y:Fill color=\"" + color + "\" transparent=\"false\"/>" +
                "<y:BorderStyle type=\"line\" width=\"1.0\" color=\"#000000\"/>" +
                "<y:NodeLabel>" + label + "</y:NodeLabel></y:ShapeNode></data></node>\n";

        return node;
    }


    /**
     * Create edges
     *
     * @param label    label of the edge
     * @param sourceId sourceID for the current edge
     * @param targetId targetID for the current edge
     * @param directed wether the edges should be directed
     * @return
     */
    public String createEdge(String label, String sourceId, String targetId, String directed) {

        String targetDir;

        if (directed.equals("true")) {
            targetDir = "standard";
        } else {
            targetDir = "none";
        }

        String edge = "<edge id=\"" + sourceId + targetId + "\" source=\"" + sourceId + "\" target=\"" + targetId + "\">" +
                "<data key=\"e\">" +
                "<y:PolyLineEdge>" +
                "<y:Arrows source=\"none\" target=\"" + targetDir + "\"/>" +
                "<y:LineStyle type=\"line\" width=\"1.0\" color=\"#000000\"/>" +
                "<y:EdgeLabel>" + label + "</y:EdgeLabel>" +
                "</y:PolyLineEdge>" +
                "</data></edge>\n";

        return edge;
    }


    /**
     * Function to save the graph
     *
     * @param out  defines where the output should be written
     * @param data all data as String
     */
    public void createGraph(FileOutputStream out, String data) {

        try {
            String header = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.1/graphml.xsd\" xmlns:y=\"http://www.yworks.com/xml/graphml\">\n";
            String key = "<key id=\"n\" for=\"node\" yfiles.type=\"nodegraphics\"/><key id=\"e\" for=\"edge\" yfiles.type=\"edgegraphics\"/>\n";
            String graph = "<graph id=\"G\" edgedefault=\"undirected\">\n";
            String graphEnd = "</graph></graphml>";

            out.write((header + key + graph + data + graphEnd).getBytes());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
