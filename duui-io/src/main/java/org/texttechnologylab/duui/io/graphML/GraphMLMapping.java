package org.texttechnologylab.duui.io.graphML;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Mapping
 * <p>
 * Mapping for GraphMLWriter
 *
 * @author Anna-Lena Buccoli
 */
public class GraphMLMapping {

    JSONArray nodes;
    JSONArray edges;
    JSONArray groups;

    /**
     * Creates a new mapping
     */
    public GraphMLMapping() {
        this.nodes = new JSONArray();
        this.edges = new JSONArray();
        this.groups = new JSONArray();
    }

    /**
     * Creates a new mapping with nodes and edges
     *
     * @param nodes JSONArray with defined nodes
     * @param edges JSONArray with defined edges
     */
    public GraphMLMapping(JSONArray nodes, JSONArray edges) {
        this.nodes = nodes;
        this.edges = edges;
        this.groups = new JSONArray();
    }

    /**
     * Creates a new mapping with nodes, edges and groups
     *
     * @param nodes  JSONArray with defined nodes
     * @param edges  JSONArray with defined edges
     * @param groups JSONArray with defined groups
     */
    public GraphMLMapping(JSONArray nodes, JSONArray edges, JSONArray groups) {
        this.nodes = nodes;
        this.edges = edges;
        this.groups = groups;
    }

    /**
     * Returns the mapping as String
     *
     * @return mapping as String
     */
    public String getMapping() {
        return new JSONObject().put("nodes", this.nodes).put("edges", this.edges).put("groups", this.groups).toString();
    }

    /**
     * Returns the mapping as JSONObject
     *
     * @return mapping as JSONObject
     */
    public JSONObject getMappingAsJson() {
        return new JSONObject().put("nodes", this.nodes).put("edges", this.edges).put("groups", this.groups);
    }

    /**
     * Add a node to the mapping
     *
     * @param node add a node definition given as JSONObject
     */
    public void addNode(JSONObject node) {
        this.nodes.put(node);
    }

    /**
     * Add a node to the mapping
     *
     * @param node add a node definition given as GraphMLNode
     */
    public void addNode(GraphMLNode node) {
        addNode(new JSONObject().put("typeID", node.typeID).put("class", node.nodeClass).put("label", node.label).put("labelFunction", node.labelFunction).put("isUnique", node.isUnique).put("color", node.color).put("colorFunction", node.colorFunction));
    }

    /**
     * Add an edge to the mapping
     *
     * @param edge add an edge definition given as JSONObject
     */
    public void addEdge(JSONObject edge) {
        this.edges.put(edge);
    }

    /**
     * Add an edge to the mapping
     *
     * @param edge add an edge definition given as GraphMLEdge
     */
    public void addEdge(GraphMLEdge edge) {
        addEdge(new JSONObject().put("source", edge.source).put("target", edge.target).put("class", edge.edgeClass).put("label", edge.label).put("labelFunction", edge.labelFunction).put("directed", edge.directed));
    }

    /**
     * Add a group to the mapping
     *
     * @param group add an group definition given as JSONObject
     */
    public void addGroup(JSONObject group) {
        this.groups.put(group);
    }

    /**
     * Add a group to the mapping
     *
     * @param group add an group definition given as GraphMLGroup
     */
    public void addGroup(GraphMLGroup group) {
        addGroup(new JSONObject().put("typeID", group.typeID).put("nodes", group.nodesID));
    }
}
