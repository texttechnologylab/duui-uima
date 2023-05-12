package org.texttechnologylab.duui.io.borland;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * BorlandMapping
 * <p>
 * Defines a mapping for BorlandWriter
 *
 * @author Anna-Lena Buccoli
 */
public class BorlandMapping {

    JSONArray nodes;
    JSONArray edges;

    /**
     * Creates a new mapping
     */
    public BorlandMapping() {
        this.nodes = new JSONArray();
        this.edges = new JSONArray();
    }

    /**
     * Creates a new mapping with nodes and edges
     *
     * @param nodes JSONArray with defined nodes
     * @param edges JSONArray with defined edges
     */
    public BorlandMapping(JSONArray nodes, JSONArray edges) {
        this.nodes = nodes;
        this.edges = edges;
    }

    /**
     * Returns the mapping as String
     *
     * @return mapping as String
     */
    public String getMapping() {
        return new JSONObject().put("nodes", this.nodes).put("edges", this.edges).toString();
    }

    /**
     * Returns the mapping as JSONObject
     *
     * @return mapping as JSONObject
     */
    public JSONObject getMappingAsJson() {
        return new JSONObject().put("nodes", this.nodes).put("edges", this.edges);
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
     * @param node add a node definition given as BorlandNode
     */
    public void addNode(BorlandNode node) {
        addNode(new JSONObject().put("typeID", node.typeID).put("class", node.nodeClass).put("attributes", node.attributes).put("isUnique", node.isUnique));
    }

    /**
     * Add a edge to the mapping
     *
     * @param edge add an edge definition given as JSONObject
     */
    public void addEdge(JSONObject edge) {
        this.edges.put(edge);
    }

    /**
     * Add a edge to the mapping
     *
     * @param edge add an edge definition given as BorlandEdge
     */
    public void addEdge(BorlandEdge edge) {
        addEdge(new JSONObject().put("source", edge.source).put("target", edge.target).put("class", edge.edgeClass).put("attributes", edge.attributes));
    }

}
