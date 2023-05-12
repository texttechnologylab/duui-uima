package org.texttechnologylab.duui.io.graphML;

import org.json.JSONObject;

/**
 * GraphMLEdge
 * <p>
 * Edge definition for GraphMLMapping
 *
 * @author Anna-Lena Buccoli
 */
public class GraphMLEdge {

    String source;
    String target;
    String edgeClass;
    String label;
    boolean labelFunction;
    boolean directed;

    /**
     * Creates a edge
     *
     * @param source ID of source node
     * @param target ID of target node
     */
    public GraphMLEdge(String source, String target) {
        this.source = source;
        this.target = target;
        this.edgeClass = "";
        this.label = "";
        this.labelFunction = false;
        this.directed = false;
    }

    /**
     * Returns the edge as JSONObject
     *
     * @return edge definition as JSONObject
     */
    public JSONObject getEdge() {
        return new JSONObject().put("source", source).put("target", target).put("class", edgeClass).put("label", label).put("labelFunction", labelFunction).put("directed", directed);
    }

    /**
     * Set a class for the edge
     *
     * @param edgeClass name of the class for the edge label
     */
    public void setEdgeClass(String edgeClass) {
        this.edgeClass = edgeClass;
    }

    /**
     * Set a label for the edge
     *
     * @param label name of the value for the edge label
     */
    public void setLabel(String label) {
        this.label = label;
    }

    /**
     * Set whether the the label should be defined by a function
     *
     * @param labelFunction whether the label should be defined by a function
     */
    public void setLabelFunction(boolean labelFunction) {
        this.labelFunction = labelFunction;
    }

    /**
     * Set whether the edge should be directed
     *
     * @param directed whether the edge should be directed
     */
    public void setDirected(boolean directed) {
        this.directed = directed;
    }
}
