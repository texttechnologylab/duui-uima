package org.texttechnologylab.duui.io.borland;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * BorlandEdge
 * <p>
 * Defines an edge for BorlandMapping
 *
 * @author Anna-Lena Buccoli
 */
public class BorlandEdge {

    String source;
    String target;
    String edgeClass;
    JSONArray attributes;

    /**
     * Creates a new edge
     *
     * @param source typeID of the source node
     * @param target typeID of the target node
     */
    public BorlandEdge(String source, String target) {
        this.source = source;
        this.target = target;
        this.edgeClass = "";
        this.attributes = new JSONArray();
    }

    /**
     * Returns the edge as JSONObject
     *
     * @return edge definition as JSONObject
     */
    public JSONObject getEdge() {
        return new JSONObject().put("source", source).put("target", target).put("class", edgeClass).put("attributes", attributes);
    }

    /**
     * Set a class for the edge
     *
     * @param edgeClass name of the class for the edge attributes
     */
    public void setEdgeClass(String edgeClass) {
        this.edgeClass = edgeClass;
    }

    /**
     * Add an attribute for the edge
     *
     * @param value name of the attribute value
     * @param type  type of the attribute value
     */
    public void addAttribute(String value, String type) {
        this.attributes.put(new JSONObject().put("value", value).put("type", type).put("function", false));
    }

    /**
     * Add an attribute for the edge
     *
     * @param value    name of the attribute value
     * @param type     type of the attribute value
     * @param function whether the attribute should be defined by a function
     */
    public void addAttribute(String value, String type, boolean function) {
        this.attributes.put(new JSONObject().put("value", value).put("type", type).put("function", function));
    }
}
