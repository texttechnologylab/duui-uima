package org.texttechnologylab.duui.io.borland;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * BorlandNode
 * Defines a node for BorlandMapping
 *
 * @author Anna-Lena Buccoli
 */
public class BorlandNode {

    String typeID;
    String nodeClass;
    JSONArray attributes;
    boolean isUnique;

    /**
     * Creates a new node
     *
     * @param typeID    typeID of the node
     * @param nodeClass name of the class for node attributes
     */
    public BorlandNode(String typeID, String nodeClass) {
        this.typeID = typeID;
        this.nodeClass = nodeClass;
        this.attributes = new JSONArray();
        this.isUnique = false;
    }

    /**
     * Returns the node as JSONObject
     *
     * @return node as JSONObject
     */
    public JSONObject getNode() {
        return new JSONObject().put("typeID", typeID).put("class", nodeClass).put("attributes", attributes).put("isUnique", isUnique);
    }

    /**
     * Add an attribute for the node
     *
     * @param value name of the attribute value
     * @param type  type of the attribute value
     */
    public void addAttribute(String value, String type) {
        this.attributes.put(new JSONObject().put("value", value).put("type", type).put("function", false));
    }

    /**
     * Add an attribute for the node
     *
     * @param value    name of the attribute value
     * @param type     type of the attribute value
     * @param function whether the attribute should be defined by a function
     */
    public void addAttribute(String value, String type, boolean function) {
        this.attributes.put(new JSONObject().put("value", value).put("type", type).put("function", function));
    }

    /**
     * Set whether the the node should be unique
     *
     * @param isUnique whether each node label should be unique
     */
    public void setIsUnique(boolean isUnique) {
        this.isUnique = isUnique;
    }
}
