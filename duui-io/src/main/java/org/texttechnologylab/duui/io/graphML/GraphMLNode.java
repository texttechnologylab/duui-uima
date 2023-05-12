package org.texttechnologylab.duui.io.graphML;

import org.json.JSONObject;

/**
 * GraphMLNode
 * <p>
 * Node definition for GraphMLMapping
 *
 * @author Anna-Lena Buccoli
 */
public class GraphMLNode {

    String typeID;
    String nodeClass;
    String label;
    boolean labelFunction;
    boolean isUnique;
    String color;
    boolean colorFunction;

    /**
     * Creates a node
     *
     * @param typeID    typeID of the node
     * @param nodeClass name of the class for the node label
     */
    public GraphMLNode(String typeID, String nodeClass) {
        this.typeID = typeID;
        this.nodeClass = nodeClass;
        this.label = "";
        this.labelFunction = false;
        this.isUnique = false;
        this.color = "";
        this.colorFunction = false;
    }

    /**
     * Returns the node as JSONObject
     *
     * @return node as JSONObject
     */
    public JSONObject getNode() {
        return new JSONObject().put("typeID", typeID).put("class", nodeClass).put("label", label).put("labelFunction", labelFunction).put("isUnique", isUnique).put("color", color).put("colorFunction", colorFunction);
    }

    /**
     * Set a label for the node
     *
     * @param label name of the value for the node label
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
     * Set whether the the node should be unique
     *
     * @param isUnique whether each node label should be unique
     */
    public void setIsUnique(boolean isUnique) {
        this.isUnique = isUnique;
    }

    /**
     * Set the color of the node
     *
     * @param color color of the node
     */
    public void setColor(String color) {
        this.color = color;
    }

    /**
     * Set whether the the color should be defined by a function
     *
     * @param colorFunction whether the node color should be defined by a function
     */
    public void setColorFunction(boolean colorFunction) {
        this.colorFunction = colorFunction;
    }
}
