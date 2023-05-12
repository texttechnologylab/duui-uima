package org.texttechnologylab.duui.io.graphML;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * GraphMLGroup
 * <p>
 * Group definition for GraphMLMapping
 *
 * @author Anna-Lena Buccoli
 */
public class GraphMLGroup {

    String typeID;
    JSONArray nodesID;

    /**
     * Create a group
     *
     * @param typeID typeID of group
     */
    public GraphMLGroup(String typeID) {
        this.typeID = typeID;
        this.nodesID = new JSONArray();
    }

    /**
     * Returns the group as JSONObject
     *
     * @return group as JSONObject
     */
    public JSONObject getGroup() {
        return new JSONObject().put("typeID", typeID).put("nodes", nodesID);
    }

    /**
     * Add a node by typeID to the group
     *
     * @param nodeTypeID typeID of the node which should be added to the group
     */
    public void addNodeToGroup(String nodeTypeID) {
        this.nodesID.put(nodeTypeID);
    }

}
