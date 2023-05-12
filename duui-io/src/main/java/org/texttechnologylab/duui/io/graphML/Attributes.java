package org.texttechnologylab.duui.io.graphML;

import org.apache.uima.jcas.cas.TOP;

/**
 * Attributes
 * <p>
 * Class to define the node color or label by function
 *
 * @author Anna-Lena Buccoli
 */
public class Attributes {

    /**
     * Function to define the node label
     *
     * @param nodeClass class of the node
     * @param label     label of the node
     * @param elem      corresponding TOP element
     * @return label of the node
     */
    public static String getLabel(String nodeClass, String label, TOP elem) {

        return label;
    }

    /**
     * Function to define the node color
     *
     * @param nodeClass class of the node
     * @param label     label of the node
     * @param elem      corresponding TOP element
     * @return color of the node
     */
    public static String getColor(String nodeClass, String label, TOP elem) {

        return "#FFFFFF";
    }

}
