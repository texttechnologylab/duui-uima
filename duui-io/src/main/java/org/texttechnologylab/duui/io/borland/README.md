# BorlandWriter

The BorlandWriter creates a graph in Borland format from the data.
The result is written as a .bf file to the specified file path.
For this, a mapping must be defined, which is passed with the following parameter:

<table>
    <tbody>
        <tr>
            <td>PARAM_MAPPING</td>
            <td>Mapping to define the graph</td>
        </tr>
    </tbody>
</table>

It is also possible to visualize the data from several files in one graph.
For this, a file must be specified as the target location and the following parameter must be set:

<table>
    <tbody>
        <tr>
            <td>PARAM_MULTIPLE_DOCUMENTS</td>
            <td>Visualize the data from several files in one graph (default: false)</td>
        </tr>
    </tbody>
</table>

## Mapping

The mapping defines how the graph should look like.

A graph can consist of different node types that for example have different classes.
Each node type is defined by the node definition in the mapping and identified with a unique ID (typeID).
The edges of the graph are defined by the edge definitions in the mapping.
The edge definitions specify which node types are to be connected by an edge.
The related instances of the different node types are connected.

The mapping can be passed to the BorlandWriter as a string of a JSONObject or the BorlandMapping class.
Thereby nodes and edges are defined.
These can be passed directly to the constructor of the BorlandMapping class as JSONArrays.
Alternatively, node and edge definitions can be added with add functions as JSONObject or as instances of the classes
BorlandNode and BorlandEdge.
The mapping of the BorlandMapping class can then be retrieved as a string or JSONObject using the getter functions in
order to pass it to the BorlandWriter.

### Node definitions

Node definitions can be created using the BorlandNode class.
The constructor of the BorlandNode class requires the parameters typeID and class.
The remaining parameters can be set using the setter functions.
The getNode() function returns the node definition as a JSONObject.
Parameters that are not required do not need to be set (default: false, "" or []).
The following parameters can be defined for a node type:

<table>
    <tbody>
        <tr>
            <td>typeID</td>
            <td>ID of a node type</td>
        </tr>
        <tr>
            <td>class</td>
            <td>class of a node type</td>
        </tr>
        <tr>
            <td>attributes</td>
            <td>Definition of node attributes</td>
        </tr>
        <tr>
            <td>isUnique</td>
            <td>Whether to combine nodes with the same attribute</td>
        </tr>
    </tbody>
</table>

### Edge definitions

Edge definitions can be created using the BorlandEdge class.
The constructor of the BorlandEdge class requires the parameters source and target.
These specify between which node types an edge should be created.
The remaining parameters can be set using the setter functions.
The getEdge() function returns the edge definitions as JSONObject.
Parameters that are not needed do not have to be set (default: false, "" or []).
The following parameters can be defined for an edge type:

<table>
    <tbody>
        <tr>
            <td>source</td>
            <td>ID of a node type</td>
        </tr>
        <tr>
            <td>target</td>
            <td>ID of a node type</td>
        </tr>
        <tr>
            <td>class</td>
            <td>class for edge labeling</td>
        </tr>
        <tr>
            <td>attributes</td>
            <td>Definition of the edge attributes</td>
        </tr>
    </tbody>
</table>

### Attributes

The parameter attributes consists of the following parameters for the node and edge definitions:

<table>
    <tbody>
        <tr>
            <td>value</td>
            <td>Name of the attribute (attribute belonging to the class; "text" shows the text of the corresponding element)</td>
        </tr>
        <tr>
            <td>type</td>
            <td>type of the attribute</td>
        </tr>
        <tr>
            <td>function</td>
            <td>Whether the attributes should be created with a function</td>
        </tr>
    </tbody>
</table>

### Example

Example of a mapping as JSONObject:

```
{"nodes":[
    {"typeID":"1","class":"Dependency","attributes":[
        {"value":"Governor","type":"String"}
        ]},
    {"typeID":"2","class":"Dependency","attributes":[
        {"value":"Dependent","type":"String"}
        ]}
    ],
"edges":[
    {"source":"1","target":"2","class":"Dependency","attributes":[
        {"value":"DependencyType","type":"String"}
        ]}
    ]
}
```

Example of a mapping with the BorlandMapping class:

```
BorlandNode node1 = new BorlandNode("1", "Dependency");
node1.addAttribute("Governor", "String");

BorlandNode node2 = new BorlandNode("2", "Dependency");
node2.addAttribute("Dependent", "String");

BorlandEdge edge = new BorlandEdge("1", "2");
edge.setEdgeClass("Dependency");
edge.addAttribute("DependencyType", "String");


BorlandMapping mapping = new BorlandMapping();
mapping.addNode(node1);
mapping.addNode(node2);
mapping.addEdge(edge);
```