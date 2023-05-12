# GraphMLWriter

The GraphMLWriter creates a graph in GraphML format from the data.
The result is written as a .graphml file to the specified file path.
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
It is also possible to define groupings in the mapping.
In this case, the connected instances of the node types specified in the mapping are combined into a grouping.

The mapping can be passed to the GraphMLWriter as a string of a JSONObject or the GraphMLMapping class.
Thereby nodes, edges and groupings are defined.
These can be passed directly to the constructor of the GraphMLMapping class as JSONArrays.
Alternatively, node, edge and grouping definitions can be added with add functions as JSONObject or as instances of the
classes GraphMLNode, GraphMLEdge and GraphMLGroup.
The mapping of the GraphMLMapping class can then be retrieved as a string or JSONObject using getter functions to pass
it to the GraphMLWriter.

### Node definitions

Node definitions can be created using the GraphMLNode class.
The constructor of the GraphMLNode class requires the parameters typeID and class.
The remaining parameters can be set using the setter functions.
The getNode() function returns the node definition as a JSONObject.
Parameters that are not required do not need to be set (default: false or "").
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
            <td>label</td>
            <td>label for the nodes (attribute associated with the class; "text" shows the text of the corresponding element)</td>
        </tr>
        <tr>
            <td>labelFunction</td>
            <td>Whether the node label should be defined with a function</td>
        </tr>
        <tr>
            <td>isUnique</td>
            <td>Whether nodes with the same label should be grouped together</td>
        </tr>
        <tr>
            <td>color</td>
            <td>color of the node type</td>
        </tr>
        <tr>
            <td>colorFunction</td>
            <td>Whether to define the color using a function</td>
        </tr>
    </tbody>
</table>

### Edge definitions

Edge definitions can be created using the GraphMLEdge class.
The constructor of the GraphMLEdge class requires the parameters source and target.
These specify between which node types an edge should be created.
The remaining parameters can be set using the setter functions.
The getEdge() function returns the edge definitions as JSONObject.
Parameters that are not needed do not have to be set (default: false or "").
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
            <td>label</td>
            <td>label for the edge type (attribute associated with the class; "text" shows the text of the corresponding element)</td>
        </tr>
        <tr>
            <td>labelFunction</td>
            <td>Whether to define the edge label with a function</td>
        </tr>
        <tr>
            <td>directed</td>
            <td>Whether edge should be directed (from source to target)</td>
        </tr>
    </tbody>
</table>

### Grouping definitions

Grouping definitions are optional and can be created using the GraphMLGroup class.
A grouping groups all related nodes of the node types specified in the grouping definition.
The constructor of the GraphMLGroup class requires the typID parameter.
The node types to be included in this grouping can be added using the add function.
The getGroup() function returns the grouping definitions as JSONObject.
By specifying the typeID of a grouping in another grouping definition, it is possible to create nested groupings.
Note that the typeID of a grouping must be greater than all typeIDs of the node types specified in the nodes list.
The following parameters must be defined for a grouping definition:

<table>
    <tbody>
        <tr>
            <td>typeID</td>
            <td>typeID of the grouping definition</td>
        </tr>
        <tr>
            <td>nodes</td>
            <td>List with the typeIDs of the node types of the grouping</td>
        </tr>
    </tbody>
</table>

### Example

Example of a mapping as JSONObject:

```
{"nodes":[
        {"typeID":"1","class":"NamedEntity","label":"value","isUnique":true},
        {"typeID":"2","class":"Token","label":"text"}
    ],
"edges":[
        {"source":"1","target":"2"}
    ],
"groups":[
        {"typeID":"3","nodes":["1","2"]}
    ]
}
```

Example of a mapping with the GraphMLMapping class:

```
GraphMLNode node1 = new GraphMLNode("1", "NamedEntity");
node1.setLabel("value");
node1.setIsUnique(true);

GraphMLNode node2 = new GraphMLNode("2", "Token");
node2.setLabel("text");

GraphMLEdge edge = new GraphMLEdge("1", "2");

GraphMLGroup group = new GraphMLGroup("3");
group.addNodeToGroup("1");
group.addNodeToGroup("2");

GraphMLMapping mapping = new GraphMLMapping();
mapping.addNode(node1);
mapping.addNode(node2);
mapping.addEdge(edge);
mapping.addGroup(group);
```