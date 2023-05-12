# JSON Reader & Writer

## JsonReader

The JsonReader reads the data of a given JSON file.
When called, the following parameter can be specified:

<table>
    <tbody>
        <tr>
            <td>PARAM_SOURCE_ENCODING</td>
            <td>Encoding (default: DEFAULT_ENCODING)</td>
        </tr>
    </tbody>
</table>

## JsonWriter

The JsonWriter writes the data to a JSON file.
The result is written as a .org.texttechnologylab.duui.io.json file to the given file path.
When called, the following parameters can be specified:

<table>
    <tbody>
        <tr>
            <td>PARAM_JSON_CONTEXT</td>
            <td>JsonContextFormat (default: "omitExpandedTypeNames")</td>
        </tr>
        <tr>
        <td>PARAM_PRETTY_PRINT</td>
        <td>Format output (default: true)</td>
        </tr>
    </tbody>
</table>