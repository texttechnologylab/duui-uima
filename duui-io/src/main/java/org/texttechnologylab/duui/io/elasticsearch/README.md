# Elasticsearch Reader & Writer

## ElasticsearchReader

The ElasticsearchReader reads data from Elasticsearch.
When called, the following parameters are used to connect to Elasticsearch:

<table>
    <tbody>
        <tr>
            <td>PARAM_INDEX</td>
            <td>Name of the index</td>
        </tr>
        <tr>
            <td>PARAM_USERNAME</td>
            <td>Username (default: "elastic")</td>
        </tr>
        <tr>
            <td>PARAM_PASSWORD</td>
            <td>Password</td>
        </tr>
        <tr>
            <td>PARAM_HOST</td>
            <td>Name of the host (default: "localhost")</td>
        </tr>
        <tr>
            <td>PARAM_PORT</td>
            <td>Portnumber (default: "9200")</td>
        </tr>
        <tr>
            <td>PARAM_SCHEME</td>
            <td>Scheme for Http connection (default: "https")</td>
        </tr>
        <tr>
            <td>PARAM_CONNECT_TIMEOUT</td>
            <td>ConnectTimeout in ms (default: "5000")</td>
        </tr>
        <tr>
            <td>PARAM_SOCKET_TIMEOUT</td>
            <td>SocketTimeout in ms (default: "60000")</td>
        </tr>
    </tbody>
</table>

In addition, it is possible to use Docker.
For this it is necessary to define the following parameter:

<table>
    <tbody>
        <tr>
            <td>PARAM_CONTAINER_NAME</td>
            <td>Name of the Docker container</td>
        </tr>
        <tr>
            <td>PARAM_DOCKER_TIMEOUT</td>
            <td>Time to wait to start the container in sec (default: "30")</td>
        </tr>
    </tbody>
</table>

## ElasticsearchWriter

The ElasticsearchWriter writes the data to Elasticsearch.
When called, the following parameters are used to connect to Elasticsearch:

<table>
    <tbody>
        <tr>
            <td>PARAM_INDEX</td>
            <td>Name of the index</td>
        </tr>
        <tr>
            <td>PARAM_USERNAME</td>
            <td>Username (default: "elastic")</td>
        </tr>
        <tr>
            <td>PARAM_PASSWORD</td>
            <td>Password</td>
        </tr>
        <tr>
            <td>PARAM_HOST</td>
            <td>Name of the host (default: "localhost")</td>
        </tr>
        <tr>
            <td>PARAM_PORT</td>
            <td>Portnumber (default: "9200")</td>
        </tr>
        <tr>
            <td>PARAM_SCHEME</td>
            <td>Scheme for Http connection (default: "https")</td>
        </tr>
        <tr>
            <td>PARAM_CONNECT_TIMEOUT</td>
            <td>ConnectTimeout in ms (default: "5000")</td>
        </tr>
        <tr>
            <td>PARAM_SOCKET_TIMEOUT</td>
            <td>SocketTimeout in ms (default: "60000")</td>
        </tr>
    </tbody>
</table>

In addition, it is possible to use Docker.
For this it is necessary to define the following parameter:

<table>
    <tbody>
        <tr>
            <td>PARAM_CONTAINER_NAME</td>
            <td>Name of the Docker container</td>
        </tr>
        <tr>
            <td>PARAM_DOCKER_TIMEOUT</td>
            <td>Time to wait to start the container in sec (default: "30")</td>
        </tr>
    </tbody>
</table>