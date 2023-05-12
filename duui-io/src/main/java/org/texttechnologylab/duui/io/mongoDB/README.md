# MongoDB Reader & Writer

## MongoDBReader

The MongoDBReader reads the data from a Mongo database.
When called, the following parameters are used to establish the connection to the database:

<table>
    <tbody>
        <tr>
            <td>PARAM_DB_USER</td>
            <td>Username</td>
        </tr>
        <tr>
            <td>PARAM_DB_PW</td>
            <td>Password</td>
        </tr>
        <tr>
            <td>PARAM_DB_HOST</td>
            <td>Host (default: "localhost")</td>
        </tr>
        <tr>
            <td>PARAM_DB_NAME</td>
            <td>Name of the database</td>
        </tr>
        <tr>
            <td>PARAM_DB_AUTH_SOURCE</td>
            <td>Name of the authentication database (default: "admin")</td>
        </tr>
        <tr>
            <td>PARAM_DB_PORT</td>
            <td>Port for database connection (default: "27017")</td>
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
    </tbody>
</table>

Because large files can exceed the maximum size of MongoDB, GridFS is used.
For this the following parameter must be defined:

<table>
    <tbody>
        <tr>
            <td>PARAM_GRIDFS_BUCKET</td>
            <td>Name of the GridFS bucket (default: "fs")</td>
        </tr>
    </tbody>
</table>

## MongoDBWriter

The MongoDBWriter writes the data into a Mongo database.
When called, the following parameters are used to establish the connection to the database:

<table>
    <tbody>
        <tr>
            <td>PARAM_DB_USER</td>
            <td>Username</td>
        </tr>
        <tr>
            <td>PARAM_DB_PW</td>
            <td>Password</td>
        </tr>
        <tr>
            <td>PARAM_DB_HOST</td>
            <td>Host (default: "localhost")</td>
        </tr>
        <tr>
            <td>PARAM_DB_NAME</td>
            <td>Name of the database</td>
        </tr>
        <tr>
            <td>PARAM_DB_AUTH_SOURCE</td>
            <td>Name of the authentication database (default: "admin")</td>
        </tr>
        <tr>
            <td>PARAM_DB_PORT</td>
            <td>Port for database connection (default: "27017")</td>
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
            <td>PARAM_DOCKER_IMAGE</td>
            <td>Name of a Docker Images (default: "mongo")</td>
        </tr>
    </tbody>
</table>

Because large files can exceed the maximum size of MongoDB, GridFS is used.
For this the following parameter must be defined:

<table>
    <tbody>
        <tr>
            <td>PARAM_GRIDFS_BUCKET</td>
            <td>Name of the GridFS bucket (default: "fs")</td>
        </tr>
    </tbody>
</table>