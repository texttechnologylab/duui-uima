package org.hucompute.textimager.uima.easability;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.ServerSocket;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.UUID;

/**
 * The Coh-Metrix image with GermaNet mounted read-only, for use as a remote component (the Docker driver cannot
 * mount volumes). GermaNet is licensed and not part of the image: the directory with its XML files is given by the
 * system property {@value #GERMANET_PATH_PROPERTY}. It must be readable by the Docker daemon (a network mount may
 * not be; copy it to a local directory then). Stops the container on {@link #close()}.
 */
public class GermanetCohMetrixContainer implements AutoCloseable {

    public static final String GERMANET_PATH_PROPERTY = "cohmetrix.germanet.path";
    private static final String GERMANET_CONTAINER_PATH = "/usr/src/app/src/main/resources/germanet";
    private static final Duration STARTUP_TIMEOUT = Duration.ofMinutes(10);

    private final String name = "duui-coh-metrix-germanet-" + UUID.randomUUID().toString().substring(0, 8);
    private final String url;

    public GermanetCohMetrixContainer(String image) throws IOException, InterruptedException {
        String germanet = System.getProperty(GERMANET_PATH_PROPERTY);
        if (germanet == null || !Files.isDirectory(Paths.get(germanet))) {
            throw new IllegalArgumentException("-D" + GERMANET_PATH_PROPERTY + "=<GermaNet XML directory> required");
        }
        Path source = Paths.get(germanet).toAbsolutePath().normalize();

        int port;
        try (ServerSocket socket = new ServerSocket(0)) {
            port = socket.getLocalPort();
        }
        docker(List.of("docker", "run", "--detach", "--rm", "--name", name,
                "--publish", "127.0.0.1:" + port + ":9714",
                "--mount", "type=bind,source=" + source + ",target=" + GERMANET_CONTAINER_PATH + ",readonly",
                "--env", "DUUI_COH_METRIX_GERMANET_PATH=" + GERMANET_CONTAINER_PATH,
                image));
        url = "http://127.0.0.1:" + port;
        try {
            waitUntilReady();
        } catch (IOException | InterruptedException | RuntimeException e) {
            close();
            throw e;
        }
    }

    /** URL for {@code DUUIRemoteDriver.Component}. */
    public String url() {
        return url;
    }

    @Override
    public void close() throws IOException, InterruptedException {
        docker(List.of("docker", "stop", name));
    }

    private static void docker(List<String> command) throws IOException, InterruptedException {
        Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
        String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        if (process.waitFor() != 0) {
            throw new IllegalStateException(String.join(" ", command) + " failed: " + output);
        }
    }

    /** The component loads GermaNet at startup, which takes a while. */
    private void waitUntilReady() throws IOException, InterruptedException {
        Instant deadline = Instant.now().plus(STARTUP_TIMEOUT);
        while (Instant.now().isBefore(deadline)) {
            try {
                HttpURLConnection connection = (HttpURLConnection) URI.create(url + "/v1/communication_layer")
                        .toURL().openConnection();
                if (connection.getResponseCode() == 200) {
                    return;
                }
            } catch (IOException ignored) {
                // not listening yet
            }
            Thread.sleep(2000);
        }
        throw new IOException("Coh-Metrix container not ready after " + STARTUP_TIMEOUT + ": " + url);
    }
}
