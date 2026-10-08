# Java Docker Validation


`CohMetrixDockerValidationTest` is the Java end-to-end regression test for
`duui-coh-metrix`. It follows the general structure of `SpaCyMultiTest` and
uses frozen bilingual CAS files as input. This tests the production path
through Java, DUUI, the Lua communication layer, Docker, and the Python service
without rerunning spaCy for every test execution.

The test enables official DUUI component logging at `WARN`. Coh-Metrix
warnings and exceptions returned by DUUIlogger are therefore visible in the
Maven console, while the two normal `INFO` records for every successful request
remain suppressed. ANSI colors are disabled so redirected validation logs stay
machine-readable.

## Test workflow

1. JUnit starts one `DUUIComposer` pipeline with the image configured in
   `CohMetrixDockerValidationTest.DOCKER_IMAGE`.
2. A compressed CAS snapshot (`input.xmi.gz`) is loaded for each testcase. It
   contains the frozen spaCy, sentence, paragraph, morphology, dependency,
   noun-chunk, and vector annotations.
3. Existing Coh-Metrix `Index` annotations and their associated
   `AnnotatorMetaData` entries are removed.
4. The CAS is processed by the Docker image.
5. The newly generated index values are checked against the corresponding
   `expected.csv`. Results are resolved through both `labelTTLab` and
   `labelV3`; no dedicated label testcases are generated.

By default, the test runs without the licensed GermaNet XML data. If
`cohmetrix.germanet.path` is supplied, the test starts the same pinned image
with the specified directory mounted read-only and applies the dedicated
`germanet-expected.csv` expectation overlay. The standard fixtures and their
default expectations are not modified by this optional mode.

No separate spaCy or Syntok container is started. The frozen annotations ensure
that the test specifically exposes changes in Coh-Metrix, its Lua communication
layer, and the container service.

## Current scope

- 322 bilingual testcases
- 3,496 curated index expectations
- 2,613 numeric expectations
- 883 explicit `NaN` expectations under the input-sufficiency rule
- 0 empty or uncurated expectations
- 5 additional contract tests
- 3,501 JUnit tests in total
- 521 GermaNet-mount expectations covering the German Situation Model and
  German Word Information indices

The file
`src/test/resources/validation-bilingual/null-semantics-migration.csv`
documents expectations that were changed from a numeric value to `NaN` during
the None-semantics migration.

The optional GermaNet expectations are stored in:

```text
src/test/resources/validation-bilingual/germanet-expected.csv
```

They are keyed by suite, testcase, and index. They are loaded only when the
GermaNet mount is enabled. This allows the standard run to retain its
without-GermaNet `NaN` expectations while the mounted run checks the available
numeric GermaNet reference values. Rows without lexical coverage retain
`NaN`, following the same input-sufficiency rule as the default suite.

## Evaluation rules

- **Numeric expected value:** The result must be finite and within the stated
  tolerance. A calculated `0.0` remains a regular measurement.
- **`NaN`, `None`, or `null`:** The index cannot be computed from the available
  input. The UIMA CAS must contain `Double.NaN`.
- **Empty `expected_value`:** The test is aborted as uncurated. The current
  suite no longer contains any such values.
- **Known-issue marker:** If the notes contain `KNOWN_FAIL_*`,
  `KNOWN_ISSUE_*`, `known fail`, or `known issue`, only an actual mismatch is
  aborted. If the value matches the expectation, the assertion passes
  normally.

The five additional contract tests enforce the basic None rule and protect the
punctuation-only sentence handling introduced for Issue #272:

- A completely empty document produces no numeric Coh-Metrix results.
- Sentence- and paragraph-pair indices return `NaN` when only one sentence and
  one paragraph are available.
- An available comparison pair with no noun overlap still returns the genuine
  calculated value `0.0`.
- Adding a punctuation-only sentence to an otherwise valid annotated document
  does not invalidate coarse-POS, fine-POS, dependency, or Situation Model
  indices and does not change their word-based values.
- A document that contains only punctuation still returns `NaN` for those
  indices because no non-punctuation word denominator exists.

## Prerequisites

- Java 21
- Maven with the required plugins and dependencies
- a running Docker daemon
- the pinned Coh-Metrix image available in the local Docker image store
- the complete test resources under
  `src/test/resources/validation-bilingual`
- the DUUI Core revision pinned in `pom.xml`; it contains the component-log
  transport introduced after release `1.5.7`

The optional GermaNet run additionally requires a local directory containing
the licensed GermaNet XML files. The XML data must not be committed, copied to
`target/classes`, or included in the Docker image.

The test does not fetch Docker images automatically. The image named by
`CohMetrixDockerValidationTest.DOCKER_IMAGE` must already exist in the local
Docker image store before Maven starts the test. This is intentional: it keeps
offline runs deterministic and prevents a test from silently switching to a
different remote image.

The current test references
`docker.texttechnologylab.org/duui-coh-metrix:0.1.1`. The earlier build
verified on September 17, 2026 used the historical local tag
`duui-coh-metrix:review-fixes-20260909`, which Docker resolved to the following
digest:

```text
duui-coh-metrix@sha256:5d63902ad2d0d83921d62c22e0bfe387ef759709b24669ee51e6b1a6d9ad751b
```

The digest documents that earlier verification only. New verification logs
must record the digest resolved for `0.1.1`. `latest` must not be used for a
reproducible regression test.

## Preparing the Docker image

Choose exactly one of the following procedures.

### Testing an already published image

Pull the exact tag referenced by `DOCKER_IMAGE`:

```powershell
$imageTag = "docker.texttechnologylab.org/duui-coh-metrix:0.1.1"
docker pull $imageTag
docker image inspect $imageTag
```

This downloads the image once. Docker stores it locally and reuses the cached
layers in subsequent runs. Users do not build the image themselves in this
workflow.

### Testing the current local source revision

Build the image from the checked-out source and assign the exact tag expected
by the Java test:

```powershell
cd C:\PATH\TO\duui-uima\duui-coh-metrix

$imageTag = "docker.texttechnologylab.org/duui-coh-metrix:0.1.1"

docker build `
  --file .\src\main\docker\Dockerfile `
  --build-arg DUUI_COH_METRIX_ANNOTATOR_VERSION="0.1.1" `
  --tag $imageTag `
  .

docker image inspect $imageTag
```

Assigning a registry-style tag locally does not upload or publish the image.
It only makes the locally built image available under the name expected by the
test. This is the required workflow when validating uncommitted changes or a
release candidate that has not yet been published.

## Updating the test resources

The suite is stored under:

```text
src/test/resources/validation-bilingual/
```

Run `mvn clean` at least once after replacing the suite. This prevents removed
or renamed resources from an earlier run from remaining in
`target/test-classes`. The `target` directory is Maven build output and must not
be versioned as a source resource.

## Running the online test

PowerShell:

```powershell
cd C:\PATH\TO\YOUR\COH-METRIX

mvn clean -U `
  "-Dtest=CohMetrixDockerValidationTest" `
  test *>&1 |
  Tee-Object .\validation-online-coh-metrix.log
```

`-U` allows Maven to refresh metadata and retrieve missing dependencies. The
Docker image under test must already have been built from the current source
revision.

During failures, records forwarded by the Coh-Metrix container appear in this
same console output with severity and component/document source prefixes. For
startup failures or requests that never return, inspect the container log as
well because request-scoped logs can only reach Java together with an HTTP
response.

The expected successful summary after adding the Issue #272 regression tests
is:

```text
Tests run: 3501, Failures: 0, Errors: 0
BUILD SUCCESS
```

## Running the optional GermaNet test

The regular test above remains the default and requires no GermaNet
configuration. To validate the German resource-dependent indices as well,
provide the host directory containing the licensed GermaNet XML files:

```powershell
cd C:\PATH\TO\YOUR\COH-METRIX

mvn clean -U `
  "-Dtest=CohMetrixDockerValidationTest" `
  "-Dcohmetrix.germanet.path=C:\PATH\TO\GERMANET" `
  test *>&1 |
  Tee-Object .\validation-germanet-coh-metrix.log
```

The supplied path must be a directory and must contain at least one `.xml`
file. The test then:

1. selects an available localhost port;
2. starts the pinned Coh-Metrix image as a temporary Docker container;
3. mounts the GermaNet directory read-only at
   `/usr/src/app/src/main/resources/germanet`;
4. sets `DUUI_COH_METRIX_GERMANET_PATH` inside the container;
5. connects to the service through `DUUIRemoteDriver`;
6. applies `germanet-expected.csv` instead of the corresponding default
   expectations; and
7. removes the temporary container after the test, including after setup or
   assertion failures where cleanup is still possible.

The GermaNet XML files remain outside the project, Maven output, image, and
test resources. Only the independently calculated expected values and their
provenance notes are versioned.

The verified successful summary is:

```text
Tests run: 3501, Failures: 0, Errors: 0, Skipped: 5
BUILD SUCCESS
```

The five skipped assertions are the two documented Pyphen discrepancies and
three German `SMTEMP` discrepancies caused by the frozen spaCy annotations.
The four assertions that are skipped in the standard run solely because
GermaNet is unavailable become active and pass in the mounted run.

## Running the Maven-offline test

After the online run has made all Maven dependencies available:

```powershell
mvn -o `
  "-Dtest=CohMetrixDockerValidationTest" `
  test *>&1 |
  Tee-Object .\validation-offline-coh-metrix.log
```

`-o` prevents Maven network access. The Docker driver uses the image that is
already available locally. The WordNet resources `wordnet` and `omw-1.4` are
installed during the image build and do not need to be downloaded at runtime.

Maven offline mode is not the same as complete network isolation of the
container. An additional run with blocked container egress provides the
stricter proof that the service itself does not require runtime downloads.

The optional GermaNet mode can also be executed with Maven offline after all
dependencies and the pinned Docker image are available locally:

```powershell
mvn -o `
  "-Dtest=CohMetrixDockerValidationTest" `
  "-Dcohmetrix.germanet.path=C:\PATH\TO\GERMANET" `
  test *>&1 |
  Tee-Object .\validation-offline-germanet-coh-metrix.log
```

## Planned script automation

The planned project Bash script should preserve these same explicit steps:

1. verify that Docker is available;
2. select either the published-image or local-build workflow;
3. ensure that the exact `DOCKER_IMAGE` tag exists locally;
4. run the online Java validation when requested;
5. optionally run the Maven-offline validation afterwards;
6. optionally accept a GermaNet host path and forward it as
   `cohmetrix.germanet.path` without copying the licensed files;
7. retain the Maven and container logs and return a non-zero exit status on
   build or test failure.

The Java validation should remain optional in that script. Starting the normal
Coh-Metrix pipeline must not implicitly run the complete regression suite.

## Last verified status

The standard and optional GermaNet runs were verified on September 24, 2026
with:

```text
docker.texttechnologylab.org/duui-coh-metrix:0.1.1
sha256:7dab6c7016fbd6e7ff579b609661967f968d67cf1e01d909bcd72fa54d53a433
```

Standard run without GermaNet:

```text
Tests run: 3501, Failures: 0, Errors: 0, Skipped: 9
BUILD SUCCESS
```

Optional run with the licensed GermaNet XML directory mounted read-only:

```text
Tests run: 3501, Failures: 0, Errors: 0, Skipped: 5
BUILD SUCCESS
```

The nine skipped assertions in the standard run are known and documented
external discrepancies:

- 2 Pyphen resource discrepancies in English syllable counts
- 3 German `SMTEMP` discrepancies caused by frozen spaCy annotations
- 4 German GermaNet expectations for which the licensed resource is not
  included in the static test image

In the optional mounted run, the four GermaNet assertions are enabled and
pass. The dedicated overlay also validates the remaining German
resource-dependent Situation Model and Word Information values while
preserving the agreed `NaN` behavior for missing lexical coverage.
