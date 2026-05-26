# DUUI Lua `process` Interface

This component implements the Lua `process` interface added in DUUI v1.5.0.

## Description

Instead of exposing two methods (`serialize` and `deserialize`) which are called just once from DUUI, the single `process` method exposed by this components Lua script offers greater flexibility. 
While `serialize` is given a CAS and an `OutputStream`, `process` receives the CAS and a `RequestHandler` object, which allows arbitrary API calls to the component service.
This enables conditional API calls or looping over batches of element extracted from the CAS to be processed by the component.

![](https://mermaid.ink/svg/pako:eNqNU91v2jAQ_1ese5hASyBhhRRrQurSSn3YR1X6tGUPJrlCpMSX2Q4aQ_zvsxNSIOu0-ck-3_0-fOc9pJQhcND4o0aZ4m0u1kqUiWR2idTkW2GQ3UhJRpicJIuprEiiNIlsk1678xeLtx9rwZapyivDWaUoRa0H8c3SY4-OSpt7IbMC1bBFKYgqRpWDEUWx48fAFhVbCZNuULd53fpMVlZzfc7zoNB_aLnYG3YnnblT3SnT6buUwXuyRp3kTBgxPGFcZvkO6OwBnogKq-LL8mkw3objI8bYYyvKdsO_OfgDoBdg71dqwY6-UHcaTnC9fKfK79u7N6Z6RG37o3HQk9MzZavPH7Wr2rNcGqYtUa1j-7DW1c7gt--NOXb4n_60DOxOKVLaubpF16IXCvbBQrl4rPBy7o7tR5m1m4te-q_NIGfdhGb4j0EGD9Yqz4AbVaMHJapSuCPsHUACZoMlJmARIcNnURcmgUQebFkl5FeisqtUVK83wJ9Foe2pruzwdD_qJaqsB1Qx1dIAv7qaNCDA9_ATeBhej8LZPIimQTiZBFHwzoOdDUfBaDqfzaL57DqaHjz41XAGI3vyQNSGljuZdiIwyw2pT-2_br734Te5nUsP?theme=dark)
