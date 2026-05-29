#### OpenAI Privacy Filter component for DUUI

OpenAI Privacy Filter: https://github.com/openai/privacy-filter

#### Input/Output:

input: Text in the Sofa. Optional selection offsets can be passed through Lua options.

output: structured redaction spans and redacted text

#### Output Shape:

Privacy Filter detects 8 privacy span categories:

- `account_number`
- `private_address`
- `private_email`
- `private_person`
- `private_phone`
- `private_url`
- `private_date`
- `secret`

The model emits BIOES token classes for these categories plus `O`, and the service turns the resulting spans into DUUI annotations and redacted text.

#### Parameter:

[optional] OPF redaction options such as `model`, `context_window_length`, `trim_whitespace`, `device`, `output_mode`, `decode_mode`, `discard_overlapping_predicted_spans`, `viterbi_calibration_path`, and selection offsets (`selection_begin` / `selection_end`).

#### Modes:

- `replacement`: default mode, replaces detected spans with a consistent placeholder.
- `pseudo`: kept as a stub / TODO mode and currently returns the input unchanged.
- `mode` is passed through Lua options.

#### Entry points:

- `src/main/docker/python/duui_opf.py`: new OPF entrypoint wrapper.
- `src/main/docker/python/duui_whisperx.py`: compatibility implementation file while the migration is in progress.
