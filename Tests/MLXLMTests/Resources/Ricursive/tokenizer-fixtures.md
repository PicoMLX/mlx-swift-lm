Ripple tokenizer and chat-template fixtures come from
`ricursiveAI/Ripple-65B-Preview` at
`e27e171792f303d74c048184e8259ba663cfa0d5` (Apache-2.0).
`tokenizer-reference.json` records source hashes and generator versions.

The unmodified official `chat_template.jinja` supplies the complete tool and
reasoning format. Its `tokenizer_config.json` embeds a basic ChatML template,
so loading that configuration alone loses the tool and reasoning behavior.
`RicursiveTokenizer` renders the full format in the production factory.
JSON object keys are sorted because Swift dictionaries have no insertion order;
the reference passes recursively sorted objects to the unchanged HF template.

`tokenizer.json` is a reduced offline BPE fixture, not a deployable tokenizer.
It keeps the original IDs, added tokens, normalizer, pre-tokenizer and decoder,
all single-byte vocabulary entries, and every merge whose result occurs in a
pre-tokenized reference input. Merge order is unchanged. The generator verifies
its complete token IDs and decoded strings against the full official tokenizer
for all six upstream `ripple_format.py` probes and all nine rendered prompts.
This exercises digit grouping, case-insensitive contractions, whitespace, code,
Unicode, control tokens, tools, grouped results and reasoning history with the
real Swift BPE implementation, without a network dependency.

Regenerate with `transformers==4.57.6` and `tokenizers==0.22.2`:

```
python scripts/generate-ricursive-tokenizer-reference.py /path/to/pinned/tokenizer/files
```

`IntegrationTestingTests/RicursiveTokenizerTests` also tests the complete pinned
tokenizer (downloads tokenizer assets only). Set `RIPPLE_TOKENIZER_DIRECTORY` to
a local directory containing those full official files to run it without a
download. The factory test uses the independent tiny model weights alongside
the offline tokenizer and verifies preparation plus both official EOS IDs.
