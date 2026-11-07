These fixtures come from Ricursive Intelligence's Apache-2.0 Ripple-65B Preview
reference at revision `e27e171792f303d74c048184e8259ba663cfa0d5`.
Source hashes, dependency versions, and generation details are in the provenance
JSON files. The upstream license is in `ThirdPartyLicenses/Ricursive.txt`.

Regenerate the numerical fixtures with `scripts/generate-ricursive-reference.py`.
It downloads only the reference Python files and configuration and creates deterministic tiny
weights, with no dependency on the released 65B checkpoint. Install the Python
dependencies pinned in the script's docstring first.

Strict float32 numerical and cache checks run on CPU to avoid the GPU's default
TF32 matmul precision. Separate Metal checks cover the GPU. BF16 checks bound both
maximum and mean error; the numerical provenance records the difference between
PyTorch's own default and math attention kernels on this same fixture.

Regenerate the tokenizer fixtures with
`scripts/generate-ricursive-tokenizer-reference.py /path/to/official/tokenizer`.
That directory must contain the pinned `tokenizer.json`, `tokenizer_config.json`,
and `chat_template.jinja`. The reduced BPE keeps original token IDs, merge ordering,
and preprocessing rules, and the generator checks its outputs against the complete
tokenizer before writing it. JSON objects are sorted before rendering because Swift
dictionaries do not retain insertion order.

The integration project's `RicursiveTokenizerTests` checks the real Swift tokenizer
against both the offline fixture and the complete published tokenizer. Set
`RIPPLE_TOKENIZER_DIRECTORY` to a local directory containing the pinned tokenizer
assets to use existing downloads. Otherwise the test downloads only those assets.
