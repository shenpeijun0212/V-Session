# Provenance and reconstruction boundary

## Upstream baseline

- Repository: `https://github.com/shenpeijun0212/V-Session`
- Archive commit comment: `7293d0bda7381a206867c2b59ffd698899a6fac6`
- Source archive SHA-256:
  `14f64ee3d66993c23f1f1d99bd777430e120255106819446975c2b99f27c04e8`

The cleanup does not rewrite archived data, top-level few-shot prompts, or reference
logs. Their baseline hashes are listed below so later changes can be detected.

The supplied upstream archive did not contain a software license. At the authors'
direction, this polished distribution adds the MIT License for the repository's
original software and documentation. Third-party artifacts retain their own terms.

The paper-derived additions were reconstructed from the local revised manuscript
source `sn-article.tex` with SHA-256
`d2da932aaa1cfce8f2edaccd8ccdf0ace0dd4af26c4db2b29391c73e09336779`.
That manuscript source is not bundled in this repository; update this identifier if
the reconstruction is regenerated from a later revision.

### Data

| File | SHA-256 |
|---|---|
| `data/GSM8K.json` | `3484cd9501a90a832cede49ca305caf686ab344e07e129c50320b54c92bbd5ba` |
| `data/GSM8K.jsonl` | `52b8b7cdfc29e44b104ccca763315bb12e088d3205aa351325ab9e9a919c9080` |
| `data/MATH.jsonl` | `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132` |

### Archived prompts

| File | SHA-256 |
|---|---|
| `prompt/base_8-shot.txt` | `1693d3df7399c7721a8df7f144bb27730a2cf348d2bd5170eefd96cd55608770` |
| `prompt/CoT_8-shot.txt` | `e965d4c2bb45234909cc571de41e4a1a4d18861f568778930edf83c2b1ff1446` |
| `prompt/PoT_8-shot.txt` | `91a7bbe763d48e7eada227339db6ba2ba2db1f1ec0422188b763e6d083074966` |
| `prompt/PS_8-shot.txt` | `04dae7eb741be06cba3533944cad6b888de4f2fdd3a1f27a50df689bcd68478e` |
| `prompt/ToT_8-shot.txt` | `0745eaa5314a242dfd01841545781550b5ea2922e906084f560c13f18de72d70` |
| `prompt/V-Session_8-shot.txt` | `45426cf5493c05fb2e322ed4bbf0b4a56dc71e81e60fb0737af52bee0b4e8d25` |
| `prompt/base_5-shot-math.txt` | `e078b3ecea78e327cece30fae64f3c94ee3e8421f76e9b40aebef81ad5f77472` |
| `prompt/CoT_5-shot-math.txt` | `c1d74ee2fdb03304584a9647f1a0c40cf9572f79325a9e9262e7f4b3ce46fc28` |
| `prompt/V-Session_5-shot-math.txt` | `03ba6af36da1c78f129fc80156681e643867d4a7544b6d48c9c000c069f5d06a` |

### Reference logs

| File | SHA-256 |
|---|---|
| `log/Qwen2.5-3B_V-Session_GSM8K1000.log` | `f87d75753bb891b03f8d47a376a4117f5184b045fb50ac916852eb145b2a81f4` |
| `log/Qwen2.5-3B_V-Session_MATH500.log` | `4b3082308594ca4a89522ed5a02cbd721f9b63c385f9c6616934c4a42fe0f5a5` |

## Paper-known Qwen3.5 main protocol

- Full GSM8K test (1,319) and MATH500 (500).
- Direct, CoT, PoT, Plan-and-Solve, and V-Session.
- Zero-shot instructions without worked examples.
- Pass@1 greedy decoding without sampling or beam search.
- V-Session has exactly Goal, Solution, Thinking, Reasoning, Result stages.
- Official calculation delimiters are Unicode `≪ ≫`.
- V-Session ends in `#### <number>` for GSM8K and
  `Final Answer: <answer>` for MATH500.

## Not recoverable from the manuscript/archive

The exact Qwen3.5 checkpoint type/revision, raw-vs-chat wrapper, whitespace,
`max_new_tokens`, stop strings, repetition penalty, dtype, seed, backend, batching,
dataset revision/order/hash, baseline answer parser, complete ablation wording,
bootstrap seed/count, and exact McNemar variant are not published. The new code
makes these choices visible and calls its templates/protocol reconstructed rather
than historical.

The RSQ aggregation equations and four sensitivity-weight presets are published and
implemented. Automatic RSQ judging is not reconstructed: the prose describes four
dimension scores, whereas the supplied appendix prompt figure requests one overall
score per response. The sampled item IDs, evaluator-level ratings, dimension anchors,
judge revisions/settings, fitted calibration parameters, and several reliability-
statistic conventions are unavailable. The RSQ utility therefore aggregates only
external ratings supplied by the user.

The paper's 8-shot trace-conversion prompt and validated converted trace dataset are
also absent. The validator and training wrapper can check supplied traces, but they
do not recreate missing experimental data.
