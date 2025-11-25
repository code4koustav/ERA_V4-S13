# Reverse engineering SmolLM2-135M from the config file 

### Config-Derived Parameters
- `architectures=["LlamaForCausalLM"]`, `model_type="llama"` ⇒ confirms decoder-only LLaMA block arrangement and informs which stock reference implementation to mirror.
- `hidden_size=576`, `num_hidden_layers=30` ⇒ define embedding width and depth; dividing 576 by `num_attention_heads=9` yields the per-head dimension (64) used throughout attention projections.
- `num_key_value_heads=3` ⇒ grouped-query ratio of 3:1; informs how many KV projections to instantiate and how often to repeat them via `repeat_interleave`.
- `intermediate_size=1536`, `hidden_act="silu"` ⇒ mandates a SwiGLU-style MLP (gate/up/down) with SiLU activation to reach the intermediate width before projecting back.
- `rms_norm_eps=1e-5`, `attention_bias=false`, `attention_dropout=0.0` ⇒ RMSNorm parameters and absence of attention bias/dropout implement the pre-norm, bias-free style.
- `max_position_embeddings=8192`, `rope_theta=100000`, `rope_interleaved=false` ⇒ configure rotary embeddings and maximum context length; since `rope_scaling=null`, no extrapolation logic is required.
- `vocab_size=49152`, `bos_token_id=0`, `eos_token_id=0`, `tie_word_embeddings=true` ⇒ fixes embedding matrix shape, special token IDs, and dictates tying the LM head to the embedding weights.
- `initializer_range=0.041666... (1/24)` and `torch_dtype="bfloat16"` ⇒ translate into weight initialization std and preferred checkpoint dtype.
- `use_cache=true`, `max_position_embeddings=8192`, `attention_dropout=0` ⇒ indicate inference-time KV caching is supported and no dropout layers should be wired in.

## Architecture
- Decoder-only transformer mirroring `LlamaForCausalLM` with 30 stacked decoder blocks, ~135M parameters, hidden size 576, and pre-norm RMS layers so residual streams stay numerically stable even under bfloat16 training.
- Each block: `RMSNorm → GroupedQueryAttention → residual → RMSNorm → SwiGLU MLP → residual`, yielding a clean separation between attention- and feed-forward sublayers without extra bias terms.
- Grouped-query attention uses 9 query heads and 3 key/value heads (ratio 3:1) so KV projections are reused across three queries, cutting memory bandwidth while keeping full 9-head expressiveness; rotary embeddings with `rope_theta=100000` are applied before scaling, enabling 8,192-token context without absolute position tables.
- Attention bias/dropout are disabled (set to 0) to minimize inference branching, while causal masking is enforced through `scaled_dot_product_attention` in PyTorch 2.6+, matching the native FlashAttention path when available.
- SwiGLU MLP consists of parallel `gate`/`up` projections to 1,536 dims, applies SiLU on the gate path, multiplies elementwise, then projects back down via `down`; this variant boosts expressivity with minimal parameter increase versus ReLU MLPs.
- Token embeddings tie directly to the LM head (`tie_word_embeddings=True`), and weights initialize with std `1/24`, ensuring logits share the same geometry as the embedding space; max position embeddings fixed at 8,192 and no tensor parallel shards so checkpoints remain single-file portable.


### Reverse-Engineering Flow
1. Start from `architectures`/`model_type` to pick the base template (LLaMA decoder block).
2. Read width/depth (`hidden_size`, `num_hidden_layers`) and derive head dimension using `num_attention_heads`.
3. Inspect `num_key_value_heads` to decide on grouped-query attention, then wire rotary embeddings using the `rope_*` values.
4. Configure the MLP using `intermediate_size` plus `hidden_act`, add RMSNorm with `rms_norm_eps`, and confirm dropout/bias flags remain disabled per config.
5. Finish by sizing embeddings/logits using `vocab_size`, tie them if requested, and initialize weights with `initializer_range` in the target dtype noted by `torch_dtype`.

## Training Procedure
- Data: a single plain-text corpus (`input.txt`) tokenized with `HuggingFaceTB/SmolLM2-135M` tokenizer, chunked into 32-token blocks and batched as 4 sequences per micro-batch; dataset trims/pads to `(block_size * batch_size + 1)` to guarantee aligned next-token targets.
- Loader: pre-materialized list of `(x, y)` tensors reused across steps to simplify resumption; each item provides `batch_size x block_size` inputs/labels.
- Optimizer: AdamW (`β₁=0.9`, `β₂=0.95`, weight decay 0.1) with learning rate warmup toward `6e-4` (floor `6e-5`), gradient clipping at `1.0`, and 16-step gradient accumulation so every optimization step sees 2,048 tokens.
- Checkpointing: training resumes from `checkpoint_step_5000.pt`, carrying over optimizer/mode state via `torch.load` plus `add_safe_globals` to reconstruct the dataclass config; checkpoints saved every 500 steps into `checkpoints/checkpoint_step_{N}.pt`.
- Execution: automatic mixed precision enabled whenever CUDA/MPS is available (bfloat16 on CUDA, float16 on MPS) to reduce memory without rewriting kernels; training loop runs a fixed `max_steps=50` on top of the resumed `global_step`.

## Loss
- Standard next-token cross-entropy computed on shifted logits/labels inside `SmolLM2ForCausalLM`; logits at position *t* predict token *t+1*, and padding positions use `ignore_index=-100`.
- Loss is divided by the gradient-accumulation factor before `backward()` so the effective scale matches a larger batch; progress bar reports the rescaled value (`loss * accumulation_steps`) for readability.
- Because embeddings are tied, the loss directly reflects token embedding quality, encouraging consistent geometry between input and output spaces.

Training: 100%|█████████▉| 4987/5000 [3:44:02<00:30,  2.36s/it, loss=0.2753]
Training: 100%|█████████▉| 4987/5000 [3:44:02<00:30,  2.36s/it, loss=0.3065]
Training: 100%|█████████▉| 4988/5000 [3:44:04<00:27,  2.32s/it, loss=0.3065]
Training: 100%|█████████▉| 4988/5000 [3:44:04<00:27,  2.32s/it, loss=0.2727]
Training: 100%|█████████▉| 4989/5000 [3:44:06<00:25,  2.33s/it, loss=0.2727]
Training: 100%|█████████▉| 4989/5000 [3:44:06<00:25,  2.33s/it, loss=0.2656]
Training: 100%|█████████▉| 4990/5000 [3:44:09<00:24,  2.45s/it, loss=0.2656]
Training: 100%|█████████▉| 4990/5000 [3:44:09<00:24,  2.45s/it, loss=0.4380]
Training: 100%|█████████▉| 4991/5000 [3:44:11<00:21,  2.39s/it, loss=0.4380]
Training: 100%|█████████▉| 4991/5000 [3:44:11<00:21,  2.39s/it, loss=0.3784]
Training: 100%|█████████▉| 4992/5000 [3:44:14<00:18,  2.35s/it, loss=0.3784]
Training: 100%|█████████▉| 4992/5000 [3:44:14<00:18,  2.35s/it, loss=0.1973]
Training: 100%|█████████▉| 4993/5000 [3:44:16<00:16,  2.32s/it, loss=0.1973]
Training: 100%|█████████▉| 4993/5000 [3:44:16<00:16,  2.32s/it, loss=0.2702]
Training: 100%|█████████▉| 4994/5000 [3:44:18<00:13,  2.31s/it, loss=0.2702]
Training: 100%|█████████▉| 4994/5000 [3:44:18<00:13,  2.31s/it, loss=0.2870]
Training: 100%|█████████▉| 4995/5000 [3:44:21<00:12,  2.46s/it, loss=0.2870]
Training: 100%|█████████▉| 4995/5000 [3:44:21<00:12,  2.46s/it, loss=0.4291]
Training: 100%|█████████▉| 4996/5000 [3:44:23<00:09,  2.39s/it, loss=0.4291]
Training: 100%|█████████▉| 4996/5000 [3:44:23<00:09,  2.39s/it, loss=0.3940]
Training: 100%|█████████▉| 4997/5000 [3:44:25<00:07,  2.36s/it, loss=0.3940]
Training: 100%|█████████▉| 4997/5000 [3:44:25<00:07,  2.36s/it, loss=0.3753]
Training: 100%|█████████▉| 4998/5000 [3:44:28<00:04,  2.33s/it, loss=0.3753]
Training: 100%|█████████▉| 4998/5000 [3:44:28<00:04,  2.33s/it, loss=0.3150]
Training: 100%|█████████▉| 4999/5000 [3:44:30<00:02,  2.33s/it, loss=0.3150]
Training: 100%|█████████▉| 4999/5000 [3:44:30<00:02,  2.33s/it, loss=0.2092]
Training: 100%|██████████| 5000/5000 [3:44:33<00:00,  2.48s/it, loss=0.2092]
Training: 100%|██████████| 5000/5000 [3:44:33<00:00,  2.48s/it, loss=0.3623]


## sample output after every 50 step

--- Sample Generation at Step 5000 ---
Prompt: 'The '
The  find find with and power love.
First:F
 told lie mother good; would my cannot ac.
A for! amen this ca indeed--
SIN EDARD
 mine--
ist it,?,iltent away
 shallow bid,,,ost knowilyatter,WhenHe coming thy heart thy by,
 told my and are, I-- you I?
ent of.
First,LEES
's resolve on my,am,lab: I lean
Prompt: 'In '
In , p, I atThey a manal.
MARUSY
Tor
How yout
 division I to piece sore you sir
 with one, to but is end way
 I thy and b you Hortio
 accuse daily
It bearsray in earnest a he done?
MARUS
SINENT:I not the it;We beenble the in part?
SINUS
 have ro thousandthough mountain: is I, back one uncle in a.

Prompt: 'To '
To  alone so yet welcome I: was own
 of power what is, mistake or accuse
 hell person so, never upon head in
 his
 of, all shall him hisers,
 cannot it the did out a day neer,That I accuse
 most all ears hate was first meught more
an till little
 a e I, death I, a to to him your father
all tyrann gone he, will I, think' to a-orrow a to me- there comfort
Prompt: 'A '
A  is and and power deliver ruledUpon
 hardable to against., how must so
 comm, to; what I, the here cheer you,G many heartOfhim
 an where:o little king stayWith to
's'd to against! not by to! else this to!A grave as?
Second spoke fault
, wit him home.
MENI: not of are gift by,am, sweetly, let be known will the of,That are long take
