# PAG-GRPO

**PAG-GRPO: PPR-Guided Agentic Graph Retrieval with Trajectory-Level Group Relative Policy Optimization for Multi-Hop Question Answering**

PAG-GRPO is an agentic graph-retrieval framework for multi-hop question answering. It combines OpenIE-based graph construction, dense retrieval with NV-Embed-v2, LLM-guided triple filtering, Personalized PageRank over an entity-passage graph, iterative evidence memory, a trainable Llama-3 8B controller, trajectory-level Group Relative Policy Optimization (GRPO), and frozen evidence selection and grounded answer generation.

The framework is currently evaluated on **MuSiQue** multi-hop question answering.

> **Project status:** the frozen retrieval and finalization baseline is implemented and evaluated. The trajectory-level GRPO training code is implemented, and end-to-end integration validation is in progress.

---

## 1. Motivation

Multi-hop question answering requires more than retrieving passages independently. A system must repeatedly decide:

1. what information is currently missing,
2. which entities or relations should be explored,
3. whether the retrieved evidence is sufficient,
4. when to stop searching and answer.

PAG-GRPO treats retrieval as a sequential decision-making problem. A language-model controller interacts with a graph retrieval environment through a small action space and receives one reward for the quality of the complete trajectory.

The main design goal is to train the controller to retrieve evidence that is complete, relevant, non-redundant, grounded, and efficient to obtain.

---

## 2. High-Level Architecture

```text
Offline corpus processing
        |
        v
OpenIE extraction
        |
        v
Entity-passage-triple graph
        |
        v
NV-Embed-v2 indexes
  - passage embeddings
  - entity embeddings
  - triple embeddings

Online agent trajectory
        |
        v
Llama-3 8B + LoRA controller
        |
        +-----------------------------+
        |                             |
        v                             v
SearchGraph                    SubmitFinalAnswer
        |                             |
        v                             |
Query embedding                       |
        |                             |
        v                             |
Candidate triple retrieval             |
        |                             |
        v                             |
Frozen 70B triple filtering            |
        |                             |
        v                             |
PPR reset construction                 |
        |                             |
        v                             |
Personalized PageRank                  |
        |                             |
        v                             |
Retrieved passages and triples         |
        |                             |
        v                             |
Update EvidenceMemory -----------------+
        |
        v
Frozen 70B selector
        |
        v
Hybrid evidence fusion
        |
        v
Frozen grounded reader
        |
        v
Final answer and trajectory reward
```

---

## 3. Main Components

### 3.1 OpenIE Extraction

Documents are converted into passages and processed by an OpenIE extractor. Each extracted fact is represented as a triple:

\[
(s, r, o)
\]

where \(s\) is the subject, \(r\) is the relation or predicate, and \(o\) is the object.

Each triple retains its source passage identifier so that graph reasoning can be mapped back to textual evidence.

### 3.2 Entity-Passage Graph

The graph contains entity nodes, passage nodes, entity-entity relation edges, passage-entity context edges, and source-passage links for extracted triples.

The graph is used as the environment for Personalized PageRank.

### 3.3 Dense Embedding Index

Three embedding stores are constructed:

```text
chunk_embeddings/   passage vectors
entity_embeddings/  entity vectors
fact_embeddings/    triple/fact vectors
```

The current NV-Embed-v2 index uses:

```text
embedding dimension: 4096
dtype: float32
normalization: L2 normalized
```

During online retrieval, the stored vectors are reused. Only the current search query is embedded.

### 3.4 SearchGraph Retrieval

For every `SearchGraph` action, the system:

1. Builds a controlled retrieval query from the original question, current search focus, seed entities, relation hints, and evidence collected so far.
2. Embeds the query with NV-Embed-v2.
3. Retrieves top candidate triples by dense similarity.
4. Passes the candidates to the frozen Llama-3.3 70B triple filter.
5. Constructs a PPR reset vector from filtered-triple entities, controller-provided seed entities, source passages of selected triples, and dense passage retrieval scores.
6. Runs Personalized PageRank.
7. Returns ranked passages, candidate triples, filtered triples, and seed information.

Default retrieval settings:

| Parameter | Default |
|---|---:|
| Candidate triples | 40 |
| Dense reset top-k | 50 |
| Linking top-k | 50 |
| Returned passages | 10 |
| PPR damping | 0.5 |
| Passage reset weight | 0.05 |

### 3.5 EvidenceMemory

Each trajectory owns an independent `EvidenceMemory`.

After every search, the memory is updated with retrieved passages, candidate triples, filtered triples, entity and passage seeds, previous searches, compact evidence text, and search statistics.

At the next step, the controller sees a compact representation of this memory and decides whether to continue searching or submit an answer.

### 3.6 Controller Action Space

The trainable controller has only two valid actions:

```text
SearchGraph
SubmitFinalAnswer
```

A `SearchGraph` action can contain a search focus, seed entities, relation hints, requested triple top-k, and requested passage top-k.

`SubmitFinalAnswer` terminates the controller trajectory and starts frozen evidence finalization.

### 3.7 Frozen Finalization Pipeline

After the trajectory stops, the collected evidence is processed by:

1. **EvidenceSelectorV2** — a frozen 70B model selects the most useful passages.
2. **HybridEvidenceFuser** — preserves important high-PPR passages and adds selector-chosen passages.
3. **GroundedAnswerReader** — a frozen 70B reader generates the final answer from the fused evidence.

The selector, fuser, reader, PPR engine, embedding model, and triple filter remain frozen during controller training.

---

## 4. Trajectory-Level GRPO

The controller is a directly loaded **Llama-3 8B model with a trainable LoRA adapter**. No separate 8B vLLM controller is used during training.

For each question, the current policy samples \(G\) complete and independent trajectories:

\[
	au_i \sim \pi_{	heta_{\mathrm{old}}}, \qquad i=1,\ldots,G
\]

Each trajectory has its own evidence memory, action history, search count, retrieved evidence, final answer, and reward.

The default training group size is \(G=4\). The integration test initially uses \(G=2\).

---

## 5. Trajectory Reward

One scalar reward is computed after the complete trajectory finishes:

\[
egin{aligned}
R(	au)=
&\;2.0R_{\mathrm{F1}}
+1.0R_{\mathrm{Recall@5}}
+1.0R_{\mathrm{FullSupport@5}} \\
&+0.10R_{\mathrm{Format}}
+0.10R_{\mathrm{Novelty}} \\
&-0.05C_{\mathrm{Search}}
-0.15C_{\mathrm{Duplicate}} \\
&-0.10C_{\mathrm{ForcedStop}}
-0.10C_{\mathrm{Unknown}}.
\end{aligned}
\]

### Reward components

**Answer F1** measures token-level F1 between the grounded reader answer and the gold answer set.

**Support Recall@5** is:

\[
\mathrm{Recall@5}
=
rac{
|	ext{gold supporting passages} \cap 	ext{top-5 final evidence}|
}{
|	ext{gold supporting passages}|
}
\]

**FullSupport@5** is one only when all gold supporting passages appear in the top-five final evidence.

Gold and retrieved MuSiQue passages are matched using normalized title and passage text because dataset-local numeric identifiers and graph-global chunk identifiers are not directly comparable.

**Format validity** rewards valid controller-generated JSON actions.

**Evidence novelty** rewards searches that add new evidence instead of repeatedly returning the same passages.

Small penalties discourage unnecessary searches, duplicate searches, forced termination, and unsupported unknown answers.

---

## 6. Group-Relative Advantage

For all trajectories sampled for the same question:

\[
ar{R}
=
rac{1}{G}
\sum_{i=1}^{G} R_i
\]

\[
A_i
=
rac{R_i-ar{R}}
{\sigma_R+\epsilon}
\]

where:

\[
\epsilon=10^{-4}
\]

No learned value model or critic is required.

When all rewards in a group are equal, the group has zero variance and the policy advantages are zero. The trainer logs the zero-variance rate as a diagnostic.

---

## 7. Token-Level Clipped GRPO Loss

Only controller-generated JSON action tokens receive loss.

```text
system prompt          mask 0
question               mask 0
retrieved evidence     mask 0
environment output     mask 0
JSON action tokens     mask 1
```

For every generated action token:

\[
r_{i,t,k}(	heta)
=
\exp\left(
\log\pi_	heta(a_{i,t,k}\mid s_{i,t})
-
\log\pi_{\mathrm{old}}(a_{i,t,k}\mid s_{i,t})

ight)
\]

The clipped policy objective is:

\[
L_{\mathrm{policy}}
=
-\mathbb{E}
\left[
\min\left(
r_{i,t,k}A_i,\;
\operatorname{clip}
\left(
r_{i,t,k},
1-\epsilon_{\mathrm{low}},
1+\epsilon_{\mathrm{high}}

ight)A_i

ight)

ight]
\]

Current clipping parameters:

```text
epsilon_low  = 0.2
epsilon_high = 0.2
ratio range  = approximately [0.8, 1.2]
```

---

## 8. Reference KL Regularization

The original Llama-3 8B base policy is used as the frozen reference model. The same model instance is reused with the LoRA adapter disabled, avoiding a second full 8B copy in memory.

\[
L_{\mathrm{GRPO}}
=
L_{\mathrm{policy}}
+
eta L_{\mathrm{KL}}
\]

with:

\[
eta=0.01
\]

---

## 9. Trajectory-Mean Microbatching

Trajectories may contain different numbers of actions and action tokens. PAG-GRPO gives every trajectory equal top-level weight:

\[
L
=
rac{1}{G}
\sum_{i=1}^{G}
rac{
\sum_{t,k} m_{i,t,k}L_{i,t,k}
}{
\sum_{t,k}m_{i,t,k}
}
\]

During microbatching:

\[
	ext{scale}
=
rac{
	ext{active tokens in the microbatch}
}{
	ext{active tokens in the trajectory}
}
\cdot
rac{1}{G}
\]

The initial implementation processes one policy step per microbatch.

---

## 10. Optimization

### LoRA configuration

| Parameter | Value |
|---|---:|
| Rank \(r\) | 16 |
| Alpha | 32 |
| Dropout | 0.0 |
| Bias | none |
| Target modules | q, k, v, o, gate, up, down projections |

### Optimizer configuration

| Parameter | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | \(5	imes10^{-6}\) |
| \(eta_1\) | 0.9 |
| \(eta_2\) | 0.999 |
| Epsilon | \(10^{-8}\) |
| Weight decay | 0.0 |
| Gradient clipping | 1.0 |
| Scheduler | cosine |
| Warmup ratio | 0.05 |

Update sequence:

```text
GRPO loss
    |
    v
loss.backward()
    |
    v
clip global gradient norm to 1.0
    |
    v
AdamW optimizer.step()
    |
    v
cosine scheduler.step()
    |
    v
optimizer.zero_grad(set_to_none=True)
```

---

## 11. Current Baseline

The current integrated frozen baseline on 1,000 MuSiQue examples achieves:

| Metric | Score |
|---|---:|
| Exact Match | 0.3840 |
| Answer F1 | 0.5047 |
| Recall@5 | 0.8187 |
| FullSupport@5 | 0.6250 |
| Recall@10 | 0.8496 |
| FullSupport@10 | 0.6740 |

These results are from the frozen retrieval and finalization pipeline before GRPO controller training.

---

## 12. Repository Structure

```text
pag_grpo/
├── configs/
├── scripts/
│   ├── build_triple_index.py
│   ├── serve_nvembed.py
│   ├── test_grpo_rollout.py
│   ├── train_controller_grpo.py
│   └── ...
├── src/
│   └── ppr_agent/
│       ├── agent_env.py
│       ├── answer_reader.py
│       ├── embedding_store.py
│       ├── evidence_fusion.py
│       ├── evidence_selector.py
│       ├── graph_builder.py
│       ├── grpo_loss.py
│       ├── grpo_policy.py
│       ├── grpo_rollout.py
│       ├── grpo_types.py
│       ├── openie_extractor.py
│       ├── ppr_search.py
│       ├── reasoning_agent.py
│       ├── trajectory_reward.py
│       ├── triple_filter.py
│       ├── triple_index.py
│       └── ...
├── .gitignore
├── README.md
└── requirements.txt
```

Generated data, graph artifacts, embedding indexes, checkpoints, logs, and model weights are excluded from Git.

---

## 13. Main GRPO Files

| File | Responsibility |
|---|---|
| `grpo_types.py` | Action, trajectory, group, reward, and training-sample records |
| `trajectory_reward.py` | Answer, support, format, novelty, and efficiency rewards |
| `grpo_policy.py` | Direct 8B action sampling and old token-log-probability capture |
| `grpo_rollout.py` | Independent complete trajectory collection |
| `grpo_loss.py` | Clipped token-level GRPO objective and reference KL |
| `test_grpo_rollout.py` | End-to-end rollout, loss, backward, and memory test |
| `train_controller_grpo.py` | Main training loop, optimizer, scheduler, and checkpoints |

---

## 14. Environment Separation

Two environments are used to avoid incompatible model dependencies.

### `pag_rl`

Used for frozen services:

- Llama-3.3 70B vLLM server
- NV-Embed-v2 HTTP service
- older compatible Transformers/vLLM stack

### `pag_grpo`

Used for:

- directly loaded Llama-3 8B
- PEFT LoRA
- trajectory collection
- GRPO loss and backpropagation
- modern Transformers/PEFT training stack

The environments communicate through localhost HTTP services.

---

## 15. Example GPU Allocation

```text
GPU 0 + GPU 1:
    frozen Llama-3.3 70B vLLM server

GPU 1 spare memory or CPU:
    NV-Embed-v2 service

GPU 2:
    trainable Llama-3 8B + LoRA
```

---

## 16. Start the Frozen 70B Service

```bash
conda activate pag_rl
cd /home/ib5539/code/pag_ppr_grpo

export OPENAI_API_KEY=EMPTY

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
python -m vllm.entrypoints.openai.api_server \
  --model /home/ib5539/models/Llama-3.3-70B-Instruct \
  --served-model-name llama70b-filter \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 8192 \
  --port 8002
```

Test:

```bash
curl http://127.0.0.1:8002/v1/models
```

---

## 17. Start the NV-Embed Service

```bash
conda activate pag_rl
cd /home/ib5539/code/pag_ppr_grpo

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python scripts/serve_nvembed.py \
  --model_name nvidia/NV-Embed-v2 \
  --host 127.0.0.1 \
  --port 8003 \
  --device cuda:0 \
  --dtype float16 \
  --max_length 4096 \
  --expected_dimension 4096 \
  --trust_remote_code
```

CPU fallback:

```bash
CUDA_VISIBLE_DEVICES="" \
python scripts/serve_nvembed.py \
  --model_name nvidia/NV-Embed-v2 \
  --host 127.0.0.1 \
  --port 8003 \
  --device cpu \
  --dtype float32 \
  --max_length 4096 \
  --expected_dimension 4096 \
  --trust_remote_code
```

Test:

```bash
curl http://127.0.0.1:8003/health
```

---

## 18. GRPO Integration Test

```bash
conda activate pag_grpo
cd /home/ib5539/code/pag_ppr_grpo

export OPENAI_API_KEY=EMPTY

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python scripts/test_grpo_rollout.py \
  --questions_path data/raw/hipporag2/musique.json \
  --graph_dir outputs/ppr_agent/test_graph_llama70b \
  --index_dir outputs/ppr_agent/test_index_nvembed \
  --output_dir outputs/grpo_controller/integration_test \
  --model_name_or_path /home/ib5539/models/Meta-Llama-3-8B-Instruct \
  --policy_device cuda:0 \
  --torch_dtype bfloat16 \
  --attn_implementation sdpa \
  --embedding_backend remote \
  --embedding_base_url http://127.0.0.1:8003 \
  --embedding_timeout 300 \
  --triple_filter_backend openai \
  --triple_filter_model_name llama70b-filter \
  --triple_filter_base_url http://127.0.0.1:8002/v1 \
  --selector_model_name llama70b-filter \
  --selector_base_url http://127.0.0.1:8002/v1 \
  --answer_backend openai \
  --answer_model_name llama70b-filter \
  --answer_base_url http://127.0.0.1:8002/v1 \
  --group_size 2 \
  --start 0 \
  --max_prompt_tokens 2048 \
  --max_action_tokens 256 \
  --max_policy_steps_per_microbatch 1 \
  --trust_remote_code
```

By default, the integration test performs a backward pass but does not call `optimizer.step()`.

Use this only when deliberately testing one parameter update:

```bash
--optimizer_step
```

---

## 19. Main Training Flow

```text
Load Llama-3 8B base model
        +
attach trainable LoRA
        |
        v
Take one MuSiQue question
        |
        v
Sample G independent trajectories
        |
        v
Generate SearchGraph or SubmitFinalAnswer actions
        |
        v
Execute frozen retrieval and finalization environment
        |
        v
Compute one trajectory reward
        |
        v
Normalize rewards within the question group
        |
        v
Assign trajectory advantage to all action tokens
        |
        v
Compute clipped GRPO loss and reference KL
        |
        v
Backward pass
        |
        v
Gradient clipping
        |
        v
AdamW optimizer update
        |
        v
Learning-rate scheduler update
        |
        v
Generate new on-policy trajectories
```

---

## 20. Training Outputs

The trainer saves or logs:

- rollout trajectories,
- per-group rewards,
- advantages,
- zero-variance diagnostics,
- policy loss,
- KL loss,
- policy ratios,
- clipping fraction,
- gradient norm,
- learning rate,
- rollout time,
- GPU memory usage,
- LoRA checkpoints,
- optimizer state,
- scheduler state,
- random-number-generator state,
- training cursor.

---

## 21. Reproducibility Notes

- The retrieval environment is frozen during GRPO.
- Only the Llama-3 8B LoRA parameters are trainable.
- Every trajectory has an independent evidence memory.
- Old policy log probabilities are captured at rollout time.
- Loss is applied only to generated controller action tokens.
- The reference model is obtained by disabling LoRA on the base model.
- Existing NV-Embed-v2 indexes are reused and are not rebuilt during training.
- MuSiQue supporting passages are evaluated by normalized title/text content.

---

## 22. Current Development Status

Implemented:

- OpenIE graph construction
- embedding index construction
- candidate triple retrieval
- frozen 70B triple filtering
- PPR retrieval
- iterative evidence memory
- frozen selector, fusion, and reader
- trajectory reward
- group-relative advantages
- direct on-policy 8B rollout generation
- old-policy token-log-probability storage
- clipped GRPO loss
- reference KL
- LoRA backward and optimizer code
- integration-test script
- checkpoint and metric logging

In progress:

- complete remote NV-Embed integration validation
- successful end-to-end GRPO integration test
- multi-question training runs
- controlled ablations
- post-training baseline comparison

---

## 23. Planned Ablations

- without the 70B triple filter
- without direct source-passage reset
- without dense passage reset
- without PPR
- without evidence novelty reward
- without support rewards
- without search-cost penalties
- without reference KL
- greedy controller versus GRPO-trained controller
- different group sizes
- different maximum search budgets
- different evidence fusion strategies

---

## 24. Citation

This repository currently contains unpublished research code. A citation will be added after publication.

---

## 25. License

No public license has been assigned yet. Until a license is added, all rights are reserved by the repository owner.
