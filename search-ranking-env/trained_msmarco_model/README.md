---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:668
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- accuracy
- accuracy_threshold
- f1
- f1_threshold
- precision
- recall
- average_precision
model-index:
- name: CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2
  results:
  - task:
      type: cross-encoder-binary-classification
      name: Cross Encoder Binary Classification
    dataset:
      name: ms marco dev
      type: ms-marco-dev
    metrics:
    - type: accuracy
      value: 0.8721071863580999
      name: Accuracy
    - type: accuracy_threshold
      value: 5.709959030151367
      name: Accuracy Threshold
    - type: f1
      value: 0.3904109589041096
      name: F1
    - type: f1_threshold
      value: 2.3247265815734863
      name: F1 Threshold
    - type: precision
      value: 0.3048128342245989
      name: Precision
    - type: recall
      value: 0.5428571428571428
      name: Recall
    - type: average_precision
      value: 0.28925214613647393
      name: Average Precision
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['what is a diabetic kidney', 'One of the more common long-term complications of diabetes is diabetic renal disease (renal refers to the kidneys). Also known as diabetic nephropathy, this condition is a result of direct vascular abnormalities that accompany diabetes.'],
    ['can you take left hand lane to turn right on a dual carriageway roundabout', 'When turning right, approach the roundabout in any lane marked for turning right (usually the right-hand lane), indicating right. As you pass the exit before the one you want to take, signal left. If there are two lanes on your exit road then you will exit onto the right-hand lane. The left-hand lane traffic can also turn right. Any vehicle wanting to turn into the road at X would use the left hand lane. If there are no lanes marked on a one-way street and you want to turn right, make the right turn from the right-hand side of the road, as shown in diagram R below.'],
    ['what is a goods receipt', 'You have the following possibilities for goods receipt posting: Goods Receipt Handling With Reference to Inbound Delivery. If you use the decentralized Warehouse Management system or Handling Unit Management, the data necessary for creating the transfer orders is transferred from the inbound delivery.'],
    ['was ronald reagan a democrat', 'In his younger years, Ronald Reagan was a member of the Democratic Party and campaigned for Democratic candidates; however, his views grew more conservative over time, and in the early 1960s he officially became a Republican. In November 1984, Ronald Reagan was reelected in a landslide, defeating Walter Mondale and his running mate Geraldine Ferraro (1935-), the first female vice-presidential candidate from a major U.S. political party.'],
    ['example of involuntary muscle tissue is', 'Visceral muscle tissue, or smooth muscle, is tissue associated with the internal organs of the body, especially those in the abdominal cavity. There are three types of muscle in the body: skeletal, smooth, and cardiac. As with any muscle, the smooth, involuntary muscles of the visceral muscle tissue (which lines the blood vessels, stomach, digestive tract, and other internal organs) are composed of bundles of specialized cells capable of contraction and relaxation to create movement. If one... Click to read more below.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'what is a diabetic kidney',
    [
        'One of the more common long-term complications of diabetes is diabetic renal disease (renal refers to the kidneys). Also known as diabetic nephropathy, this condition is a result of direct vascular abnormalities that accompany diabetes.',
        'When turning right, approach the roundabout in any lane marked for turning right (usually the right-hand lane), indicating right. As you pass the exit before the one you want to take, signal left. If there are two lanes on your exit road then you will exit onto the right-hand lane. The left-hand lane traffic can also turn right. Any vehicle wanting to turn into the road at X would use the left hand lane. If there are no lanes marked on a one-way street and you want to turn right, make the right turn from the right-hand side of the road, as shown in diagram R below.',
        'You have the following possibilities for goods receipt posting: Goods Receipt Handling With Reference to Inbound Delivery. If you use the decentralized Warehouse Management system or Handling Unit Management, the data necessary for creating the transfer orders is transferred from the inbound delivery.',
        'In his younger years, Ronald Reagan was a member of the Democratic Party and campaigned for Democratic candidates; however, his views grew more conservative over time, and in the early 1960s he officially became a Republican. In November 1984, Ronald Reagan was reelected in a landslide, defeating Walter Mondale and his running mate Geraldine Ferraro (1935-), the first female vice-presidential candidate from a major U.S. political party.',
        'Visceral muscle tissue, or smooth muscle, is tissue associated with the internal organs of the body, especially those in the abdominal cavity. There are three types of muscle in the body: skeletal, smooth, and cardiac. As with any muscle, the smooth, involuntary muscles of the visceral muscle tissue (which lines the blood vessels, stomach, digestive tract, and other internal organs) are composed of bundles of specialized cells capable of contraction and relaxation to create movement. If one... Click to read more below.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Binary Classification

* Dataset: `ms-marco-dev`
* Evaluated with [<code>CEBinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator)

| Metric                | Value      |
|:----------------------|:-----------|
| accuracy              | 0.8721     |
| accuracy_threshold    | 5.71       |
| f1                    | 0.3904     |
| f1_threshold          | 2.3247     |
| precision             | 0.3048     |
| recall                | 0.5429     |
| **average_precision** | **0.2893** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 668 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 668 samples:
  |         | sentence_0                                                                                     | sentence_1                                                                                      | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                         | string                                                                                          | float                                                          |
  | details | <ul><li>min: 11 characters</li><li>mean: 34.36 characters</li><li>max: 76 characters</li></ul> | <ul><li>min: 88 characters</li><li>mean: 419.3 characters</li><li>max: 922 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.13</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                              | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:----------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>what is a diabetic kidney</code>                                                  | <code>One of the more common long-term complications of diabetes is diabetic renal disease (renal refers to the kidneys). Also known as diabetic nephropathy, this condition is a result of direct vascular abnormalities that accompany diabetes.</code>                                                                                                                                                                                                                                                                                                                                                | <code>0.0</code> |
  | <code>can you take left hand lane to turn right on a dual carriageway roundabout</code> | <code>When turning right, approach the roundabout in any lane marked for turning right (usually the right-hand lane), indicating right. As you pass the exit before the one you want to take, signal left. If there are two lanes on your exit road then you will exit onto the right-hand lane. The left-hand lane traffic can also turn right. Any vehicle wanting to turn into the road at X would use the left hand lane. If there are no lanes marked on a one-way street and you want to turn right, make the right turn from the right-hand side of the road, as shown in diagram R below.</code> | <code>0.0</code> |
  | <code>what is a goods receipt</code>                                                    | <code>You have the following possibilities for goods receipt posting: Goods Receipt Handling With Reference to Inbound Delivery. If you use the decentralized Warehouse Management system or Handling Unit Management, the data necessary for creating the transfer orders is transferred from the inbound delivery.</code>                                                                                                                                                                                                                                                                              | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 1
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 1
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | ms-marco-dev_average_precision |
|:-----:|:----:|:------------------------------:|
| -1    | -1   | 0.3058                         |
| 1.0   | 42   | 0.2893                         |


### Framework Versions
- Python: 3.14.3
- Sentence Transformers: 5.3.0
- Transformers: 5.5.0
- PyTorch: 2.11.0+cu130
- Accelerate: 1.13.0
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->