# Training Primer

What happens inside `rxnorm_vandf/train.py`, in the order it happens. For the
W&B side see [wandb-primer.md](wandb-primer.md); for the results, the
[write-up](post/README.md).

## Embeddings and cosine similarity

The model turns a string into a vector (an *embedding*: 384 numbers for MiniLM,
768 for SapBERT). I scale every vector to length 1, so the dot product of two of them is the cosine
of the angle between them: 1.0 for identical direction, 0 for unrelated. To map a
VA string I embed it, embed all 27,287 candidate names (once), and take the
candidates with the highest cosine. That is what `retrieve()` does.

The first base model, `all-MiniLM-L6-v2`, already produces sensible embeddings for
general English (22M parameters, trained on a billion sentence pairs). It knows
nothing about "TAB,SA" meaning "Extended Release Oral Tablet." Fine-tuning teaches it.
The final model starts from SapBERT (110M parameters), which was pre-trained on
UMLS synonym pairs and so already knows that `HCL` and `hydrochloride` are the same
thing; everything below applies to both.

## Contrastive training

I never tell the model "the answer is candidate #8172." I show it pairs and
ask it to make the right ones close and the wrong ones far. Each training row is a
**triplet**:

| anchor | positive | negative |
|---|---|---|
| `METOPROLOL TARTRATE 12.5MG TAB` | `metoprolol tartrate 12.5 MG Oral Tablet` | `metoprolol tartrate 25 MG Oral Tablet` |

The loss (`MultipleNegativesRankingLoss`) works on a whole batch at once. For a
batch of 64 rows, each anchor is scored against 128 candidates: its own positive,
its own negative, and the 63 positives and 63 negatives belonging to the *other*
rows. The loss is cross-entropy over those 128 scores with the true positive as the
answer: it goes down when the correct name outscores every other one in the batch.

- **In-batch negatives** are the other rows' names. They're free and mostly easy
  (a random other drug).
- **Hard negatives** are the third column. I choose a product with the *same
  ingredients* but a different strength or dose form: exactly the mistakes TF-IDF
  made. That's the `negatives=ingredient` setting. `tfidf` uses the top wrong
  TF-IDF hit instead; `none` drops the column and relies on in-batch negatives only.
  The sweep compared the three: ingredient-matched negatives were worth about 3
  points of validation accuracy; TF-IDF-mined ones were no better than none.
- **`NO_DUPLICATES` batch sampler:** if two rows in a batch had the same positive
  (two VA strings for one product), each would be told the other's positive is
  wrong. The sampler prevents that.
- Hard negatives are drawn **only from train-split candidates**. A val/test drug's
  name never appears in training, even as a wrong answer.

## The knobs

| Knob | Default | What it does |
|---|---|---|
| **epoch** | 4 | One pass over all 9,287 training rows. More epochs = more learning, and eventually memorizing. |
| **batch size** | 64 | Rows per update. Larger batches mean more in-batch negatives (better for this loss) but more GPU memory. 64 fits the 1070 easily. |
| **learning rate** | 2e-5 | How far each update moves the weights. Too high and training diverges or wrecks what the base model knew; too low and it barely changes. 2e-5 is the usual fine-tuning default for BERT-sized models. |
| **warmup ratio** | 0.1 | The learning rate ramps up from 0 over the first 10% of steps, then decays linearly to 0. Warmup avoids large, random early updates. |
| **max_seq_length** | 96 | Tokens per string. VA strings are ≤ 64 characters; candidate names top out around 220 characters, which is under 96 tokens. |
| **fp16** | on GPU | Half-precision arithmetic: about 2× faster, same result for this purpose. |
| **seed** | 42 | Fixes the random choices (negatives, batch order) so a run is repeatable. |

## Train, validation, test: who decides what

- **Train** rows are what the loss sees.
- After every epoch, `ValEval` runs full-pool retrieval on the **validation** split
  and logs `val/acc@1`, `val/recall@5`, and the component accuracies. The epoch
  with the best `val/acc@1` is saved as `best/`. Validation picks the model.
- **Test** is scored once, at the end, with that best model. That is the number
  reported. Test never influences any choice, which is what makes it honest.

The split is by ingredient, so a good val or test score means the model handles
*drugs it has never seen*, not that it memorized metoprolol.

## Reading the W&B charts

- **`train/loss`** should fall quickly in the first few hundred steps and then
  flatten. Noisy is normal (each point is one batch). Rising or NaN means the
  learning rate is too high.
- **`val/acc@1` per epoch** should rise and then plateau. If it *falls* while
  train loss keeps dropping, the model is memorizing the training drugs: stop
  earlier or use fewer epochs. The `best/` checkpoint already guards against this.
- **`train/learning_rate`** shows the warmup-then-decay shape; it's a sanity check
  that the schedule is the intended one.
- **`test/failures` Table:** read it the same way as the baseline's. The
  interesting question is whether the failure mix changed (fewer strength errors?
  still losing to short generic names?).

## What "beat the baseline" means here

TF-IDF on test: acc@1 0.509, recall@5 0.820, coverage@99% precision 0.4%. The
first fine-tuned MiniLM landed at 0.836 / 0.967; the final SapBERT model at
0.931 / 0.984. Either way the failure table says why, and calibration (see
[calibration-and-sweeps.md](calibration-and-sweeps.md)) is what turns the cosine
score into a confidence a threshold can be set on.
