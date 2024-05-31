import yaml
from pathlib import Path
from datasets import Dataset, DatasetDict
import evaluate
import json
import numpy as np
import pandas as pd
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import classification_report


params_dir = Path(__file__).parent.resolve()

with open(params_dir / 'train_8shot_gens.yaml', 'r') as f:
    params = yaml.load(f, Loader=yaml.Loader)

print(f'Training generation {params["generation"]}')

n_shot = params['n_shot']

run_name = f'setfit_{n_shot}shot_generations{params["generation"]}_{params["add_train"]}tr'
short_run_name = f'{n_shot}shot_gen{params["generation"]}'

next_gen_n = params["generation"] + 1

in_dir = Path(params['in_dir'])

next_gen_dir = Path(in_dir / f'{n_shot}shot_gen_{next_gen_n}')
next_gen_dir.mkdir(exist_ok=True, parents=True)

if 'out_dir' not in params:
    model_save_dir = Path(params['model_save_dir'])
    model_save_dir.mkdir(exist_ok=True, parents=True)

    report_dir = Path(params['report_dir'])
    report_dir.mkdir(exist_ok=True, parents=True)
else:
    out_dir = Path(params['out_dir']) / short_run_name
    out_dir.mkdir(exist_ok=True, parents=True)
    model_save_dir = out_dir
    report_dir = out_dir

if params["generation"] == 1:
    curr_gen_dir = in_dir / f'{n_shot}shot'
else:
    curr_gen_dir = in_dir / f'{n_shot}shot_gen_{params["generation"]}'

train_df = pd.read_json(curr_gen_dir / 'train.jsonl', orient='records', lines=True).sample(frac=1.0, random_state=99).reset_index(drop=True)
base_val_df = pd.read_json(in_dir/ f'{n_shot}shot' / 'valid.jsonl', orient='records', lines=True).sample(frac=1.0, random_state=999).reset_index(drop=True)
ext_val_df = pd.read_json(in_dir/ f'{n_shot}shot' / 'valid_extended.jsonl', orient='records', lines=True).sample(frac=1.0, random_state=999).reset_index(drop=True)

test_df = pd.read_json(in_dir / 'test.jsonl', orient='records', lines=True)
holdout = pd.read_json(in_dir / 'artificial_test.jsonl', orient='records', lines=True)
unlabeled_df = pd.read_json(curr_gen_dir / 'unlabeled.jsonl', orient='records', lines=True)

train = Dataset.from_pandas(train_df)
valid = Dataset.from_pandas(ext_val_df)
test = Dataset.from_pandas(test_df)
unlabeled = Dataset.from_pandas(unlabeled_df)

dataset = DatasetDict({
    'train': train,
    'test': test,
    'valid': valid,
    'unlabeled': unlabeled
})

f1_metric = evaluate.load("f1", "micro")

def compute_metrics(y_pred, y_test):
    return {
        "f1": f1_metric.compute(predictions=y_pred, references=y_test, average="micro")["f1"],
    }

model = SetFitModel.from_pretrained(params['model_name'])

args = TrainingArguments(
    **params['train_args'],
    loss=CosineSimilarityLoss,
    run_name=run_name
)

trainer = Trainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'],
    args=args,
    column_mapping={"text": "text", "target": "label"}
)

trainer.train()

print('Validation data CR')
eval_preds = model.predict(base_val_df['text'].tolist(), show_progress_bar=False)
base_val_df['pred'] = eval_preds
print(classification_report(base_val_df['target'], base_val_df['pred']))
cr = classification_report(base_val_df['target'], base_val_df['pred'], output_dict=True)
with open(report_dir / f'{short_run_name}_val_base_cr.json', 'w') as f:
    json.dump(cr, f, ensure_ascii=False)


print('Artificial holdout data CR')
preds = model.predict(holdout['text'], show_progress_bar=True)

with open(report_dir / f'{short_run_name}_artif_holdout.txt', 'w') as f:
    for p in preds:
        f.write(f'{p}\n')

holdout['pred'] = preds
print(classification_report(holdout['target'], holdout['pred']))

cr = classification_report(holdout['target'], holdout['pred'], output_dict=True)

with open(report_dir / f'{short_run_name}_artif_test_cr.json', 'w') as f:
    json.dump(cr, f, ensure_ascii=False)


print('Test data CR')
preds = model.predict(test['text'], show_progress_bar=True)

with open(report_dir / f'{short_run_name}.txt', 'w') as f:
    for p in preds:
        f.write(f'{p}\n')

test_df['pred'] = preds
print(classification_report(test_df['target'], test_df['pred']))

cr = classification_report(test_df['target'], test_df['pred'], output_dict=True)

with open(report_dir / f'{short_run_name}_test_cr.json', 'w') as f:
    json.dump(cr, f, ensure_ascii=False)

model.save_pretrained(model_save_dir / f'{short_run_name}_model')

unlabeled_preds = model.predict_proba(unlabeled['text'], show_progress_bar=True, as_numpy=True)

np.save(next_gen_dir / f'{short_run_name}_probas.pkl', unlabeled_preds, allow_pickle=True)

pred_class_nums = np.argmax(unlabeled_preds, axis=1)

pred_probs = np.max(unlabeled_preds, axis=1)

unlabeled_preds_df = pd.DataFrame(list(zip(pred_class_nums, pred_probs)),
             columns=['target_num', 'prob'])

class_map = {
    0: 'бізнес',
    1: 'новини',
    2: 'політика',
    3: 'спорт',
    4: 'технології',
}

unlabeled_preds_df['target'] = unlabeled_preds_df['target_num'].map(class_map)

top_prob_preds = unlabeled_preds_df.groupby('target')['prob'].nlargest(params['next_add_train']).reset_index()
top_prob_preds = top_prob_preds[top_prob_preds['prob'] >= params['add_prob_threshold']]

pseudo_labeled = unlabeled_df.loc[top_prob_preds['level_1'].tolist(), ].reset_index(drop=True)
pseudo_labeled['target'] = top_prob_preds['target']
pseudo_labeled['prob'] = top_prob_preds['prob']
pseudo_labeled['gen'] = params["generation"]


train_df_new_iter = pd.concat([train_df, pseudo_labeled], ignore_index=True)

train_df_new_iter.to_json(next_gen_dir / 'train.jsonl', orient='records', lines=True, force_ascii=False)
unlabeled_df.drop(index=top_prob_preds['level_1'].tolist()).to_json(next_gen_dir / 'unlabeled.jsonl', orient='records', lines=True, force_ascii=False)

unlabeled_preds_df.to_csv(next_gen_dir / f'{n_shot}shot_preds_table.csv', index=False)
unlabeled_df.loc[unlabeled_df.index.isin(top_prob_preds['level_1']), 'id']\
    .to_csv(next_gen_dir / 'used_unlabeled_idx.csv', index=False)

if 'out_dir' in params:
    data_out_dir = out_dir / 'data_for_next_gen'
    data_out_dir.mkdir(exist_ok=True, parents=True)
    unlabeled_preds_df.to_csv(data_out_dir / f'{n_shot}shot_preds_table.csv', index=False)
    unlabeled_df.loc[unlabeled_df.index.isin(top_prob_preds['level_1']), 'id']\
        .to_csv(data_out_dir / 'used_unlabeled_idx.csv', index=False)
    train_df_new_iter.to_json(data_out_dir / 'train.jsonl', orient='records', lines=True, force_ascii=False)
    unlabeled_df.drop(index=top_prob_preds['level_1'].tolist()).to_json(data_out_dir / 'unlabeled.jsonl', orient='records', lines=True, force_ascii=False)


params['generation'] = params['generation'] + 1
params['add_train'] = params['next_add_train']
params['next_add_train'] = params['next_add_train'] + int(np.round(params['next_add_train'] / 2))

if params['add_prob_threshold'] < 0.85:
    params['add_prob_threshold'] = params['add_prob_threshold'] + 0.05

with open(params_dir / f'train_params_{n_shot}shot_gen{next_gen_n}.yaml', 'w') as f:
    yaml.dump(params, f, Dumper=yaml.Dumper)

if 'out_dir' in params:
    with open(out_dir / f'train_params_{n_shot}shot_gen{next_gen_n}.yaml', 'w') as f:
        yaml.dump(params, f, Dumper=yaml.Dumper)