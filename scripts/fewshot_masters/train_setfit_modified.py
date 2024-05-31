import yaml
from pathlib import Path
import random
from datasets import Dataset, DatasetDict
import evaluate
import json
import numpy as np
import pandas as pd
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import torch

from setfit.text_augmentations import augment_text, tokenizer
from setfit.custom_model_head import NN_Trainer
import nlpaug.augmenter.word as naw


np.random.seed(99)
random.seed(99)
torch.manual_seed(99)
torch.cuda.manual_seed(99)


class_map = {
    'бізнес': 0,
    'новини': 1,
    'політика': 2,
    'спорт': 3,
    'технології': 4,
}

inverse_class_map = {
    0: 'бізнес',
    1: 'новини',
    2: 'політика',
    3: 'спорт',
    4: 'технології',
}

cyr_class_map = {
    0: 'business',
    1: 'news',
    2: 'politics',
    3: 'sport',
    4: 'tech',
}

tfidf_aug = {k: naw.TfIdfAug(model_path=f'aug/tfidf_model/{v}', #add full path to trained tf-idf here
                             tokenizer=tokenizer) for k, v in cyr_class_map.items()}

params_dir = Path(__file__).parent.resolve()

with open(params_dir / 'train_8shot_aug.yaml', 'r') as f:
    params = yaml.load(f, Loader=yaml.Loader)

print(f'Training generation {params["generation"]}')

n_shot = params['n_shot']

run_name = f'setfit_{n_shot}shot_aug_nn_gen{params["generation"]}'
short_run_name = run_name

next_gen_n = params["generation"] + 1

in_dir = Path(params['in_dir'])

next_gen_dir = Path(in_dir / f'{n_shot}shot_aug_nn_gen{next_gen_n}')
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
    curr_gen_dir = in_dir / f'{n_shot}shot_aug_nn_gen{params["generation"]}'

train_df = pd.read_json(curr_gen_dir / 'train_backtranslated.jsonl', orient='records', lines=True).sample(frac=1.0, random_state=99).reset_index(drop=True)
base_val_df = pd.read_json(in_dir/ f'{n_shot}shot' / 'valid.jsonl', orient='records', lines=True).sample(frac=1.0, random_state=999).reset_index(drop=True)
ext_val_df = pd.read_json(in_dir/ f'{n_shot}shot' / 'valid_extended.jsonl', orient='records', lines=True).sample(frac=1.0, random_state=999).reset_index(drop=True)

test_df = pd.read_json(in_dir / 'test.jsonl', orient='records', lines=True)
holdout = pd.read_json(in_dir / 'artificial_test.jsonl', orient='records', lines=True)
unlabeled_df = pd.read_json(curr_gen_dir / 'unlabeled.jsonl', orient='records', lines=True)

if 'id' not in unlabeled_df.columns:
    unlabeled_df['id'] = unlabeled_df.index

train_df['target'] = train_df['target'].map(class_map)

train = Dataset.from_pandas(train_df[train_df['backtranslated'] == False])
valid = Dataset.from_pandas(ext_val_df)
test = Dataset.from_pandas(test_df)
unlabeled = Dataset.from_pandas(unlabeled_df)

if not params['text_augment_params']['backtranslation']:
    aug_train_df = train_df[train_df['backtranslated'] == False] #.reset_index(drop=True)
else:
    if params['text_augment_params']['lang'] == 'all':
        aug_train_df = train_df

    elif params['text_augment_params']['lang'] == 'eng':
        aug_train_df = train_df[(train_df['backtranslated'] == False) | (train_df['lang'] == 'eng')]

aug_train_df = aug_train_df.reset_index(drop=True)

# augment texts for classifier training
x_orig = aug_train_df.loc[aug_train_df['backtranslated'] == False, 'text'].tolist()
y_orig = aug_train_df.loc[aug_train_df['backtranslated'] == False, 'target'].tolist()

if params['text_augment_params']['backtranslation']:
    x_backtr = aug_train_df.loc[aug_train_df['backtranslated'] == True, 'text'].tolist()
    y_backtr = aug_train_df.loc[aug_train_df['backtranslated'] == True, 'target'].tolist()

    x_aug, y_aug = augment_text(x_orig, y_orig, x_backtr, y_backtr,
                                **params['text_augment_params'], tfidf_models=tfidf_aug)
else:
    x_aug, y_aug = augment_text(x_orig, y_orig,
                                **params['text_augment_params'], tfidf_models=tfidf_aug)

print(f'Body training size: {len(train["text"])}')
print(f'Head training size: {len(x_aug)}')

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

model = SetFitModel.from_pretrained(params['model_name'], use_differentiable_head=True, head_params=params['head_params'])


args = TrainingArguments(
    **params['train_args'],
    **params['head_train_args'],
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

print('Extended valid data CR')
eval_preds = model.predict(valid['text'], show_progress_bar=False).tolist()
ext_val_df['pred'] = eval_preds
ext_val_df['pred'] = ext_val_df['pred'].map(inverse_class_map)
print(classification_report(ext_val_df['target'], ext_val_df['pred']))
cr = classification_report(ext_val_df['target'], ext_val_df['pred'], output_dict=True)
with open(report_dir / f'{short_run_name}_val_ext_cr.json', 'w') as f:
    json.dump(cr, f, ensure_ascii=False)

print('Base valid data CR')
eval_preds = model.predict(base_val_df['text'].tolist(), show_progress_bar=False).tolist()
base_val_df['pred'] = eval_preds
base_val_df['pred'] = base_val_df['pred'].map(inverse_class_map)
print(classification_report(base_val_df['target'], base_val_df['pred']))
cr = classification_report(base_val_df['target'], base_val_df['pred'], output_dict=True)
with open(report_dir / f'{short_run_name}_val_base_cr.json', 'w') as f:
    json.dump(cr, f, ensure_ascii=False)


print('Artificial holdout data CR')
preds = model.predict(holdout['text'], show_progress_bar=True).tolist()

holdout['pred'] = preds
holdout['pred'] = holdout['pred'].map(inverse_class_map)
print(classification_report(holdout['target'], holdout['pred']))

cr = classification_report(holdout['target'], holdout['pred'], output_dict=True)

with open(report_dir / f'{short_run_name}_artif_test_cr.json', 'w') as f:
    json.dump(cr, f, ensure_ascii=False)

with open(report_dir / f'{short_run_name}_artif_holdout.txt', 'w') as f:
    for p in holdout['pred'].tolist():
        f.write(f'{p}\n')


print('Test data CR')
preds = model.predict(test['text'], show_progress_bar=True).tolist()

test_df['pred'] = preds
test_df['pred'] = test_df['pred'].map(inverse_class_map)
print(classification_report(test_df['target'], test_df['pred']))

cr = classification_report(test_df['target'], test_df['pred'], output_dict=True)

with open(report_dir / f'{short_run_name}_test_cr.json', 'w') as f:
    json.dump(cr, f, ensure_ascii=False)

with open(report_dir / f'{short_run_name}.txt', 'w') as f:
    for p in test_df['pred'].tolist():
        f.write(f'{p}\n')


model.save_pretrained(model_save_dir / f'{short_run_name}_model_non_aug')

model.freeze('body')

print('Training aug head')

optimizer_params = params['head_train_args']
optimizer_params['body_learning_rate'] = params['train_args']['body_learning_rate']

nn_trainer = NN_Trainer(x_train=x_aug, y_train=y_aug, setfit_model=model, params=params['head_params'],
                        optimizer_params=optimizer_params, n_epochs=params['train_args']['batch_size'][1],
                        device='cuda:0', **params['emb_aug_params'])

nn_trainer.fit(emb_aug_multipl=params['emb_aug_params']['emb_aug_multipl'])


classifier = nn_trainer.classifier


print('Extended valid data CR AUG')

x_valid = model.encode(ext_val_df['text'].tolist())

eval_preds = classifier.predict(x_valid).tolist()

ext_val_df['pred'] = eval_preds
ext_val_df['pred'] = ext_val_df['pred'].map(inverse_class_map)
print(classification_report(ext_val_df['target'], ext_val_df['pred']))
cr = classification_report(ext_val_df['target'], ext_val_df['pred'], output_dict=True)
with open(report_dir / f'{short_run_name}_val_ext_cr_aug.json', 'w') as f:
    json.dump(cr, f, ensure_ascii=False)

print('Base valid data CR')
x_valid = model.encode(base_val_df['text'].tolist())

eval_preds = classifier.predict(x_valid).tolist()

base_val_df['pred'] = eval_preds
base_val_df['pred'] = base_val_df['pred'].map(inverse_class_map)
print(classification_report(base_val_df['target'], base_val_df['pred']))
cr = classification_report(base_val_df['target'], base_val_df['pred'], output_dict=True)
with open(report_dir / f'{short_run_name}_val_base_cr_aug.json', 'w') as f:
    json.dump(cr, f, ensure_ascii=False)


print('Artificial holdout data CR AUG')
x_holdout = model.encode(holdout['text'].tolist())

preds = classifier.predict(x_holdout).tolist()

holdout['pred'] = preds
holdout['pred'] = holdout['pred'].map(inverse_class_map)
print(classification_report(holdout['target'], holdout['pred']))

cr = classification_report(holdout['target'], holdout['pred'], output_dict=True)

with open(report_dir / f'{short_run_name}_artif_test_cr_aug.json', 'w') as f:
    json.dump(cr, f, ensure_ascii=False)

with open(report_dir / f'{short_run_name}_artif_holdout_aug.txt', 'w') as f:
    for p in holdout['pred'].tolist():
        f.write(f'{p}\n')

print('Test data CR')
x_test = model.encode(test_df['text'].tolist())

preds = classifier.predict(x_test).tolist()

test_df['pred'] = preds
test_df['pred'] = test_df['pred'].map(inverse_class_map)
print(classification_report(test_df['target'], test_df['pred']))

cr = classification_report(test_df['target'], test_df['pred'], output_dict=True)

with open(report_dir / f'{short_run_name}_test_cr_aug.json', 'w') as f:
    json.dump(cr, f, ensure_ascii=False)

with open(report_dir / f'{short_run_name}.txt', 'w') as f:
    for p in test_df['pred'].tolist():
        f.write(f'{p}\n')


model.model_head = classifier

model.save_pretrained(model_save_dir / f'{short_run_name}_model')

unlabeled_preds = model.predict_proba(unlabeled['text'], show_progress_bar=True, as_numpy=True)

np.save(next_gen_dir / f'{short_run_name}_probas.pkl', unlabeled_preds, allow_pickle=True)

pred_class_nums = np.argmax(unlabeled_preds, axis=1)

pred_probs = np.max(unlabeled_preds, axis=1)

unlabeled_preds_df = pd.DataFrame(list(zip(pred_class_nums, pred_probs)),
             columns=['target_num', 'prob'])

unlabeled_preds_df['target'] = unlabeled_preds_df['target_num'].map(inverse_class_map)

top_prob_preds = unlabeled_preds_df.groupby('target')['prob'].nlargest(params['next_add_train']).reset_index()
# top_prob_preds = top_prob_preds[top_prob_preds['prob'] >= params['add_prob_threshold']]

train_df['target'] = train_df['target'].map(inverse_class_map)

pseudo_labeled = unlabeled_df.loc[top_prob_preds['level_1'].tolist(), ].reset_index(drop=True)
pseudo_labeled['target'] = top_prob_preds['target']
pseudo_labeled['prob'] = top_prob_preds['prob']
pseudo_labeled['gen'] = next_gen_n
pseudo_labeled['lang'] = 'ukr'
pseudo_labeled['backtranslated'] = False

train_df_new_iter = pd.concat([train_df, pseudo_labeled], ignore_index=True)

unlabeled_preds_df.to_csv(next_gen_dir / f'{n_shot}shot_preds_table.csv', index=False)
unlabeled_df.loc[unlabeled_df.index.isin(top_prob_preds['level_1']), 'id']\
    .to_csv(next_gen_dir / 'used_unlabeled_idx.csv', index=False)
train_df_new_iter.to_json(next_gen_dir / 'train.jsonl', orient='records', lines=True, force_ascii=False)
unlabeled_df.drop(index=top_prob_preds['level_1'].tolist()).to_json(next_gen_dir / 'unlabeled.jsonl', orient='records', lines=True, force_ascii=False)


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

with open(params_dir / f'train_params_{n_shot}shot_gen{next_gen_n}_aug_nn.yaml', 'w') as f:
    yaml.dump(params, f, Dumper=yaml.Dumper)

if 'out_dir' in params:
    with open(out_dir / f'train_params_{n_shot}shot_gen{next_gen_n}_aug_nn.yaml', 'w') as f:
        yaml.dump(params, f, Dumper=yaml.Dumper)