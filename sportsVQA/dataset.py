import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder


class vqaDataset(Dataset):
    def __init__(self, vqa_annotations, vqa_questions, image_features, only_yes_no=False):
        super().__init__()
        self.vqa_annotations = vqa_annotations
        self.vqa_questions = vqa_questions
        self.image_features = image_features
        self.used_question_idxs = list(range(len(self.vqa_annotations)))
        self.only_yes_no = only_yes_no
        if only_yes_no:
            for idx, annotations in enumerate(self.vqa_annotations):
                if annotations['answer_type'] != 'yes/no':
                    self.used_question_idxs.remove(idx)
        self.annotations = [self.vqa_annotations[i] for i in self.used_question_idxs]
        self.questions = [self.vqa_questions[i] for i in self.used_question_idxs]
        self.answers = list(map(lambda annotation: annotation['multiple_choice_answer'], self.annotations))
        label_encoder = LabelEncoder()
        self.labels = torch.tensor(label_encoder.fit_transform(self.answers))
        

    def __len__(self):
        return len(self.used_question_idxs)

    def __getitem__(self, idx):
        ds_idx = self.used_question_idxs[idx]
        # annotation = self.vqa_annotations[ds_idx]
        # question = self.vqa_questions[ds_idx]
        annotation = self.annotations[idx]
        question = self.questions[idx]
        assert annotation['question_id'] == question['question_id']
        image_features = self.image_features[annotation['image_id']]
        sentence_representation = question['bert_embedding'][0]
        word_representations = question['bert_embedding'][1:]
        answer = annotation['multiple_choice_answer']
        label = self.labels[idx]

        result = {
            'image_id': annotation['image_id'],
            'answer': answer,
            'label': label,
            'question': question['question'],
            'sentence_representation': sentence_representation,
            'word_representations': word_representations,
            'image_features': image_features
        }
        if not self.only_yes_no:
            result['answer_type'] = annotation['answer_type']
            result['multiple_choices'] = question['multiple_choices']
        return result

    def collate(self, list_of_samples):
        img_ids, answers, labels, questions, sentence_representations, word_representations, image_features = zip(
            *(
                (s['image_id'],
                 s['answer'],
                 s['label'],
                 s['question'],
                 s['sentence_representation'],
                 s['word_representations'],
                 s['image_features']
                 )
                for s in list_of_samples
            )
        )
        max_length = max([r.size(0) for r in word_representations])
        padded_word_representations = [F.pad(r, [0, 0, 0, max_length - r.size(0)]) for r in word_representations]
        word_masks = [torch.ones(r.size(0)) for r in word_representations]
        padded_word_masks = [F.pad(r, [0, max_length - r.size(0)]) for r in word_masks]
        result = {
            'image_ids': img_ids,
            'answers': answers,
            'labels': torch.stack(labels),
            'questions': questions,
            'sentence_representations': torch.stack(sentence_representations),
            'word_representations': torch.stack(padded_word_representations),
            'word_masks': torch.stack(padded_word_masks).bool(),
            'image_features': torch.stack(image_features),
        }

        if not self.only_yes_no:
            answer_types, multiple_choices = zip(
                *(
                    (s['answer_type'],
                     s['multiple_choices'],
                     )
                    for s in list_of_samples
                )
            )
            result['answer_types'] = answer_types
            result['multiple_choices'] = multiple_choices
        return result


def create_vqa_datasets(vqa_path, image_features_path, part2=False):
    print('loading datafiles...')
    vqa_data = torch.load(vqa_path)
    image_features = torch.load(image_features_path)
    print('loading datafiles done!')

    train_ds = vqaDataset(vqa_data['train_annotations'], vqa_data['train_questions'], image_features['train'], only_yes_no=not part2)
    val_ds = vqaDataset(vqa_data['val_annotations'], vqa_data['val_questions'], image_features['val'], only_yes_no=not part2)
    test_ds = vqaDataset(vqa_data['test_annotations'], vqa_data['test_questions'], image_features['test'], only_yes_no=not part2)
    return train_ds, val_ds, test_ds
