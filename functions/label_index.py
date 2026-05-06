import os
from collections import Counter


def _clean_label(name):
    name = str(name).replace('_', ' ').replace('-', ' ').strip().lower()
    return ' '.join(name.split())


def _label_from_path(path):
    base = os.path.basename(str(path))
    stem, _ = os.path.splitext(base)
    parent = os.path.basename(os.path.dirname(str(path)))
    if parent and parent not in ('', '.', '/'):  # imagenet-like datasets
        return _clean_label(parent)
    return _clean_label(stem)


def _hf_labels(image_paths, max_labels=3):
    try:
        from PIL import Image
        from transformers import pipeline
    except Exception:
        return None

    clf = pipeline('image-classification', model='google/vit-base-patch16-224')
    labels = []
    for image_path in image_paths:
        try:
            pred = clf(Image.open(image_path), top_k=1)
            if pred:
                labels.append(_clean_label(pred[0].get('label', '')))
        except Exception:
            continue
    if not labels:
        return None
    return [k for k, _ in Counter(labels).most_common(max_labels)]


def get_label_selectivity_idx(neuron_data, max_images=12, max_labels=3):
    image_paths = [p for p in neuron_data.images_id[:max_images] if p]
    labels = _hf_labels(image_paths, max_labels=max_labels)
    source = 'model'
    if not labels:
        labels = [_label_from_path(path) for path in image_paths]
        labels = [k for k, _ in Counter(labels).most_common(max_labels)]
        source = 'path'

    if not labels:
        text = 'No representative label found'
    elif len(labels) == 1:
        text = labels[0]
    else:
        text = ', '.join(labels[:-1]) + ' and ' + labels[-1]

    return {'label': text, 'source': source, 'top_labels': labels}
