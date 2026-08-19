import torch

from models.classifier import IncrementalClassifier, get_task_logits


def test_incremental_classifier_expansion_preserves_old_logits():
    torch.manual_seed(42)
    classifier = IncrementalClassifier(in_features=4, num_classes=2)
    features = torch.randn(5, 4)
    old_logits = classifier(features).detach()

    classifier.expand(num_new_classes=3)
    new_logits = classifier(features).detach()

    assert new_logits.shape == (5, 5)
    assert torch.allclose(new_logits[:, :2], old_logits, atol=1e-7)


def test_freeze_old_classes_masks_old_gradients():
    classifier = IncrementalClassifier(in_features=3, num_classes=2)
    classifier.expand(num_new_classes=2, freeze_old=True)
    features = torch.randn(6, 3)
    loss = classifier(features).sum()
    loss.backward()

    assert torch.allclose(classifier.weight.grad[:2], torch.zeros_like(classifier.weight.grad[:2]))
    assert torch.count_nonzero(classifier.weight.grad[2:]) > 0


def test_get_task_logits_slices_task_window():
    logits = torch.arange(30, dtype=torch.float32).reshape(3, 10)

    task_logits = get_task_logits(logits, task_id=1, classes_per_task=4)

    assert torch.equal(task_logits, logits[:, 4:8])

