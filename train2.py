import functools
import time

import ignite
import torch

import dataset
import glomerus_detector
import metrics


def evaluate(train_engine, test_engine, data_loader, net=None):
    if net is not None:
        with test_engine.add_event_handler(
                ignite.engine.Events.COMPLETED,
                ignite.handlers.ModelCheckpoint(
                    "./models/",
                    "hub_detector9_5120",
                    score_function=lambda e: e.state.metrics['dice'],
                    score_name="dice",
                    n_saved=5,
                    require_empty=False,
                    create_dir=True), {'model': net}):
            metrics = test_engine.run(data_loader, max_epochs=1).metrics
    else:
        metrics = test_engine.run(data_loader, max_epochs=1).metrics

    for name, value in metrics.items():
        print(f"[{time.asctime()}] {train_engine.state.epoch}: {name} {value}")


def prepare_batch(batch, device, non_blocking):
    img, mask = batch
    img = img.float().to(device)
    mask = mask.float().to(device)

    return img, mask


class OutputTransform():
    def __init__(self, threshold=0.5):
        self._threshold = threshold

    def __call__(self, output):
        predicted, target = output
        predicted = predicted[0].flatten(1) >= self._threshold
        target = target.flatten(1) >= self._threshold

        return predicted, target


class OptimCriterion:
    def __init__(self, criterion):
        self._criterion = criterion

    def __call__(self, predicted, target):
        return functools.reduce(lambda x, y: x + y, [
            self._criterion(
                p, target if i == 1 else target[:, :, 0::2**i, 0::2**i])
            for i, p in enumerate(predicted, start=1)
        ])


def main():
    if True and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(device)

    #data_loader = torch.utils.data.DataLoader(dataset.HubmapDataset(
    #    "./data/train", tiles_per_polygon=1, tile_size=(1024, 1024)),
    #                                          batch_size=24,
    #                                          shuffle=False,
    #                                          num_workers=2)
    #data_loader = torch.utils.data.DataLoader(dataset.HubmapDataset2(
    #    "./data/train", tile_size=1024),
    #                                          batch_size=24,
    #                                          shuffle=False,
    #                                          num_workers=2)
    data_loader = torch.utils.data.DataLoader(dataset.HubmapDataset2(
        "./data/train", tile_size=1024 * 5),
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=2)

    score_criterion = OptimCriterion(torch.nn.BCELoss(reduction='mean'))

    net = glomerus_detector.GlomerusDetector9()
    net.load_state_dict(
        torch.load("./models/hub_detector9_5120_model_dice=0.9162.pt",
                   map_location=device))
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.000005)

    trainer = ignite.engine.create_supervised_trainer(
        net,
        optimizer,
        score_criterion,
        device=device,
        prepare_batch=prepare_batch)

    threshold = 0.5
    evaluator = ignite.engine.create_supervised_evaluator(
        net, {
            "loss": ignite.metrics.Loss(score_criterion, device=device),
            "precision": metrics.Precision(OutputTransform(threshold)),
            "recall": metrics.Recall(OutputTransform(threshold)),
            "dice": metrics.Dice(OutputTransform(threshold))
        },
        device,
        prepare_batch=prepare_batch)

    trainer.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED(every=5),
                              evaluate, evaluator, data_loader, net)

    #evaluate(trainer, evaluator, data_loader)
    trainer.run(data_loader, max_epochs=5 * 2)


if __name__ == "__main__":
    main()
