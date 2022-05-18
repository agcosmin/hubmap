import xml.etree.ElementTree

import imageio
import numpy
import torch


class RLEPoint:
    def __init__(self, start, length):
        self.start = start
        self.length = length


def decode_rle(stream, width, height, offset=0):
    mask = torch.zeros((width, height), dtype=torch.bool).flatten()
    for start, run in zip(list(stream[0::2]), list(stream[1::2])):
        mask[start - offset:start - offset + run] = True

    return mask


def encode_rle(stream, offset, continue_from=None, write_last=False, step=1):
    point = continue_from if continue_from is not None else RLEPoint(None, 0)
    encoding = []
    for i in range(0, stream.shape[0], step):
        if stream[i]:
            if point.length == 0:
                point.start = i + offset
            point.length += 1
        else:
            if point.length != 0:
                encoding.append(point.start)
                encoding.append(point.length)
                point.length = 0

    if write_last and point.start is not None:
        encoding.append(point.start)
        encoding.append(point.length)

    if stream[-1]:
        return encoding, point
    else:
        return encoding, None


def encode_mask_rle(mask, offset, step=1):
    point = RLEPoint(None, 0)
    encoding = []
    for w in range(0, mask.shape[0], 1):
        for h in range(0, mask.shape[1], step):
            p = w * mask.shape[1] + h
            if mask[w][h]:
                if point.length == 0:
                    point.start = p + offset
                point.length += step
            else:
                if point.length != 0:
                    encoding.append(point.start)
                    encoding.append(point.length)
                    point.length = 0
                    point.start = None

    if point.start is not None:
        encoding.append(point.start)
        encoding.append(point.length)

    return encoding


def get_permutation_to_CWH(shape, channels, width, height):
    shape = list(shape)
    return (shape.index(channels), shape.index(width), shape.index(height))


def get_shape(metadata):
    root = xml.etree.ElementTree.fromstring(metadata['description'])
    image = root.find(
        '{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image')
    width = int(
        image.find('{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels'
                   ).get("SizeX"))
    height = int(
        image.find('{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels'
                   ).get("SizeY"))
    channels = int(
        image.find('{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels'
                   ).get("SizeC"))

    return (channels, width, height)


def read_image(file):
    reader = imageio.get_reader(file)
    if len(reader) != 1:
        data = numpy.stack([reader.get_data(i) for i in range(len(reader))],
                           axis=0)
    else:
        data = reader.get_data(0)
    shape = get_shape(reader.get_meta_data(0))
    permutation = get_permutation_to_CWH(data.shape, shape[0], shape[1],
                                         shape[2])
    img = torch.from_numpy(numpy.asarray(data))
    img = img.permute(permutation)

    return img


def patch_indices(img_shape, patch_shape, patch_strides):
    a0_bounds = torch.arange(0, img_shape[0], patch_strides[0])
    a0_bounds = torch.stack(
        (a0_bounds, torch.clip(a0_bounds + patch_shape[0], 0, img_shape[0])),
        dim=1)

    a1_bounds = torch.arange(0, img_shape[1], patch_strides[1])
    a1_bounds = torch.stack(
        (a1_bounds, torch.clip(a1_bounds + patch_shape[1], 0, img_shape[1])),
        dim=1)

    bounds = torch.stack([
        torch.cat((a0_bounds[a0.item()], a1_bounds[a1.item()]))
        for a0, a1 in torch.cartesian_prod(torch.arange(a0_bounds.shape[0]),
                                           torch.arange(a1_bounds.shape[0]))
    ])
    return bounds


def generate_batches(img, patch_shape, batch_size, overlap=0):
    patches_coords = patch_indices(img.shape[1:], patch_shape,
                                   patch_shape - overlap)

    for batch_i in range(0, patches_coords.shape[0], batch_size):
        coords = patches_coords[batch_i:min(batch_i +
                                            batch_size, patches_coords.shape[0]
                                            )]
        batch = torch.stack([
            torch.nn.functional.pad(img[:, c[0]:c[1], c[2]:c[3]].clone(),
                                    (0, patch_shape[1] - (c[3] - c[2]), 0,
                                     patch_shape[0] - (c[1] - c[0])))
            for c in coords
        ])
        yield batch, coords


def predict_mask(img, net, device, patch_shape, batch_size):
    prediction = torch.zeros(img.shape[1:])
    for batch, coords in generate_batches(img, torch.tensor(patch_shape),
                                          batch_size, 0):
        batch = batch.to(device).float()
        with torch.no_grad():
            pred = net(batch)[0].cpu()

        for b, coord in enumerate(coords):
            prediction[coord[0]:coord[1],
                       coord[2]:coord[3]] = pred[b, 0, 0:coord[1] - coord[0],
                                                 0:coord[3] - coord[2]]
    return prediction


def predict_and_threhold_mask(img, net, threshold, device, patch_shape,
                              batch_size):
    prediction = torch.zeros(img.shape[1:], dtype=torch.bool)
    for batch, coords in generate_batches(img, torch.tensor(patch_shape),
                                          batch_size, 0):
        batch = batch.to(device).float()
        with torch.no_grad():
            pred = (net(batch)[0] >= threshold).cpu()

        for b, coord in enumerate(coords):
            prediction[coord[0]:coord[1],
                       coord[2]:coord[3]] = pred[b, 0, 0:coord[1] - coord[0],
                                                 0:coord[3] - coord[2]]
    return prediction

