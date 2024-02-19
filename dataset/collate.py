import torch
import re
import collections
import torch.nn.functional as f

np_str_obj_array_pattern = re.compile(r'[SaUO]')

""" Adapted from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py"""

def collate_variable_length_series(batch, pad_val):
    elem_0 = None
    it = iter(batch)
    while elem_0 is None:
        elem_0 = next(it)
    elem_type = type(elem_0)
    if isinstance(elem_0, torch.Tensor):
        shapes = [elem.shape for elem in batch if elem is not None]
        if not all([s == shapes[0] for s in shapes[1:]]): # the tensors have differing shapes (can be the case for target tensors)
            temp_shapes = [s[1] for s in shapes] # assumes the 1st dimension is the temporal dimension
            if all([s == temp_shapes[0] for s in temp_shapes[1:]]):
                raise IndexError('Tensors to collate should have similar shape for all non-temporal dimensions')
            else: # pad along temporal dimension
                # print('padding targets')
                max_temp = max(temp_shapes)
                batch = tuple(elem  if s == max_temp
                                    else f.pad(elem, (0, max_temp-s) + (0,) * 2 * len(elem.shape[1:]), mode='constant', value=pad_val)
                                    for s, elem in zip(temp_shapes, batch)) # the padding value will be ignored by the loss
                # elem_0 might have been padded
                elem_0 = None
                it = iter(batch)
                while elem_0 is None:
                    elem_0 = next(it)
        if None in batch: #the current time step is not available in all patches (can be the case for inputs)
            # replace None by empty tensors
            batch = tuple(torch.zeros_like(elem_0) if elem is None else elem for elem in batch)
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem_0._typed_storage()._new_shared(numel)
            out = elem_0.new(storage).resize_(len(batch), *list(elem_0.size())) 
        return torch.stack(batch, 0, out=out) 
    elif isinstance(elem_0, list) or isinstance(elem_0, tuple):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            # pad elements so that they all have the same lengths
            max_len = max([len(elem) for elem in batch])
            batch = tuple(elem if len(elem)==max_len 
                                else elem + elem_type([None for _ in range(max_len-len(elem))]) 
                                for elem in batch)
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem_0, tuple):
            return [collate_variable_length_series(samples, pad_val) for samples in transposed]  # Backwards compatibility.
        else:
            return elem_type([collate_variable_length_series(samples, pad_val) for samples in transposed])

    raise TypeError('collate_variable_length_series only supports lists, tuples and torch.Tensor objects,'
                    'but {} was provided'.format(elem_type))

def my_default_collate(batch):
    "strings not supported"
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.typed_storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError()

            return my_default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: my_default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: my_default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(my_default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [my_default_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([my_default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [my_default_collate(samples) for samples in transposed]

    raise TypeError()