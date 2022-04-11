import math
import os
from threading import Lock, Thread
from time import sleep
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from composer.datasets.streaming.download import safe_download
from composer.datasets.streaming.format import (StreamingDatasetIndex, bytes_to_sample_dict, get_index_basename,
                                                get_shard_basename)
from composer.utils import dist


def get_partition() -> Tuple[int, int]:
    """Get how to partition the dataset.

    Returns:
        part (int): Our partition.
        num_parts (int): Out of how many partitions.
    """
    info = get_worker_info()
    if info:
        local_worker_id = info.id
        num_local_workers = info.num_workers
    else:
        local_worker_id = 0
        num_local_workers = 1
    device_id = dist.get_global_rank()
    num_devices = dist.get_world_size()
    return device_id, num_devices, local_worker_id, num_local_workers


class StreamingDataset(IterableDataset):
    """Streaming dataset."""

    def __init__(self,
                 remote: str,
                 local: str,
                 decoders: Dict[str, Optional[Callable]],
                 shuffle: bool,
                 device_batch_size: int = None) -> None:
        """Initialize with the given remote path and local cache.

        Loads all the samples that are available in local cache, then starts a
        background thread to download the rest during training. As samples are
        added, shuffled sample selection becomes more random.

        Args:
            remote (str): Download shards from this remote directory.
            local (str): Download shards to this local filesystem directory for reuse.
            decoders (Dict[str, Optional[Callable]): Raw bytes decoder per sample field.
            shuffle (bool): Whether to shuffle the samples.
        """
        self.remote = remote
        self.local = local
        self.decoders = decoders
        self.shuffle = shuffle
        self.device_batch_size = device_batch_size

        # Load the index file containing the shard metadata, either over the
        # network or cached locally.
        # Precomputes the shard and offset in bytes of each sample (for direct
        # access).
        index_filename = self._download_if_missing(get_index_basename())
        self.index = StreamingDatasetIndex.load(open(index_filename, 'rb'))

        # Fields, protected by the lock, relating to loading shards in the background.
        self._lock = Lock()
        self._next_epoch = 0
        self._epoch_to_todo_ids = {}
        self._downloaded_ids = []
        self._are_all_shards_downloaded = False

    @classmethod
    def split(cls, split: str, remote: str, local: str, decoders: Dict[str, Optional[Callable]], shuffle: bool):
        remote = os.path.join(remote, split)
        local = os.path.join(local, split)
        return cls(remote, local, decoders, shuffle)

    def _download_if_missing(self, basename: str) -> str:
        """Safely download a shard from remote to local cache.

        Args:
            basename (str): Basename of shard to download.

        Returns:
            str: Local cache filename.
        """
        remote = os.path.join(self.remote, basename)
        local = os.path.join(self.local, basename)
        safe_download(remote, local)
        return local

    def _load_shards(self, shards: Sequence[int], part_min_id: int, part_max_id: int) -> None:
        """Load the given list of locally cached shards into the dataset.

        Every time you call __iter__ on this dataset, it registers the list of
        samples you have left, which will not be the full epoch if the dataset
        isn't finished loaded when you start training.

        Calls to _load_shards during training modify the samples remaining on
        these iterations on the fly to insert these new samples and then resort,
        making the shuffle as perfect as was possible.

        This operation takes the lock, so batch your _load_shards calls where
        possible.

        Args:
            shards (Sequence[int]): List of shards to load.
            part_min_id (int): Minimum sample ID of this partition.
            part_max_id (int): Maximum sample ID of this partition.
        """
        # Get all samples from the given shards that fall within our partition.
        new_ids = []
        for shard in shards:
            shard_min_id = self.index.shard_begins[shard]
            shard_max_id = self.index.shard_ends[shard] - 1
            min_id = max(part_min_id, shard_min_id)
            max_id = min(part_max_id, shard_max_id)
            new_ids += list(range(min_id, max_id + 1))

        with self._lock:
            # Extend and optionally reshuffle the remaining samples of any
            # epochs we have in progress.
            if self.shuffle:
                if not self._are_all_shards_downloaded:
                    self._downloaded_ids.extend(new_ids)
                    np.random.shuffle(self._downloaded_ids)
                for todo_ids in self._epoch_to_todo_ids.values():
                    todo_ids.extend(new_ids)
                    np.random.shuffle(todo_ids)
            else:
                if not self._are_all_shards_downloaded:
                    self._downloaded_ids.extend(new_ids)
                for todo_ids in self._epoch_to_todo_ids.values():
                    todo_ids.extend(new_ids)

    def _load_shards_if_downloaded(self, shards: Sequence[int], part_min_id: int, part_max_id: int) -> List[int]:
        """Load any of the given shards that are already present in the cache, returning the missing shards.

        Args:
            shards (Sequence[int]): The shards to attempt to load.
            part_min_id (int): Minimum sample ID of this partition.
            part_max_id (int): Maximum sample ID of this partition.

        Returns:
            list of int: The shards that remain to be loaded.
        """
        downloaded = []
        missing = []
        for shard in sorted(shards):
            basename = get_shard_basename(shard)
            local = os.path.join(self.local, basename)
            if os.path.exists(local):
                downloaded.append(shard)
            else:
                missing.append(shard)
        if downloaded:
            self._load_shards(downloaded, part_min_id, part_max_id)
        return missing

    def _done_loading(self) -> None:
        """Callback on completion of loading my shards."""
        with self._lock:
            self._are_all_shards_downloaded = True

    def _download_thread(self, shards: Sequence[int], part_min_id: int, part_max_id: int) -> None:
        """Background thread to download and assimilate missing shards.

        Args:
            shards (list of int): The shards remaining to be downloaded.
            part_min_id (int): Minimum sample ID of this partition.
            part_max_id (int): Maximum sample ID of this partition.
        """
        shards = list(shards)
        if self.shuffle:
            np.random.shuffle(shards)
        for shard in shards:
            basename = get_shard_basename(shard)
            self._download_if_missing(basename)
            shards = shard,
            self._load_shards(shards, part_min_id, part_max_id)
        self._done_loading()

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Dataset length.
        """
        return math.ceil(self.index.total_samples / dist.get_world_size())

    def _unpack_sample(self, data: bytes) -> Dict[str, Any]:
        """Unpack a sample dict from raw bytes.

        First unpacks the str to raw bytes dict, then unpacks each field's raw bytes.

        Args:
            data (bytes): The packed bytes of the sample.

        Returns:
            Dict[str, Any]: The sample dict.
        """
        key_to_raw = bytes_to_sample_dict(data, self.index.fields)
        obj = {}
        for key, decode in self.decoders.items():
            value = key_to_raw[key]
            if decode:
                value = decode(value)
            obj[key] = value
        return obj

    def __getitem__(self, idx: int) -> Any:
        """Get the sample at the index, assuming its shard is loaded.

        Do not call this directly unless all shards have been loaded. Will crash
        if the shard is not loaded.

        Args:
            idx (int): Sample ID.

        Returns:
            Any: The sample.
        """
        shard = self.index.sample_shards[idx]
        offset = self.index.sample_shard_offsets[idx]
        size = self.index.bytes_per_sample[idx]

        basename = get_shard_basename(shard)
        shard_filename = os.path.join(self.local, basename)
        fp = open(shard_filename, 'rb')
        fp.seek(offset)
        data = fp.read(size)
        fp.close()

        return self._unpack_sample(data)

    def _new_growing_epoch(self) -> int:
        """Start a new growing epoch, in which we own the sample sequence because it grows.

        Returns:
            int: The epoch ID, an identifier which is given back to the caller.
        """
        with self._lock:
            epoch = self._next_epoch
            self._next_epoch += 1
            self._epoch_to_todo_ids[epoch] = list(self._downloaded_ids)
        return epoch

    def _next_id(self, epoch: int) -> Optional[int]:
        """Get next sample of the growing epoch given by epoch, or None if done.

        If we are currently out of samples but not finished downloading the
        shards, blocks until it has new samples.

        Args:
            epoch (int): The epoch, an identifier for this sequence of samples.

        Returns:
            int: ID of next sample.
        """
        while True:
            with self._lock:
                todo_ids = self._epoch_to_todo_ids[epoch]
                if todo_ids:
                    return todo_ids.pop()
                elif self._are_all_shards_downloaded:
                    del self._epoch_to_todo_ids[epoch]
                    return None
                else:
                    pass
            sleep(0.25)

    def _iter_ids(self) -> Iterator[int]:
        """Get an iterator over all our sample IDs.

        Returns:
            Iterator[int]: Each sample ID.
        """
        with self._lock:
            have_full_epoch = self._are_all_shards_downloaded

        if have_full_epoch:
            ids = list(self._downloaded_ids)
            if self.shuffle:
                np.random.shuffle(ids)
            for idx in ids:
                yield idx
        else:
            epoch = self._new_growing_epoch()
            while True:
                idx = self._next_id(epoch)
                if idx is None:
                    break
                yield idx

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Iterate over all the samples in our partition.

        If not all samples have been downloaded yet, iterates over what it has
        while inserting the remainder into the sequence behind the scenes as it
        progresses.

        Returns:
            Iterator[Tuple[Tensor, Tensor]]: Each sample.
        """
        # We find out num workers, and therefore num partitions, when __iter__ is called.
        # From the partition, derive our shard overlap range and exact sample range.
        device_id, num_devices, local_worker_id, num_local_workers = get_partition()
        todo_shards, part_min_id, part_max_id = self.index.get_partition_shards_and_samples(
            device_id, num_devices, local_worker_id, num_local_workers, device_batch_size=self.device_batch_size)

        # print(
        #     f"{device_id=}, {num_devices=}, {local_worker_id=}, {num_local_workers=}, {todo_shards=}, {part_min_id=}, {part_max_id=}"
        # )

        # Preload all of our shards that are already available in the cache.
        todo_shards = self._load_shards_if_downloaded(todo_shards, part_min_id, part_max_id)

        # Start downloading our missing shards in a background thread, if there are any.
        if todo_shards:
            thread = Thread(target=self._download_thread, args=(todo_shards, part_min_id, part_max_id), daemon=True)
            thread.start()
        else:
            self._done_loading()

        # Iterate over the samples we have while the rest are begin loaded.
        for idx in self._iter_ids():
            yield self[idx]


class StreamingBatchPairDataset(StreamingDataset):

    def __init__(self,
                 remote: str,
                 local: str,
                 decoders: Dict[str, Optional[Callable]],
                 shuffle: bool,
                 transforms: Optional[Callable],
                 transform: Optional[Callable],
                 target_transform: Optional[Callable],
                 data_key: str = 'x',
                 target_key: str = 'y'):
        super().__init__(remote, local, decoders, shuffle)
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform
        self.data_key = data_key
        self.target_key = target_key

    @classmethod
    def split(cls,
              split: str,
              remote: str,
              local: str,
              decoders: Dict[str, Optional[Callable]],
              shuffle: bool,
              transforms: Optional[Callable],
              transform: Optional[Callable],
              target_transform: Optional[Callable],
              data_key: str = 'x',
              target_key: str = 'y'):
        remote = os.path.join(remote, split)
        local = os.path.join(local, split)
        return cls(remote, local, decoders, shuffle, transforms, transform, target_transform, data_key, target_key)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        obj = super().__getitem__(idx)
        data = obj[self.data_key]
        target = obj[self.target_key]
        if self.transforms is not None:
            assert self.transform is None
            assert self.target_transform is None
            data, target = self.transforms(data, target)
        else:
            if self.transform is not None:
                data = self.transform(data)
            if self.target_transform is not None:
                target = self.target_transform(target)
        return data, target