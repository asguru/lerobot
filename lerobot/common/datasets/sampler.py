#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Iterator, Union

import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        episode_data_index: dict,
        episode_indices_to_use: Union[list, None] = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        num_replicas: int = 1,  # Add parameter for total number of processes
        rank: int = 0,          # Add parameter for current process rank
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            episode_data_index: Dictionary with keys 'from' and 'to' containing the start and end indices of each episode.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(episode_data_index["from"], episode_data_index["to"], strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(
                    range(start_index.item() + drop_n_first_frames, end_index.item() - drop_n_last_frames)
                )

        # Store distributed training parameters
        self.num_replicas = num_replicas
        self.rank = rank

        # Calculate number of samples per process
        self.num_samples = len(indices) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
        
        # Trim indices to make evenly divisible across processes
        indices = indices[:self.total_size]

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        # if self.shuffle:
        #     for i in torch.randperm(len(self.indices)):
        #         yield self.indices[i]
        # else:
        #     for i in self.indices:
        #         yield i
        
        if self.shuffle:
            # Create a deterministic shuffle that's the same across processes
            g = torch.Generator()
            g.manual_seed(torch.initial_seed())  # Use PyTorch's initial seed
            indices = torch.randperm(len(self.indices), generator=g).tolist()
            
            # Get this rank's subset of the shuffled indices
            indices = indices[self.rank:self.total_size:self.num_replicas]
            
            # Map back to original dataset indices
            for idx in indices:
                yield self.indices[idx]
        else:
            # If not shuffling, just take every num_replicas-th index starting from rank
            for i in range(self.rank, len(self.indices), self.num_replicas):
                yield self.indices[i]

    def __len__(self) -> int:
        # return len(self.indices)
        # Return number of samples this rank will see
        return self.num_samples
