import numpy as np
import asyncio
import concurrent.futures
from core import BaseVRPTWGenerator
from typing import List, Dict, Any


class InstancePool:
    def __init__(
        self,
        generator: BaseVRPTWGenerator,
        pool_size: int = 128,
        refresh_threshold: int = 1000,
        sampling_memory: int = 20,
        max_workers: int = 8,
        chunk_size: int = 16,
        seed: int = None,
        no_refresh: bool = False,
        sequential: bool = False,
    ):
        self.generator = generator
        self.pool_size = pool_size
        self.refresh_threshold = refresh_threshold
        self.sampling_memory = sampling_memory
        self.seed = seed
        self.current_seed = seed
        self.no_refresh = no_refresh
        self.sequential = sequential
        self.chunk_size = chunk_size

        self._weights = np.ones(pool_size, dtype=np.float32)
        self.sample_counts = np.zeros(pool_size, dtype=np.int32)
        self.total_samples = 0
        self.recent_samples = []
        self.instances = []

        # Thread pool configuration
        self._max_workers = min(max_workers, self.pool_size)
        self._executor = None  # Lazy creation of thread pool

        self.initialize()

    def _create_executor(self):
        """Create new thread pool"""
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="InstanceGenerator"
            )
        return self._executor

    def _close_executor(self):
        """Safely close thread pool"""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    async def _async_generate_instance(self, index: int):
        """Asynchronously generate single instance"""
        try:
            executor = self._create_executor()  # Ensure thread pool exists
            if self.seed is not None:
                # print("generate_seed: ", self.current_seed + index)
                instance = await asyncio.get_event_loop().run_in_executor(
                    executor, self.generator.generate, self.current_seed + index
                )
            else:
                instance = await asyncio.get_event_loop().run_in_executor(
                    executor, self.generator.generate
                )
            return index, instance.get_data()
        except Exception as e:
            print(f"Error occurred when generating instance {index}: {e}")
            return index, None

    async def _process_chunk(self, chunk_indices: range):
        """Process instance generation for one batch"""
        tasks = [self._async_generate_instance(i) for i in chunk_indices]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, tuple) and result[1] is not None:
                valid_results.append(result)
            else:
                print(
                    f"Warning: Instance generation failed for index {chunk_indices[i]} in batch"
                )
        return valid_results

    async def _async_initialize(self):
        """Asynchronous initialization using chunking strategy"""
        try:
            chunks = [
                range(i, min(i + self.chunk_size, self.pool_size))
                for i in range(0, self.pool_size, self.chunk_size)
            ]

            all_results = []
            for i, chunk in enumerate(chunks):
                print(f"Processing batch {i + 1}/{len(chunks)}")
                chunk_results = await self._process_chunk(chunk)
                all_results.extend(chunk_results)

            if len(all_results) < self.pool_size * 0.8:
                print(
                    f"Warning: Only generated {len(all_results)}/{self.pool_size} valid instances"
                )

            return [r[1] for r in sorted(all_results, key=lambda x: x[0])]

        except Exception as e:
            print(f"Error occurred during initialization: {e}")
            raise

    def initialize(self):
        """Initialize or refresh instance pool"""
        try:
            self.instances.clear()

            # Ensure creating new thread pool
            self._create_executor()

            # Run asynchronous initialization
            loop = asyncio.get_event_loop()
            self.instances = loop.run_until_complete(self._async_initialize())

            # Update current seed
            if self.seed is not None:
                self.current_seed += self.pool_size

            # Reset counters
            self.sample_counts.fill(0)
            self.total_samples = 0
            self.recent_samples.clear()

            # Validate initialization results
            if len(self.instances) < self.pool_size:
                print(
                    f"Warning: Only successfully generated {len(self.instances)}/{self.pool_size} instances"
                )

        except Exception as e:
            print(f"Error occurred when initializing instance pool: {e}")
            raise
        finally:
            # Close thread pool after completion
            self._close_executor()

    def __del__(self):
        """Ensure proper resource release"""
        self._close_executor()

    def sample(self, batch_size: int = 1) -> List[dict]:
        """Uniform sampling of instances"""
        # Check if pool refresh is needed
        if not self.no_refresh and self.total_samples >= self.refresh_threshold:
            self.initialize()  # Refresh will automatically use new seed sequence

        if self.sequential:
            # Sequential sampling
            start_idx = self.total_samples % self.pool_size
            selected_indices = [
                (start_idx + i) % self.pool_size for i in range(batch_size)
            ]
            # print(selected_indices)
        else:
            # Calculate sampling weights
            np.divide(1.0, self.sample_counts + 1, out=self._weights)
            if self.recent_samples:
                self._weights[self.recent_samples] = 0
            if np.sum(self._weights) == 0:
                self._weights[:] = 1.0
                self.recent_samples.clear()
            self._weights /= np.sum(self._weights)
            # Sample instances
            selected_indices = np.random.choice(
                self.pool_size, size=batch_size, p=self._weights, replace=False
            )

        # Update counters and records
        self.sample_counts[selected_indices] += 1
        self.total_samples += batch_size

        # Update recent sampling records
        self.recent_samples.extend(selected_indices)
        self.recent_samples = self.recent_samples[-self.sampling_memory :]
        # print("Recent_samples: ", self.recent_samples)
        # Return selected instances
        if batch_size == 1:
            return self.instances[selected_indices[0]]
        else:
            return [self.instances[i] for i in selected_indices]

    def get_sampling_stats(self) -> Dict[str, Any]:
        """Return sampling statistics"""
        return {
            "total_samples": self.total_samples,
            "min_samples": np.min(self.sample_counts),
            "max_samples": np.max(self.sample_counts),
            "mean_samples": np.mean(self.sample_counts),
            "std_samples": np.std(self.sample_counts),
            "current_seed": self.current_seed if self.seed is not None else None,
        }
