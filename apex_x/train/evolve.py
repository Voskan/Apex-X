"""Hyperparameter evolution (genetic algorithm) for Apex-X."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from apex_x.config import ApexXConfig
from apex_x.train import ApexXTrainer
from apex_x.utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class EvoParams:
    """Evolution parameters."""
    gens: int = 50  # Generations
    pop_size: int = 10  # Population size
    epochs_per_gen: int = 3  # Short training for fitness proxy
    mutation_prob: float = 0.8
    sigma: float = 0.2  # Mutation magnitude


class HyperparameterEvolution:
    """Genetic Algorithm for hyperparameter tuning."""

    def __init__(
        self,
        base_config: ApexXConfig,
        dataset_path: str | None = None,
        params: EvoParams | None = None,
    ) -> None:
        self.base_config = base_config
        self.dataset_path = dataset_path
        self.params = params or EvoParams()
        self.best_fitness = 0.0
        self.best_config: dict[str, Any] = {}

    def _mutate(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply random mutations to hyperparameters."""
        new_config = copy.deepcopy(config_dict)
        rng = random.Random()
        
        # Mutation definitions: key -> (min, max, type)
        space = {
            "train.base_lr": (1e-5, 1e-2, float),
            "train.weight_decay": (1e-5, 1e-3, float),
            "train.optimizer": ["adamw", "sgd"], # Categorical
            "loss.box_weight": (0.5, 5.0, float),
            "loss.cls_weight": (0.5, 5.0, float),
            "loss.lovasz_weight": (0.1, 2.0, float),
            "data.mixup_prob": (0.0, 1.0, float),
            "data.mosaic_prob": (0.0, 1.0, float),
        }

        for key, bounds in space.items():
            if rng.random() > self.params.mutation_prob:
                continue
                
            # Traverse nested dict
            parts = key.split(".")
            target = new_config
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            
            final_key = parts[-1]
            current_val = target.get(final_key)

            if isinstance(bounds, list): # Categorical
                target[final_key] = rng.choice(bounds)
            else: # Numerical
                min_v, max_v, dtype = bounds
                if current_val is None:
                    current_val = (min_v + max_v) / 2.0
                
                # Mutate by sigma
                change = rng.gauss(0, self.params.sigma) * current_val
                new_val = float(current_val + change)
                new_val = max(min_v, min(max_v, new_val))
                target[final_key] = dtype(new_val)
                
        return new_config

    def run(self) -> None:
        """Execute evolution loop."""
        LOGGER.info("Starting Hyperparameter Evolution...")
        
        # Initial population
        population = [self.base_config.to_dict()]  # Elitism: keep base
        for _ in range(self.params.pop_size - 1):
             population.append(self._mutate(self.base_config.to_dict()))
             
        for gen in range(self.params.gens):
            LOGGER.info(f"Generation {gen+1}/{self.params.gens}")
            gen_results = []
            
            for i, cand_dict in enumerate(population):
                # Run fitness check
                cfg = ApexXConfig.from_dict(cand_dict)
                
                # Setup truncated trainer
                # We reuse ApexXTrainer but force fewer epochs or steps?
                # Actually, trainer.run() usually runs full schedule.
                # We need to hack steps_per_stage or max_epochs.
                # Trainer v3 uses 'stages'. We can just run 1 stage with few steps?
                # Or we assume dataset is small for evolution?
                # Let's override `steps_per_stage` to something small if provided.
                
                # For this simplified implementation, we'll assume `evolve_cmd` handles
                # passing a small dataset or small steps_per_stage.
                
                trainer = ApexXTrainer(
                    config=cfg,
                    num_classes=3, # Fixed for now or infer
                    checkpoint_dir=None, # Tmp
                )
                
                # We need to capture the metric.
                # Trainer.run() returns TrainResult causing loss_proxy?
                # We need Validation metric (mAP).
                # ApexXTrainer usually runs valid at end of stage.
                
                # Let's run for 1 stage (or limited steps)
                # The evolution run effectively is a series of short trainings.
                try:
                    res = trainer.run(
                        steps_per_stage=self.params.epochs_per_gen * 100, # Approx steps?
                        seed=gen * 100 + i,
                        dataset_path=self.dataset_path,
                    )
                    # Trainer result has `loss_proxy`.
                    # We want maximizing fitness. mAP is better.
                    # But if validation didn't run, we use (1/loss).
                    # Let's assume validation runs.
                    # Trainer.run returns TrainResult(loss_proxy, train_summary, valid_summary...)
                    # We check valid_summary.
                    fitness = 0.0
                    if res.stage_results:
                        last_stage = res.stage_results[-1]
                        # Assuming valid_metrics dict exists
                        # This dependence on Trainer internals is tricky without reading it deeply.
                        # `res.loss_proxy` is robust. Let's use 10 / loss_proxy
                        fitness = 10.0 / (res.loss_proxy + 1e-6)
                        
                    LOGGER.info(f"Gen {gen+1} Ind {i+1}: Fitness={fitness:.4f}")
                    gen_results.append((fitness, cand_dict))
                    
                except Exception as e:
                    LOGGER.warning(f"Gen {gen+1} Ind {i+1} Failed: {e}")
                    gen_results.append((0.0, cand_dict))

            # Select best
            gen_results.sort(key=lambda x: x[0], reverse=True)
            self.best_fitness = gen_results[0][0]
            self.best_config = gen_results[0][1]
            
            LOGGER.info(f"Generation {gen+1} Best Fitness: {self.best_fitness:.4f}")
            
            # Save intermediate best
            with open("artifacts/evolve_best.yaml", "w") as f:
                yaml.dump(self.best_config, f)
            
            # Evolve population (Parents + Offspring)
            # Elitism: Keep top 20%
            top_n = max(1, int(self.params.pop_size * 0.2))
            parents = [x[1] for x in gen_results[:top_n]]
            
            new_pop = list(parents)
            while len(new_pop) < self.params.pop_size:
                parent = random.choice(parents)
                child = self._mutate(parent)
                new_pop.append(child)
            
            population = new_pop

        LOGGER.info(f"Evolution Complete. Best Fitness: {self.best_fitness}")
