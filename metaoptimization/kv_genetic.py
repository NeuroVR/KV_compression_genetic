import copy
import math
import random
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, PretrainedConfig
from triton.runtime import driver as triton_driver


from kv_cache_size_measurement import measure_kv_cache_in_simple_evaluate

import lm_eval  

from combined_models.llama_combined import LlamaForCausalLM_MIXEDKV
from combined_models.qwen2_combined import Qwen2ForCausalLM_MIXEDKV
from combined_models.qwen3_combined import Qwen3ForCausalLM_MIXEDKV
from lm_eval.models.huggingface import HFLM


@dataclass
class KVTypeParamSpace:

    discrete: Dict[str, List[Any]] = field(default_factory=dict)
    continuous: Dict[str, Tuple[float, float, str]] = field(default_factory=dict)

    def sample(self) -> Dict[str, Any]:
        params = {}
        for name, values in self.discrete.items():
            if not values:
                continue
            params[name] = random.choice(values)
        for name, (low, high, ptype) in self.continuous.items():
            if ptype == "int":
                params[name] = random.randint(int(low), int(high))
            else:
                params[name] = random.uniform(float(low), float(high))
        return params

    def mutate(self, current: Dict[str, Any], mutation_prob: float) -> Dict[str, Any]:
        new_params = dict(current)
        for name, values in self.discrete.items():
            if not values:
                continue
            if random.random() < mutation_prob:
                new_params[name] = random.choice(values)
        for name, (low, high, ptype) in self.continuous.items():
            if random.random() < mutation_prob:
                if ptype == "int":
                    new_params[name] = random.randint(int(low), int(high))
                else:
                    new_params[name] = random.uniform(float(low), float(high))
        return new_params


@dataclass
class SearchSpace:

    num_layers: int
    algo_layer_lists: Dict[str, List[int]]
    kv_type_param_spaces: Dict[str, KVTypeParamSpace]

    def __post_init__(self):
        self.allowed_types_per_layer: Dict[int, List[str]] = {
            i: [] for i in range(self.num_layers)
        }
        for kv_type, layers in self.algo_layer_lists.items():
            for layer_idx in layers:
                if not (0 <= layer_idx < self.num_layers):
                    raise ValueError(
                        f"Layer index {layer_idx} for kv_type={kv_type} out of range 0..{self.num_layers-1}"
                    )
                self.allowed_types_per_layer[layer_idx].append(kv_type)

        for i in range(self.num_layers):
            if not self.allowed_types_per_layer[i]:
                self.allowed_types_per_layer[i] = ["none"]
                if "none" not in self.kv_type_param_spaces:
                    self.kv_type_param_spaces["none"] = KVTypeParamSpace()

    def random_type_for_layer(self, layer_idx: int) -> str:
        return random.choice(self.allowed_types_per_layer[layer_idx])

    def sample_layer_params(self, kv_type: str) -> Dict[str, Any]:
        ps = self.kv_type_param_spaces.get(kv_type, KVTypeParamSpace())
        return ps.sample()

    def mutate_layer_params(
        self, kv_type: str, current: Dict[str, Any], mutation_prob: float
    ) -> Dict[str, Any]:
        ps = self.kv_type_param_spaces.get(kv_type, KVTypeParamSpace())
        return ps.mutate(current, mutation_prob)

@dataclass
class Individual:
    layer_types: List[str]
    layer_configs: List[Dict[str, Any]]

    fitness: Optional[float] = None

    quality: Optional[float] = None     
    kv_memory: Optional[float] = None   

    def clone(self) -> "Individual":
        return Individual(
            layer_types=list(self.layer_types),
            layer_configs=[dict(c) for c in self.layer_configs],
            fitness=self.fitness,
            quality=self.quality,
            kv_memory=self.kv_memory,
        )

class GeneticKVSearch:


    def __init__(
        self,
        model_cls,                       
        base_model_name_or_path: str,    
        base_config: PretrainedConfig,   
        tokenizer: PreTrainedTokenizerBase,
        eval_dataset: Dataset,          
        calibration_dataset: Dataset,   
        text_field: str = "text",

        search_space=None,          

        population_size: int = 16,
        num_generations: int = 10,
        elite_fraction: float = 0.2,
        mutation_prob: float = 0.1,
        crossover_prob: float = 0.5,

        max_eval_samples: int = 100,
        max_seq_len: int = 512,
        batch_size: int = 4,

        faiss_candidate_layers: Optional[List[int]] = None,
        faiss_m: Optional[int] = None,
        faiss_nbits: int = 8,
        faiss_quantize_k: bool = True,
        faiss_quantize_v: bool = False,

        device_ids: Optional[List[int]] = None,  
        seed: int = 42,

        eval_args: Dict[str, Any] = None,
        quality_metric_name: str = "exact_match,strict-match",
        fitness_a: float = 1.0,
        fitness_b: float = 1.0,

        save_best: bool = True,
        save_dir: str = ".",
    ):
        self.model_cls = model_cls
        self.base_model_name_or_path = base_model_name_or_path
        self.base_config = copy.deepcopy(base_config)
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.calibration_dataset = calibration_dataset
        self.text_field = text_field

        self.search_space = search_space
        
        self.population_size = population_size
        self.num_generations = num_generations
        self.elite_fraction = elite_fraction
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob

        self.max_eval_samples = max_eval_samples
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        self.faiss_candidate_layers = faiss_candidate_layers or []
        self.faiss_m = faiss_m
        self.faiss_nbits = faiss_nbits
        self.faiss_quantize_k = faiss_quantize_k
        self.faiss_quantize_v = faiss_quantize_v

        if device_ids is None:
            if torch.cuda.is_available():
                device_ids = list(range(torch.cuda.device_count()))
            else:
                device_ids = [None] 
        self.device_ids = device_ids
        random.seed(seed)
        torch.manual_seed(seed)

        lm_eval.simple_evaluate = measure_kv_cache_in_simple_evaluate(lm_eval.simple_evaluate)  

        if eval_args is None:
            raise ValueError("eval_args must be provided (dict for lm_eval.simple_evaluate).")
        self.eval_args = copy.deepcopy(eval_args)
        self.quality_metric_name = quality_metric_name
        self.fitness_a = fitness_a
        self.fitness_b = fitness_b

        self.save_best = save_best
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

        self.faiss_template_model = None
        if self.faiss_candidate_layers:
            self._prepare_faiss_template_model()

    def _prepare_faiss_template_model(self):
        print("[GA] Building FAISS template model...")

        device0 = (
            torch.device(f"cuda:{self.device_ids[0]}")
            if self.device_ids[0] is not None
            else torch.device("cpu")
        )

        num_layers = self.base_config.num_hidden_layers

        faiss_config = copy.deepcopy(self.base_config)
        layer_kv_types = ["none"] * num_layers
        layer_kv_configs = [{} for _ in range(num_layers)]

        for idx in self.faiss_candidate_layers:
            layer_kv_types[idx] = "faiss"

        setattr(faiss_config, "layer_kv_types", layer_kv_types)
        setattr(faiss_config, "layer_kv_configs", layer_kv_configs)

        model = self.model_cls.from_pretrained(
            self.base_model_name_or_path,
            config=faiss_config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        model.to(device0)
        model.eval()
        torch.cuda.set_device(device0)

        model.build_faiss_indices(
            tokenizer=self.tokenizer,
            dataset=self.calibration_dataset,
            num_samples=self.max_eval_samples,
            max_seq_len=self.max_seq_len,
            num_quantize_layers=len(self.faiss_candidate_layers),
            target_layers=self.faiss_candidate_layers,
            m=self.faiss_m,
            nbits=self.faiss_nbits,
            text_field=self.text_field,
            device=device0,
            quantize_k=self.faiss_quantize_k,
            quantize_v=self.faiss_quantize_v,
        )

        model.to("cpu")
        torch.cuda.empty_cache()
        self.faiss_template_model = model

        print("[GA] FAISS template model is ready.")

    def _random_individual(self) -> Individual:
        num_layers = self.base_config.num_hidden_layers
        layer_types = []
        layer_configs = []

        for i in range(num_layers):
            kv_type = self.search_space.random_type_for_layer(i)
            params = self.search_space.sample_layer_params(kv_type)
            layer_types.append(kv_type)
            layer_configs.append(params)

        return Individual(layer_types=layer_types, layer_configs=layer_configs)

    def _mutate(self, ind: Individual) -> Individual:
        num_layers = len(ind.layer_types)
        new_ind = ind.clone()

        for i in range(num_layers):
            if random.random() < self.mutation_prob:
                allowed = self.search_space.allowed_types_per_layer[i]
                if len(allowed) > 1:
                    new_type = random.choice(allowed)
                    new_ind.layer_types[i] = new_type
                    new_ind.layer_configs[i] = self.search_space.sample_layer_params(
                        new_type
                    )

            cur_type = new_ind.layer_types[i]
            cur_cfg = new_ind.layer_configs[i]
            new_ind.layer_configs[i] = self.search_space.mutate_layer_params(
                cur_type, cur_cfg, self.mutation_prob
            )

        new_ind.fitness = None
        new_ind.quality = None
        new_ind.kv_memory = None
        return new_ind

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        num_layers = len(parent1.layer_types)
        child_types = []
        child_configs = []

        for i in range(num_layers):
            if random.random() < 0.5:
                child_types.append(parent1.layer_types[i])
                child_configs.append(dict(parent1.layer_configs[i]))
            else:
                child_types.append(parent2.layer_types[i])
                child_configs.append(dict(parent2.layer_configs[i]))

        return Individual(layer_types=child_types, layer_configs=child_configs)

    def _build_model_for_individual(
        self, ind: Individual, device: torch.device
    ) -> torch.nn.Module:
        config = copy.deepcopy(self.base_config)
        
        setattr(config, "layer_kv_types", ind.layer_types)
        setattr(config, "layer_kv_configs", ind.layer_configs)
        torch.cuda.set_device(device)
        dev_idx = device.index

        cuda_driver = triton_driver.active
        if hasattr(cuda_driver, "set_current_device"):
            cuda_driver.set_current_device(dev_idx)
        elif hasattr(cuda_driver, "set_device"):
            cuda_driver.set_device(dev_idx)
        else:
            raise RuntimeError("Не нашёл способ переключить устройство в Triton driver")

        model = self.model_cls.from_pretrained(
            self.base_model_name_or_path, config=config,low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        model.to(device)
        model.eval()


        if self.faiss_template_model is not None:
            for layer_idx, kv_type in enumerate(ind.layer_types):
                if kv_type == "faiss":
                    src_attn = self.faiss_template_model.model.layers[
                        layer_idx
                    ].self_attn
                    dst_attn = copy.deepcopy(src_attn)

                    cfg = ind.layer_configs[layer_idx]
                    for k, v in cfg.items():
                        if hasattr(dst_attn, k):
                            setattr(dst_attn, k, v)

                    model.model.layers[layer_idx].self_attn = dst_attn.to(device)

        return model

    def _extract_kv_cache_memory(self, eval_result: Dict[str, Any]) -> float:
    
        for key in ["kv_cache_peak_gb", "kv_cache_size_bytes", "kv_cache_memory_bytes"]:
            if key in eval_result:
                return float(eval_result[key])

        if "kv_cache" in eval_result and isinstance(eval_result["kv_cache"], dict):
            for nested_key in ["max_bytes", "max", "value", "bytes"]:
                if nested_key in eval_result["kv_cache"]:
                    return float(eval_result["kv_cache"][nested_key])

        raise KeyError(
            "Не удалось найти информацию о размере KV‑кеша в eval_result. "
            "Проверьте реализацию measure_kv_cache_in_simple_evaluate и "
            "обновите _extract_kv_cache_memory."
        )

    @torch.no_grad()
    def _evaluate_individual_lm_eval(
        self, ind: Individual, device_id: Optional[int]
    ) -> Tuple[float, float]:
        device = (
            torch.device(f"cuda:{device_id}")
            if device_id is not None
            else torch.device("cpu")
        )

        model = self._build_model_for_individual(ind, device=device)

        eval_args_local = copy.deepcopy(self.eval_args)
        eval_args_local["model"] = HFLM(pretrained=model,batch_size=self.batch_size)

        eval_args_local["device"] = str(device)

        try:
            eval_result = lm_eval.simple_evaluate(**eval_args_local)
        finally:
            del model
            torch.cuda.empty_cache()

        task_name = eval_args_local["tasks"][0]
        if task_name not in eval_result.get('results', {}):
            raise KeyError(
                f"В eval_result нет ключа таска '{task_name}'. "
                "Проверьте версию lm_eval/simple_evaluate и формат результата."
            )

        task_metrics = eval_result.get('results', {})[task_name]
        if self.quality_metric_name not in task_metrics:
            raise KeyError(
                f"В eval_result['{task_name}'] нет метрики "
                f"'{self.quality_metric_name}'. Доступные ключи: {list(task_metrics.keys())}"
            )

        quality = float(task_metrics[self.quality_metric_name])
        kv_memory = self._extract_kv_cache_memory(eval_result)

        return quality, kv_memory

    def _save_best_individual(self, ind: Individual, generation: int):
     
        best_info = {
            "generation": generation,
            "fitness": ind.fitness,
            "quality": ind.quality,
            "kv_memory": ind.kv_memory,
            "layer_types": ind.layer_types,
            "layer_configs": ind.layer_configs,
        }

        individual_path = os.path.join(self.save_dir, "best_kv_individual.json")
        with open(individual_path, "w", encoding="utf-8") as f:
            json.dump(best_info, f, ensure_ascii=False, indent=2)

        kv_config = {
            str(i): {
                "kv_type": kv_type,
                "config": ind.layer_configs[i],
            }
            for i, kv_type in enumerate(ind.layer_types)
        }
        kv_config_path = os.path.join(self.save_dir, "best_kv_config.json")
        with open(kv_config_path, "w", encoding="utf-8") as f:
            json.dump(kv_config, f, ensure_ascii=False, indent=2)

        print(f"[GA] Best individual saved to {individual_path} and {kv_config_path}")

    def run(self) -> List[Individual]:
        population: List[Individual] = [
            self._random_individual() for _ in range(self.population_size)
        ]

        num_elites = max(1, int(self.elite_fraction * self.population_size))

        best_overall: Optional[Individual] = None
        best_generation: Optional[int] = None

        for gen in range(self.num_generations):
            print(f"\n[GA] Generation {gen + 1}/{self.num_generations}")

            for idx, ind in enumerate(population):
                if ind.fitness is not None and ind.quality is not None and ind.kv_memory is not None:
                    continue

                device_id = self.device_ids[idx % len(self.device_ids)]
                q, m = self._evaluate_individual_lm_eval(ind, device_id=device_id)
                ind.quality = q
                ind.kv_memory = m

                print(f"  Individual {idx}: quality={q:.4f}, kv_memory={m:.4f}")

            qualities = [ind.quality for ind in population if ind.quality is not None]
            memories = [ind.kv_memory for ind in population if ind.kv_memory is not None]

            if not qualities or not memories:
                raise RuntimeError("Не удалось вычислить качества или памяти для поколения.")

            qmax = max(qualities)
            mmin = min(memories)

            for idx, ind in enumerate(population):
                q = ind.quality
                m = ind.kv_memory

                if q is None or m is None:
                    ind.fitness = 0.0
                    continue

                if qmax <= 0 or m <= 0:
                    ind.fitness = 0.0
                else:
                    norm_q = q / qmax
                    norm_m = mmin / m
                    fitness = (norm_q ** self.fitness_a) * (norm_m ** self.fitness_b)
                    ind.fitness = float(fitness)

                print(
                    f"  Individual {idx}: fitness={ind.fitness:.4f} "
                    f"(q={q:.4f}, qmax={qmax:.4f}, m={m:.4f}, mmin={mmin:.4f})"
                )

            population.sort(
                key=lambda x: x.fitness if x.fitness is not None else 0.0,
                reverse=True,
            )

            if best_overall is None or (
                population[0].fitness is not None
            ):
                best_overall = population[0].clone()
                best_generation = gen
                print(
                    f"[GA] New best: gen={gen + 1}, "
                    f"fitness={best_overall.fitness:.4f}, "
                    f"quality={best_overall.quality:.4f}, "
                    f"kv_memory={best_overall.kv_memory:.4f} GB"
                )
                if self.save_best:
                    self._save_best_individual(best_overall, generation=gen + 1)

            elites = [ind.clone() for ind in population[:num_elites]]
            new_population: List[Individual] = elites

            fitness_scores = [
                max(ind.fitness if ind.fitness is not None else 0.0, 1e-12)
                for ind in population
            ]
            total_score = sum(fitness_scores)

            def select_parent() -> Individual:
                r = random.random() * total_score
                acc = 0.0
                for ind, s in zip(population, fitness_scores):
                    acc += s
                    if acc >= r:
                        return ind
                return population[-1]

            while len(new_population) < self.population_size:
                parent1 = select_parent()
                parent2 = select_parent()

                if random.random() < self.crossover_prob:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.clone()

                child = self._mutate(child)
                new_population.append(child)

            population = new_population

        population.sort(
            key=lambda x: x.fitness if x.fitness is not None else 0.0,
            reverse=True,
        )

        print("\n[GA] Finished.")
        if best_overall is not None and best_generation is not None:
            print(
                f"[GA] Best overall: gen={best_generation + 1}, "
                f"fitness={best_overall.fitness:.4f}, "
                f"quality={best_overall.quality:.4f}, "
                f"kv_memory={best_overall.kv_memory:.4f}"
            )
        device0 = (
            torch.device(f"cuda:{self.device_ids[0]}")
            if self.device_ids[0] is not None
            else torch.device("cpu")
        )
        torch.cuda.set_device(device0)

        return population
