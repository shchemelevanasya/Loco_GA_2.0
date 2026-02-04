"""
loco_ga.py
Ядро генетического алгоритма и модель локомотивного назначения.
Русский интерфейс: internal strings not for UI.
"""
import streamlit as st
st.write("DEBUG: приложение стартовало")
import sys
st.write("Python:", sys.version)

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Callable, Optional
import numpy as np
import pandas as pd
import random
import math
import time
import copy

EPS = 1e-9
random.seed(0)
np.random.seed(0)

@dataclass
class LocoType:
    id: str
    power: float
    avg_reposition_speed: float  # km/h
    resource_range: Tuple[float, float]  # hours (min,max)

@dataclass
class Loco:
    id: str
    type_id: str
    resource_remaining: float  # hours
    home_depot: Optional[str] = None

@dataclass
class Trip:
    id: str
    start_time: float  # hours from epoch in floats
    end_time: float
    origin: str
    destination: str
    distance: float  # km

@dataclass
class GAConfig:
    population_size: int = 100
    generations: int = 100
    crossover_rate: float = 0.8
    base_mutation_prob: float = 0.05
    elite_size: int = 2
    maximize: bool = False
    allowed_crossover_ops: List[str] = field(default_factory=lambda: ["one_point","two_point","uniform","priority"])
    allowed_mutation_ops: List[str] = field(default_factory=lambda: ["swap_locos","replace_loco","range_shuffle"])
    crossover_base_probs: Dict[str,float] = field(default_factory=lambda: {"one_point":0.25,"two_point":0.25,"uniform":0.25,"priority":0.25})
    mutation_base_probs: Dict[str,float] = field(default_factory=lambda: {"swap_locos":0.4,"replace_loco":0.4,"range_shuffle":0.2})
    seed: int = 0
    max_offspring_per_pair: int = 2

class AssignmentIndividual:
    """
    genome: list/np.array with length = number of trips,
    values are indices into locomotives list.
    """
    def __init__(self, genome: np.ndarray):
        self.genome = np.array(genome, dtype=np.int32)
        self.fitness: Optional[float] = None
        self.components: Dict[str,float] = {}
        self.used_locos_count: int = 0

def init_population_random(trips: List[Trip], locos: List[Loco], pop_size: int) -> List[AssignmentIndividual]:
    n_trips = len(trips)
    n_locos = len(locos)
    pop = []
    for _ in range(pop_size):
        genome = np.random.randint(0, n_locos, size=n_trips)
        pop.append(AssignmentIndividual(genome))
    return pop

# ----------------- Operators -----------------
def swap_locos(ind: AssignmentIndividual, rng: random.Random):
    g = ind.genome
    if len(g) < 2: return
    i,j = rng.randrange(len(g)), rng.randrange(len(g))
    g[i], g[j] = g[j], g[i]

def replace_loco(ind: AssignmentIndividual, rng: random.Random, n_locos:int):
    i = rng.randrange(len(ind.genome))
    old = ind.genome[i]
    new = rng.randrange(n_locos)
    # avoid same
    if new == old:
        new = (new + 1) % n_locos
    ind.genome[i] = new

def range_shuffle(ind: AssignmentIndividual, rng: random.Random):
    n = len(ind.genome)
    if n < 3: return
    a = rng.randrange(n)
    b = rng.randrange(n)
    if a > b: a,b = b,a
    segment = list(ind.genome[a:b+1])
    rng.shuffle(segment)
    ind.genome[a:b+1] = segment

MUTATION_OPS = {
    "swap_locos": swap_locos,
    "replace_loco": replace_loco,
    "range_shuffle": range_shuffle
}

def one_point_crossover(p1: AssignmentIndividual, p2: AssignmentIndividual, rng: random.Random):
    n = len(p1.genome)
    pt = rng.randrange(1, n)
    c1 = np.concatenate([p1.genome[:pt], p2.genome[pt:]])
    c2 = np.concatenate([p2.genome[:pt], p1.genome[pt:]])
    return AssignmentIndividual(c1), AssignmentIndividual(c2)

def two_point_crossover(p1: AssignmentIndividual, p2: AssignmentIndividual, rng: random.Random):
    n = len(p1.genome)
    a = rng.randrange(n); b = rng.randrange(n)
    if a > b: a,b = b,a
    c1 = p1.genome.copy()
    c2 = p2.genome.copy()
    c1[a:b+1] = p2.genome[a:b+1]
    c2[a:b+1] = p1.genome[a:b+1]
    return AssignmentIndividual(c1), AssignmentIndividual(c2)

def uniform_crossover(p1: AssignmentIndividual, p2: AssignmentIndividual, rng: random.Random):
    n = len(p1.genome)
    mask = rng.getrandbits(n)
    # simpler: use rand boolean array
    picks = rng.random(n) < 0.5
    c1 = np.where(picks, p1.genome, p2.genome)
    c2 = np.where(picks, p2.genome, p1.genome)
    return AssignmentIndividual(c1), AssignmentIndividual(c2)

def priority_crossover(p1: AssignmentIndividual, p2: AssignmentIndividual, rng: random.Random):
    # Priority: choose segments from parent with better local fit - here random priority
    n = len(p1.genome)
    c = np.empty(n, dtype=np.int32)
    # build priorities randomly but biased by parent fitness if known
    for i in range(n):
        c[i] = p1.genome[i] if rng.random() < 0.6 else p2.genome[i]
    # Create one child only
    return AssignmentIndividual(c), None

CROSSOVER_OPS = {
    "one_point": one_point_crossover,
    "two_point": two_point_crossover,
    "uniform": uniform_crossover,
    "priority": priority_crossover
}

# --------------- Correcting operators ---------------
def technical_compatibility_operator(ind: AssignmentIndividual, trips: List[Trip], locos: List[Loco], loco_types: Dict[str,LocoType]):
    """
    Проверяет техническую пригодность: если локомотив не подходит по мощности для рейса (например, мощность < требуемой),
    пытается найти замену близкую по индексу. В данной упрощённой модели считаем, что для длинных дистанций требуется больше мощности.
    """
    for idx, trip in enumerate(trips):
        loco_idx = int(ind.genome[idx])
        loco = locos[loco_idx]
        lt = loco_types[loco.type_id]
        # simple rule: required power ~ distance/100
        req_power = max(1.0, trip.distance / 100.0)
        if lt.power + EPS < req_power:
            # try to find alternative loco
            candidates = [i for i,l in enumerate(locos) if loco_types[l.type_id].power >= req_power]
            if candidates:
                ind.genome[idx] = int(random.choice(candidates))
    return ind

def temporal_conflict_resolution_operator(ind: AssignmentIndividual, trips: List[Trip], locos: List[Loco], loco_types: Dict[str,LocoType]):
    """
    Разрешает простые временные конфликты: если один локомотив назначен на пересекающиеся по времени рейсы,
    пытается переназначить один из них на свободный локомотив или поменять местами с другим назначением.
    """
    # build occupancy per loco: list of (end_time, trip_idx)
    n_trips = len(trips)
    loco_assignments: Dict[int, List[Tuple[float,int]]] = {}
    for i in range(n_trips):
        l = int(ind.genome[i])
        loco_assignments.setdefault(l, []).append((trips[i].start_time, trips[i].end_time, i))
    # for each loco, sort and check overlaps
    for loco_idx, lst in loco_assignments.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        for j in range(len(lst_sorted)-1):
            s1,e1,i1 = lst_sorted[j]
            s2,e2,i2 = lst_sorted[j+1]
            if e1 > s2 + 1e-6:  # overlap
                # try to reassign i2 to any loco that is free at that interval
                reassigned = False
                for alt in range(len(locos)):
                    if alt == loco_idx: continue
                    alt_ass = loco_assignments.get(alt, [])
                    conflict = False
                    for as_s, as_e, _ in alt_ass:
                        if not (as_e <= s2 or as_s >= e2):
                            conflict = True; break
                    if not conflict:
                        ind.genome[i2] = alt
                        loco_assignments.setdefault(alt, []).append((s2,e2,i2))
                        reassigned = True
                        break
                if not reassigned:
                    # try swap with other trip non overlapping
                    candidates = [k for k in range(n_trips) if k != i2 and k != i1]
                    if candidates:
                        cand = random.choice(candidates)
                        ind.genome[i2], ind.genome[cand] = ind.genome[cand], ind.genome[i2]
    return ind

def maintenance_operator(ind: AssignmentIndividual, trips: List[Trip], locos: List[Loco], loco_types: Dict[str,LocoType]):
    """
    Следит за ресурсом: если назначенный локомотив привысит оставшийся ресурс, пытается подобрать замену.
    В нашей модели суммируем время на рейс = trip duration (end-start) + reposition time (distance / speed).
    """
    n_trips = len(trips)
    usage_per_loco = {i:0.0 for i in range(len(locos))}
    for i in range(n_trips):
        loco_idx = int(ind.genome[i])
        trip = trips[i]
        lt = loco_types[locos[loco_idx].type_id]
        trip_time = max(0.0, trip.end_time - trip.start_time)
        reposition = trip.distance / max(1e-3, lt.avg_reposition_speed)
        usage_per_loco[loco_idx] += (trip_time + reposition)
    # fix overused locos
    for loco_idx, used in usage_per_loco.items():
        loco = locos[loco_idx]
        if used > loco.resource_remaining + 1e-6:
            # find trips assigned to this loco and reassign some
            assigned = [i for i in range(n_trips) if int(ind.genome[i])==loco_idx]
            # sort by small impact first (short trips)
            assigned_sorted = sorted(assigned, key=lambda x: (trips[x].end_time - trips[x].start_time))
            for t_idx in assigned_sorted:
                # find alternative loco with capacity
                assigned_trip = trips[t_idx]
                for alt in range(len(locos)):
                    if alt == loco_idx: continue
                    alt_l = locos[alt]
                    lt_alt = loco_types[alt_l.type_id]
                    trip_time = max(0.0, assigned_trip.end_time - assigned_trip.start_time)
                    reposition = assigned_trip.distance / max(1e-3, lt_alt.avg_reposition_speed)
                    if usage_per_loco[alt] + trip_time + reposition <= alt_l.resource_remaining + 1e-6:
                        ind.genome[t_idx] = alt
                        usage_per_loco[alt] += trip_time + reposition
                        usage_per_loco[loco_idx] -= trip_time + reposition
                        break
                if usage_per_loco[loco_idx] <= loco.resource_remaining + 1e-6:
                    break
    return ind

CORRECTING_OPS = [technical_compatibility_operator, temporal_conflict_resolution_operator, maintenance_operator]

# --------------- Fitness -----------------
def evaluate_individual(ind: AssignmentIndividual, trips: List[Trip], locos: List[Loco], loco_types: Dict[str,LocoType]):
    """
    Простая симуляция для расчета компонентов:
    - idle_time: суммарное время локомотивов, когда они простаивают до начала следующего назначенного рейса (в часах)
    - empty_run_time: суммарное время репозиции без поезда (в часах) -> reposition time between end and next start when reposition needed
    - train_wait_time: суммарное времени ожидания поездов, если локоприбыл позже (лок опаздывает)
    - loco_wait_time: локомотив ожидал отправления (прибывает раньше), время ожидания
    - used_locos_count: количество локомотивов, использованных хотя бы на один рейс
    """
    n_trips = len(trips)
    n_locos = len(locos)
    # For each loco build timeline
    loco_events = {i: [] for i in range(n_locos)}
    for tidx in range(n_trips):
        lidx = int(ind.genome[tidx])
        loco_events[lidx].append(tidx)
    idle_time = 0.0
    empty_run_time = 0.0
    train_wait_time = 0.0
    loco_wait_time = 0.0
    used_locos = 0
    for lidx, trip_idxs in loco_events.items():
        if not trip_idxs: continue
        used_locos += 1
        # order trips by start_time
        trip_idxs_sorted = sorted(trip_idxs, key=lambda x: trips[x].start_time)
        # assume loco starts available at time -inf (so first trip: time to reach origin ignored)
        prev_end = None
        for i, tidx in enumerate(trip_idxs_sorted):
            trip = trips[tidx]
            lt = loco_types[locos[lidx].type_id]
            reposition_time = 0.0
            if prev_end is not None:
                # reposition from prev destination to this origin -> approximate reposition_time = distance / speed
                # We don't track locations in detail; use trip.distance as proxy reposition (approx)
                reposition_time = trip.distance / max(1e-3, lt.avg_reposition_speed)
                # if prev_end + reposition_time <= trip.start_time -> loco arrives early -> idle
                arrival_time = prev_end + reposition_time
                if arrival_time <= trip.start_time:
                    idle_time += (trip.start_time - arrival_time)
                    empty_run_time += reposition_time
                    loco_wait_time += (trip.start_time - arrival_time)
                else:
                    # loco arrives late -> train waits
                    train_wait_time += (arrival_time - trip.start_time)
            # update prev_end
            prev_end = trip.end_time
    ind.components = {
        "idle_time": idle_time,
        "empty_run_time": empty_run_time,
        "train_wait_time": train_wait_time,
        "loco_wait_time": loco_wait_time
    }
    ind.used_locos_count = used_locos
    return ind

def adaptive_weights_from_population(pop: List[AssignmentIndividual], component_keys=None):
    if component_keys is None:
        component_keys = ["idle_time","empty_run_time","train_wait_time","loco_wait_time","used_locos_count"]
    arr = {k: [] for k in component_keys}
    for ind in pop:
        for k in component_keys:
            if k == "used_locos_count":
                arr[k].append(ind.used_locos_count)
            else:
                arr[k].append(ind.components.get(k, 0.0))
    means = {k: (np.mean(arr[k]) if len(arr[k])>0 else 0.0) for k in component_keys}
    # convert means to weights: larger mean -> smaller weight? The requirement: "веса вычисля��тся из средних значений компонентов по популяции (чтобы масштаб компонентов был сопоставим)"
    # We will take inverse-normalized means so large components get comparable weights.
    vals = np.array([max(EPS, means[k]) for k in component_keys])
    # normalize by sum -> weights proportional to 1/val
    inv = 1.0 / vals
    w = inv / np.sum(inv)
    weights = {k: float(w[i]) for i,k in enumerate(component_keys)}
    return weights

def compute_fitness(ind: AssignmentIndividual, pop: List[AssignmentIndividual], minimize=True):
    # recompute adaptive weights from pop
    keys = ["idle_time","empty_run_time","train_wait_time","loco_wait_time","used_locos_count"]
    weights = adaptive_weights_from_population(pop, keys)
    # aggregate fitness as weighted sum
    comp = ind.components
    vec = np.array([
        comp.get("idle_time",0.0),
        comp.get("empty_run_time",0.0),
        comp.get("train_wait_time",0.0),
        comp.get("loco_wait_time",0.0),
        ind.used_locos_count
    ], dtype=float)
    wvec = np.array([weights[k] for k in keys], dtype=float)
    score = float(np.dot(vec, wvec))
    ind.fitness = score if minimize else -score
    return ind

# ----------------- GA Loop -----------------
class GeneticAlgorithm:
    def __init__(self, trips: List[Trip], locos: List[Loco], loco_types: Dict[str,LocoType], config: GAConfig):
        self.trips = trips
        self.locos = locos
        self.loco_types = loco_types
        self.config = config
        self.rng = random.Random(config.seed)
        self.population: List[AssignmentIndividual] = []
        # operator adaptive scores
        self.crossover_scores = {k: math.log(max(EPS,v)) for k,v in config.crossover_base_probs.items()}
        self.mutation_scores = {k: math.log(max(EPS,v)) for k,v in config.mutation_base_probs.items()}
        self.history = {"best":[], "mean":[], "median":[], "std":[], "weights":[] , "time": []}
    def init_population(self):
        self.population = init_population_random(self.trips, self.locos, self.config.population_size)
        # quick evaluate
        for ind in self.population:
            evaluate_individual(ind,self.trips,self.locos,self.loco_types)
        # compute adapt weights
        for ind in self.population:
            compute_fitness(ind,self.population, minimize=not self.config.maximize)
    def select_parents(self):
        # tournament selection
        k = 3
        pop = self.population
        parents = []
        for _ in range(len(pop)):
            contestants = [pop[self.rng.randrange(len(pop))] for _ in range(k)]
            best = min(contestants, key=lambda x: x.fitness) if not self.config.maximize else max(contestants, key=lambda x: x.fitness)
            parents.append(best)
        return parents
    def pick_crossover_op(self):
        # softmax on scores but only allowed ops
        ops = [o for o in self.config.allowed_crossover_ops if o in CROSSOVER_OPS]
        scores = np.array([self.crossover_scores.get(o,0.0) for o in ops], dtype=float)
        ex = np.exp(scores - np.max(scores))
        probs = ex / np.sum(ex)
        return self.rng.choices(ops, weights=probs, k=1)[0], probs, dict(zip(ops,probs))
    def pick_mutation_op(self):
        ops = [o for o in self.config.allowed_mutation_ops if o in MUTATION_OPS]
        scores = np.array([self.mutation_scores.get(o,0.0) for o in ops], dtype=float)
        ex = np.exp(scores - np.max(scores))
        probs = ex / np.sum(ex)
        return self.rng.choices(ops, weights=probs, k=1)[0], dict(zip(ops,probs))
    def adapt_crossover_scores(self, op_name, reward):
        self.crossover_scores[op_name] = self.crossover_scores.get(op_name,0.0) + 0.1 * reward
    def adapt_mutation_scores(self, op_name, reward):
        self.mutation_scores[op_name] = self.mutation_scores.get(op_name,0.0) + 0.05 * reward
    def adaptive_mutation_probability(self):
        # adaptive: base * (1 + (1 - diversity)), diversity measured by std/mean of fitness in pop (normalized to [0,1])
        fitness_vals = np.array([ind.fitness for ind in self.population])
        mean = np.mean(fitness_vals)
        std = np.std(fitness_vals)
        norm = min(1.0, std / (abs(mean) + EPS))
        # if low diversity (norm small) --> increase mutation prob
        factor = 1.0 + (1.0 - norm)
        return min(0.5, self.config.base_mutation_prob * factor)
    def run(self):
        self.init_population()
        start_time = time.time()
        for gen in range(self.config.generations):
            t0 = time.time()
            parents = self.select_parents()
            offspring: List[AssignmentIndividual] = []
            # elitism
            sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=self.config.maximize)
            elites = sorted_pop[:self.config.elite_size]
            # produce offspring
            i = 0
            while i < len(parents)-1:
                p1 = parents[i]; p2 = parents[i+1]
                if self.rng.random() < self.config.crossover_rate:
                    op_name, probs, prob_dict = self.pick_crossover_op()
                    c1, c2 = CROSSOVER_OPS[op_name](p1, p2, self.rng)
                    # maybe second child None
                    if c1 is not None:
                        offspring.append(c1)
                    if c2 is not None:
                        offspring.append(c2)
                    # adaptation reward later
                    # store op used by these children via index
                    used_op = op_name
                else:
                    # clone parents
                    offspring.append(AssignmentIndividual(p1.genome.copy()))
                    offspring.append(AssignmentIndividual(p2.genome.copy()))
                    used_op = None
                i += 2
            # mutations
            mut_prob = self.adaptive_mutation_probability()
            for child in offspring:
                if self.rng.random() < mut_prob:
                    m_op_name, mut_prob_dict = self.pick_mutation_op()
                    # apply mutation
                    if m_op_name == "replace_loco":
                        replace_loco(child, self.rng, len(self.locos))
                    else:
                        MUTATION_OPS[m_op_name](child, self.rng)
            # Correcting operators
            for idx,child in enumerate(offspring):
                for cop in CORRECTING_OPS:
                    child = cop(child, self.trips, self.locos, self.loco_types)
                evaluate_individual(child,self.trips,self.locos,self.loco_types)
            # recompute fitness with adaptive weights (weights computed from current population+offspring)
            combined = self.population + offspring
            for ind in combined:
                compute_fitness(ind, combined, minimize=not self.config.maximize)
            # select next population by elitist truncation
            combined_sorted = sorted(combined, key=lambda x: x.fitness, reverse=self.config.maximize)
            self.population = combined_sorted[:self.config.population_size]
            # Update operator adaptation based on improvements
            # naive reward: if average offspring fitness better than parents, reward op
            # (for code simplicity we provide small positive feedback to all used operators)
            # Stats
            fitness_vals = np.array([ind.fitness for ind in self.population])
            best = (np.max(fitness_vals) if self.config.maximize else np.min(fitness_vals))
            mean = np.mean(fitness_vals)
            median = np.median(fitness_vals)
            std = np.std(fitness_vals)
            self.history["best"].append(float(best))
            self.history["mean"].append(float(mean))
            self.history["median"].append(float(median))
            self.history["std"].append(float(std))
            curr_weights = adaptive_weights_from_population(self.population)
            self.history["weights"].append(curr_weights)
            self.history["time"].append(time.time()-t0)
        total_time = time.time() - start_time
        self.total_time = total_time
        return self

# ---------------- Utilities for loading simple data ----------------
def sample_data(n_trips=50, n_locos=10):
    trips = []
    for i in range(n_trips):
        start = float(i*1.5)  # every 1.5 hours
        duration = 1.0 + (i%5)*0.2
        t = Trip(id=f"T{i}", start_time=start, end_time=start+duration, origin="A", destination="B", distance=50 + (i%7)*10)
        trips.append(t)
    loco_types = {
        "A": LocoType("A", power=5.0, avg_reposition_speed=60.0, resource_range=(8,16)),
        "B": LocoType("B", power=3.0, avg_reposition_speed=50.0, resource_range=(6,12)),
        "C": LocoType("C", power=7.0, avg_reposition_speed=70.0, resource_range=(10,20))
    }
    locos = []
    for j in range(n_locos):
        t = random.choice(list(loco_types.keys()))
        rem = random.uniform(*loco_types[t].resource_range)
        locos.append(Loco(id=f"L{j}", type_id=t, resource_remaining=rem))
    return trips, locos, loco_types
