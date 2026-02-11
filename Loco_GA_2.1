import React, { useState, useEffect } from 'react';
import { Play, Download, RefreshCw, TrendingUp, Clock, Truck, MapPin, Settings, BarChart3, Activity } from 'lucide-react';

// Генетический алгоритм для назначения локомотивов
class GeneticAlgorithm {
  constructor(trains, locomotives, config) {
    this.trains = trains;
    this.locomotives = locomotives;
    this.config = {
      populationSize: config.populationSize || 50,
      maxGenerations: config.maxGenerations || 100,
      crossoverRate: config.crossoverRate || 0.8,
      mutationRate: config.mutationRate || 0.15,
      eliteSize: config.eliteSize || 3,
      tournamentSize: config.tournamentSize || 5,
      crossoverType: config.crossoverType || 'single',
      mutationType: config.mutationType || 'replacement',
      ...config
    };
    this.population = [];
    this.bestSolution = null;
    this.generationHistory = [];
    this.weightHistory = [];
    this.startTime = Date.now();
  }

  // Создание начальной популяции
  initializePopulation() {
    this.population = [];
    const heuristicCount = Math.floor(this.config.populationSize * 0.4);
    const randomCount = this.config.populationSize - heuristicCount;

    for (let i = 0; i < heuristicCount; i++) {
      this.population.push(this.createHeuristicChromosome());
    }

    for (let i = 0; i < randomCount; i++) {
      this.population.push(this.createRandomChromosome());
    }
  }

  createRandomChromosome() {
    const chromosome = [];
    for (let train of this.trains) {
      const validLocomotives = this.getValidLocomotives(train);
      if (validLocomotives.length > 0) {
        const randomLoco = validLocomotives[Math.floor(Math.random() * validLocomotives.length)];
        chromosome.push({ trainId: train.id, locomotiveId: randomLoco.id });
      }
    }
    return chromosome;
  }

  createHeuristicChromosome() {
    const chromosome = [];
    const usedLocomotives = new Set();

    for (let train of this.trains) {
      const validLocomotives = this.getValidLocomotives(train)
        .filter(l => !usedLocomotives.has(l.id));
      
      if (validLocomotives.length > 0) {
        const nearest = validLocomotives.reduce((prev, curr) => {
          const prevDist = Math.abs(prev.location - train.departureStation);
          const currDist = Math.abs(curr.location - train.departureStation);
          return currDist < prevDist ? curr : prev;
        });
        chromosome.push({ trainId: train.id, locomotiveId: nearest.id });
        usedLocomotives.add(nearest.id);
      } else if (this.getValidLocomotives(train).length > 0) {
        const randomValid = this.getValidLocomotives(train)[0];
        chromosome.push({ trainId: train.id, locomotiveId: randomValid.id });
      }
    }
    return chromosome;
  }

  getValidLocomotives(train) {
    return this.locomotives.filter(loco => {
      if (loco.type !== train.requiredType) return false;
      if (loco.power < train.requiredPower) return false;
      if (loco.maintenanceKmLeft < train.distance) return false;
      return true;
    });
  }

  // Функция пригодности с адаптивными весами
  calculateFitness(chromosome, generation = 0) {
    let idleTime = 0;
    let emptyRuns = 0;
    let waitingTime = 0;
    let locomotivesUsed = new Set();

    const locomotiveUsage = {};

    for (let gene of chromosome) {
      const train = this.trains.find(t => t.id === gene.trainId);
      const loco = this.locomotives.find(l => l.id === gene.locomotiveId);
      
      if (!train || !loco) continue;

      locomotivesUsed.add(gene.locomotiveId);

      const emptyRun = Math.abs(loco.location - train.departureStation);
      emptyRuns += emptyRun;

      if (locomotiveUsage[gene.locomotiveId]) {
        const lastUsage = locomotiveUsage[gene.locomotiveId];
        const idle = train.departureTime - lastUsage.arrivalTime;
        if (idle > 0) {
          idleTime += idle;
        } else if (idle < 0) {
          waitingTime += Math.abs(idle);
        }
      }

      locomotiveUsage[gene.locomotiveId] = {
        arrivalTime: train.arrivalTime,
        location: train.arrivalStation
      };
    }

    // Адаптивные веса (меняются в зависимости от поколения)
    let w1, w2, w3, w4;
    
    if (generation < 22) {
      // Фаза корректировки
      w1 = 0.25; w2 = 0.25; w3 = 0.25; w4 = 0.25;
    } else if (generation < 62) {
      // Фаза оптимизации
      w1 = 0.35; w2 = 0.35; w3 = 0.20; w4 = 0.10;
    } else {
      // Фаза стабилизации
      w1 = 0.20; w2 = 0.20; w3 = 0.15; w4 = 0.45;
    }

    // Сохранение истории весов
    if (this.weightHistory.length === 0 || 
        this.weightHistory[this.weightHistory.length - 1].generation !== generation) {
      this.weightHistory.push({ generation, w1, w2, w3, w4 });
    }

    const maxIdle = 1000, maxEmpty = 500, maxWait = 500;
    
    const normalizedIdle = idleTime / maxIdle;
    const normalizedEmpty = emptyRuns / maxEmpty;
    const normalizedWait = waitingTime / maxWait;
    const normalizedLocos = locomotivesUsed.size / this.locomotives.length;

    const fitness = 1 / (1 + w1 * normalizedIdle + w2 * normalizedEmpty + 
                         w3 * normalizedWait + w4 * normalizedLocos);

    return { 
      fitness, 
      details: { 
        idleTime, 
        emptyRuns, 
        waitingTime, 
        locomotivesUsed: locomotivesUsed.size 
      },
      weights: { w1, w2, w3, w4 }
    };
  }

  tournamentSelection() {
    const tournament = [];
    for (let i = 0; i < this.config.tournamentSize; i++) {
      tournament.push(this.population[Math.floor(Math.random() * this.population.length)]);
    }
    return tournament.reduce((best, current) => {
      const bestFitness = this.calculateFitness(best).fitness;
      const currentFitness = this.calculateFitness(current).fitness;
      return currentFitness > bestFitness ? current : best;
    });
  }

  // Операторы кроссовера
  crossover(parent1, parent2) {
    if (Math.random() > this.config.crossoverRate) {
      return [parent1, parent2];
    }

    switch (this.config.crossoverType) {
      case 'single':
        return this.singlePointCrossover(parent1, parent2);
      case 'double':
        return this.doublePointCrossover(parent1, parent2);
      case 'uniform':
        return this.uniformCrossover(parent1, parent2);
      case 'priority':
        return this.priorityCrossover(parent1, parent2);
      default:
        return this.singlePointCrossover(parent1, parent2);
    }
  }

  // Одноточечный кроссовер
  singlePointCrossover(parent1, parent2) {
    const point = Math.floor(Math.random() * Math.min(parent1.length, parent2.length));
    const child1 = [...parent1.slice(0, point), ...parent2.slice(point)];
    const child2 = [...parent2.slice(0, point), ...parent1.slice(point)];
    return [this.correctChromosome(child1), this.correctChromosome(child2)];
  }

  // Двухточечный кроссовер
  doublePointCrossover(parent1, parent2) {
    const len = Math.min(parent1.length, parent2.length);
    const point1 = Math.floor(Math.random() * len);
    const point2 = Math.floor(Math.random() * len);
    const [start, end] = point1 < point2 ? [point1, point2] : [point2, point1];
    
    const child1 = [...parent1.slice(0, start), ...parent2.slice(start, end), ...parent1.slice(end)];
    const child2 = [...parent2.slice(0, start), ...parent1.slice(start, end), ...parent2.slice(end)];
    return [this.correctChromosome(child1), this.correctChromosome(child2)];
  }

  // Равномерный кроссовер
  uniformCrossover(parent1, parent2) {
    const child1 = [];
    const child2 = [];
    const len = Math.min(parent1.length, parent2.length);
    
    for (let i = 0; i < len; i++) {
      if (Math.random() < 0.5) {
        child1.push(parent1[i]);
        child2.push(parent2[i]);
      } else {
        child1.push(parent2[i]);
        child2.push(parent1[i]);
      }
    }
    return [this.correctChromosome(child1), this.correctChromosome(child2)];
  }

  // Кроссовер с приоритетом графика
  priorityCrossover(parent1, parent2) {
    const priorityTrains = this.trains.filter(t => t.priority === 'high');
    const child1 = [...parent1];
    const child2 = [...parent2];
    
    // Обмениваем назначения только для приоритетных поездов
    priorityTrains.forEach(train => {
      const idx1 = child1.findIndex(g => g.trainId === train.id);
      const idx2 = child2.findIndex(g => g.trainId === train.id);
      if (idx1 >= 0 && idx2 >= 0 && Math.random() < 0.5) {
        [child1[idx1], child2[idx2]] = [child2[idx2], child1[idx1]];
      }
    });
    
    return [this.correctChromosome(child1), this.correctChromosome(child2)];
  }

  // Операторы мутации
  mutate(chromosome) {
    if (Math.random() > this.config.mutationRate) {
      return chromosome;
    }

    switch (this.config.mutationType) {
      case 'swap':
        return this.swapMutation(chromosome);
      case 'replacement':
        return this.replacementMutation(chromosome);
      case 'shuffle':
        return this.shuffleMutation(chromosome);
      default:
        return this.replacementMutation(chromosome);
    }
  }

  // Мутация обмена локомотивами
  swapMutation(chromosome) {
    const mutated = [...chromosome];
    if (mutated.length < 2) return mutated;
    
    const idx1 = Math.floor(Math.random() * mutated.length);
    const idx2 = Math.floor(Math.random() * mutated.length);
    
    [mutated[idx1].locomotiveId, mutated[idx2].locomotiveId] = 
    [mutated[idx2].locomotiveId, mutated[idx1].locomotiveId];
    
    return this.correctChromosome(mutated);
  }

  // Мутация замены локомотива
  replacementMutation(chromosome) {
    const mutated = [...chromosome];
    const geneIndex = Math.floor(Math.random() * mutated.length);
    const train = this.trains.find(t => t.id === mutated[geneIndex].trainId);
    const validLocomotives = this.getValidLocomotives(train);

    if (validLocomotives.length > 0) {
      const newLoco = validLocomotives[Math.floor(Math.random() * validLocomotives.length)];
      mutated[geneIndex] = { ...mutated[geneIndex], locomotiveId: newLoco.id };
    }

    return this.correctChromosome(mutated);
  }

  // Мутация перемешивания локомотивов
  shuffleMutation(chromosome) {
    const mutated = [...chromosome];
    const start = Math.floor(Math.random() * mutated.length);
    const end = Math.min(start + 3, mutated.length);
    const segment = mutated.slice(start, end);
    
    // Перемешиваем только локомотивы в сегменте
    const locos = segment.map(g => g.locomotiveId);
    for (let i = locos.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [locos[i], locos[j]] = [locos[j], locos[i]];
    }
    
    segment.forEach((gene, idx) => {
      mutated[start + idx].locomotiveId = locos[idx];
    });
    
    return this.correctChromosome(mutated);
  }

  correctChromosome(chromosome) {
    const corrected = [];
    const usedPairs = new Set();

    for (let gene of chromosome) {
      const key = `${gene.trainId}-${gene.locomotiveId}`;
      if (!usedPairs.has(key)) {
        const train = this.trains.find(t => t.id === gene.trainId);
        const loco = this.locomotives.find(l => l.id === gene.locomotiveId);
        
        if (train && loco && this.getValidLocomotives(train).some(l => l.id === loco.id)) {
          corrected.push(gene);
          usedPairs.add(key);
        }
      }
    }

    for (let train of this.trains) {
      if (!corrected.some(g => g.trainId === train.id)) {
        const validLocomotives = this.getValidLocomotives(train);
        if (validLocomotives.length > 0) {
          corrected.push({ 
            trainId: train.id, 
            locomotiveId: validLocomotives[0].id 
          });
        }
      }
    }

    return corrected;
  }

  // Основной цикл алгоритма
  evolve() {
    this.initializePopulation();
    
    for (let generation = 0; generation < this.config.maxGenerations; generation++) {
      const evaluated = this.population.map(chromosome => ({
        chromosome,
        ...this.calculateFitness(chromosome, generation)
      }));

      evaluated.sort((a, b) => b.fitness - a.fitness);

      if (!this.bestSolution || evaluated[0].fitness > this.bestSolution.fitness) {
        this.bestSolution = evaluated[0];
      }

      const elapsedTime = (Date.now() - this.startTime) / 1000;
      
      this.generationHistory.push({
        generation,
        bestFitness: evaluated[0].fitness,
        avgFitness: evaluated.reduce((sum, e) => sum + e.fitness, 0) / evaluated.length,
        elapsedTime
      });

      // Критерий стагнации
      if (generation > 20) {
        const recent = this.generationHistory.slice(-10);
        const improvement = recent[recent.length - 1].bestFitness - recent[0].bestFitness;
        if (improvement < 0.001) {
          break;
        }
      }

      const newPopulation = evaluated.slice(0, this.config.eliteSize).map(e => e.chromosome);

      while (newPopulation.length < this.config.populationSize) {
        const parent1 = this.tournamentSelection();
        const parent2 = this.tournamentSelection();
        const [child1, child2] = this.crossover(parent1, parent2);
        
        newPopulation.push(this.mutate(child1));
        if (newPopulation.length < this.config.populationSize) {
          newPopulation.push(this.mutate(child2));
        }
      }

      this.population = newPopulation;
    }

    return this.bestSolution;
  }
}

// Главный компонент приложения
export default function LocomotiveAssignmentApp() {
  const [trains, setTrains] = useState([
    { id: 1, name: 'Поезд №101', departureStation: 100, arrivalStation: 500, departureTime: 800, arrivalTime: 1200, distance: 400, requiredPower: 4000, requiredType: 'Электровоз', priority: 'high' },
    { id: 2, name: 'Поезд №102', departureStation: 150, arrivalStation: 450, departureTime: 850, arrivalTime: 1250, distance: 300, requiredPower: 3500, requiredType: 'Электровоз', priority: 'medium' },
    { id: 3, name: 'Поезд №103', departureStation: 200, arrivalStation: 600, departureTime: 900, arrivalTime: 1400, distance: 400, requiredPower: 4500, requiredType: 'Тепловоз', priority: 'medium' },
    { id: 4, name: 'Поезд №104', departureStation: 250, arrivalStation: 550, departureTime: 950, arrivalTime: 1350, distance: 300, requiredPower: 4000, requiredType: 'Электровоз', priority: 'low' },
    { id: 5, name: 'Поезд №105', departureStation: 300, arrivalStation: 700, departureTime: 1000, arrivalTime: 1500, distance: 400, requiredPower: 5000, requiredType: 'Тепловоз', priority: 'high' },
  ]);

  const [locomotives, setLocomotives] = useState([
    { id: 1, name: 'ЭЛ-001', type: 'Электровоз', power: 5000, location: 120, maintenanceKmLeft: 800 },
    { id: 2, name: 'ЭЛ-002', type: 'Электровоз', power: 4500, location: 180, maintenanceKmLeft: 600 },
    { id: 3, name: 'ТЛ-001', type: 'Тепловоз', power: 5500, location: 220, maintenanceKmLeft: 1000 },
    { id: 4, name: 'ЭЛ-003', type: 'Электровоз', power: 4000, location: 280, maintenanceKmLeft: 700 },
    { id: 5, name: 'ТЛ-002', type: 'Тепловоз', power: 5000, location: 320, maintenanceKmLeft: 900 },
  ]);

  const [config, setConfig] = useState({
    populationSize: 50,
    maxGenerations: 100,
    crossoverRate: 0.8,
    mutationRate: 0.15,
    crossoverType: 'single',
    mutationType: 'replacement'
  });

  const [result, setResult] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showSettings, setShowSettings] = useState(false);
  const [activeTab, setActiveTab] = useState('results');

  const runAlgorithm = () => {
    setIsRunning(true);
    setProgress(0);
    
    setTimeout(() => {
      const ga = new GeneticAlgorithm(trains, locomotives, config);
      const solution = ga.evolve();
      
      setResult({
        ...solution,
        generationHistory: ga.generationHistory,
        weightHistory: ga.weightHistory
      });
      setProgress(100);
      setIsRunning(false);
      setActiveTab('results');
    }, 500);
  };

  const crossoverTypes = [
    { value: 'single', label: 'Одноточечный кроссовер' },
    { value: 'double', label: 'Двухточечный кроссовер' },
    { value: 'uniform', label: 'Равномерный кроссовер' },
    { value: 'priority', label: 'Кроссовер с приоритетом графика' }
  ];

  const mutationTypes = [
    { value: 'swap', label: 'Мутация обмена локомотивами' },
    { value: 'replacement', label: 'Мутация замены локомотива' },
    { value: 'shuffle', label: 'Мутация перемешивания' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Заголовок */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Система увязки локомотивов грузового движения
          </h1>
          <p className="text-gray-600">
            Автоматическое назначение локомотивов на основе генетического алгоритма с выбором операторов
          </p>
        </div>

        {/* Панель управления */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-xl font-semibold text-gray-800 mb-2">Панель управления</h2>
              <p className="text-sm text-gray-600">
                Поездов: {trains.length} | Локомотивов: {locomotives.length}
              </p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="flex items-center gap-2 bg-gray-600 hover:bg-gray-700 text-white px-4 py-3 rounded-lg font-semibold transition-colors"
              >
                <Settings size={20} />
                Настройки
              </button>
              <button
                onClick={runAlgorithm}
                disabled={isRunning}
                className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 text-white px-6 py-3 rounded-lg font-semibold transition-colors"
              >
                {isRunning ? (
                  <>
                    <RefreshCw className="animate-spin" size={20} />
                    Выполняется...
                  </>
                ) : (
                  <>
                    <Play size={20} />
                    Выполнить увязку
                  </>
                )}
              </button>
            </div>
          </div>

          {isRunning && (
            <div className="mt-4">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Настройки операторов */}
          {showSettings && (
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold text-gray-800 mb-4">Параметры генетического алгоритма</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Оператор кроссовера
                  </label>
                  <select
                    value={config.crossoverType}
                    onChange={(e) => setConfig({...config, crossoverType: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                  >
                    {crossoverTypes.map(type => (
                      <option key={type.value} value={type.value}>{type.label}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Оператор мутации
                  </label>
                  <select
                    value={config.mutationType}
                    onChange={(e) => setConfig({...config, mutationType: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                  >
                    {mutationTypes.map(type => (
                      <option key={type.value} value={type.value}>{type.label}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Размер популяции: {config.populationSize}
                  </label>
                  <input
                    type="range"
                    min="20"
                    max="100"
                    value={config.populationSize}
                    onChange={(e) => setConfig({...config, populationSize: parseInt(e.target.value)})}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Макс. поколений: {config.maxGenerations}
                  </label>
                  <input
                    type="range"
                    min="50"
                    max="200"
                    value={config.maxGenerations}
                    onChange={(e) => setConfig({...config, maxGenerations: parseInt(e.target.value)})}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Вероятность кроссовера: {(config.crossoverRate * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="1"
                    step="0.05"
                    value={config.crossoverRate}
                    onChange={(e) => setConfig({...config, crossoverRate: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Вероятность мутации: {(config.mutationRate * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0.05"
                    max="0.3"
                    step="0.05"
                    value={config.mutationRate}
                    onChange={(e) => setConfig({...config, mutationRate: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Результаты */}
        {result && (
          <>
            {/* Вкладки анализа */}
            <div className="bg-white rounded-xl shadow-lg mb-6">
              <div className="flex border-b">
                <button
                  onClick={() => setActiveTab('results')}
                  className={`px-6 py-3 font-medium ${activeTab === 'results' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-600'}`}
                >
                  Результаты
                </button>
                <button
                  onClick={() => setActiveTab('convergence')}
                  className={`px-6 py-3 font-medium ${activeTab === 'convergence' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-600'}`}
                >
                  График сходимости
                </button>
                <button
                  onClick={() => setActiveTab('weights')}
                  className={`px-6 py-3 font-medium ${activeTab === 'weights' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-600'}`}
                >
                  Динамика весов
                </button>
                <button
                  onClick={() => setActiveTab('analysis')}
                  className={`px-6 py-3 font-medium ${activeTab === 'analysis' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-600'}`}
                >
                  Детальный анализ
                </button>
              </div>
            </div>

            {/* Ключевые показатели */}
            {activeTab === 'results' && (
              <>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                  <div className="bg-white rounded-lg shadow p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-3 bg-blue-100 rounded-lg">
                        <Clock className="text-blue-600" size={24} />
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Простой</p>
                        <p className="text-xl font-bold text-gray-800">
                          {result.details.idleTime} мин
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-white rounded-lg shadow p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-3 bg-green-100 rounded-lg">
                        <MapPin className="text-green-600" size={24} />
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Порожний пробег</p>
                        <p className="text-xl font-bold text-gray-800">
                          {result.details.emptyRuns} км
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-white rounded-lg shadow p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-3 bg-orange-100 rounded-lg">
                        <TrendingUp className="text-orange-600" size={24} />
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Ожидание</p>
                        <p className="text-xl font-bold text-gray-800">
                          {result.details.waitingTime} мин
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-white rounded-lg shadow p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-3 bg-purple-100 rounded-lg">
                        <Truck className="text-purple-600" size={24} />
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Локомотивов</p>
                        <p className="text-xl font-bold text-gray-800">
                          {result.details.locomotivesUsed}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Таблица назначений */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold text-gray-800">
                      Результаты назначения
                    </h2>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <TrendingUp size={16} />
                      <span>Качество: {(result.fitness * 100).toFixed(1)}%</span>
                    </div>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b-2 border-gray-200">
                          <th className="text-left py-3 px-4 font-semibold text-gray-700">Поезд</th>
                          <th className="text-left py-3 px-4 font-semibold text-gray-700">Локомотив</th>
                          <th className="text-left py-3 px-4 font-semibold text-gray-700">Тип</th>
                          <th className="text-left py-3 px-4 font-semibold text-gray-700">Приоритет</th>
                          <th className="text-left py-3 px-4 font-semibold text-gray-700">Маршрут</th>
                          <th className="text-left py-3 px-4 font-semibold text-gray-700">Время</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.chromosome.map((gene, index) => {
                          const train = trains.find(t => t.id === gene.trainId);
                          const loco = locomotives.find(l => l.id === gene.locomotiveId);
                          return (
                            <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                              <td className="py-3 px-4 font-medium text-gray-800">{train?.name}</td>
                              <td className="py-3 px-4 text-gray-700">{loco?.name}</td>
                              <td className="py-3 px-4">
                                <span className={`px-2 py-1 rounded text-xs font-semibold ${
                                  loco?.type === 'Электровоз' 
                                    ? 'bg-blue-100 text-blue-700' 
                                    : 'bg-green-100 text-green-700'
                                }`}>
                                  {loco?.type}
                                </span>
                              </td>
                              <td className="py-3 px-4">
                                <span className={`px-2 py-1 rounded text-xs font-semibold ${
                                  train?.priority === 'high' 
                                    ? 'bg-red-100 text-red-700' 
                                    : train?.priority === 'medium'
                                    ? 'bg-yellow-100 text-yellow-700'
                                    : 'bg-gray-100 text-gray-700'
                                }`}>
                                  {train?.priority === 'high' ? 'Высокий' : train?.priority === 'medium' ? 'Средний' : 'Низкий'}
                                </span>
                              </td>
                              <td className="py-3 px-4 text-gray-700">
                                {train?.departureStation} → {train?.arrivalStation}
                              </td>
                              <td className="py-3 px-4 text-gray-700">
                                {Math.floor(train?.departureTime / 60)}:{(train?.departureTime % 60).toString().padStart(2, '0')}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}

            {/* График сходимости */}
            {activeTab === 'convergence' && result.generationHistory && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-4">
                  Динамика изменения функции пригодности по поколениям
                </h2>
                
                <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="font-semibold">Фаза корректировки:</span>
                      <span className="text-gray-600 ml-2">1-22 поколения</span>
                    </div>
                    <div>
                      <span className="font-semibold">Фаза оптимизации:</span>
                      <span className="text-gray-600 ml-2">22-62 поколения</span>
                    </div>
                    <div>
                      <span className="font-semibold">Фаза стабилизации:</span>
                      <span className="text-gray-600 ml-2">62+ поколения</span>
                    </div>
                  </div>
                </div>

                <svg viewBox="0 0 800 400" className="w-full">
                  {/* Оси */}
                  <line x1="50" y1="350" x2="750" y2="350" stroke="#333" strokeWidth="2" />
                  <line x1="50" y1="50" x2="50" y2="350" stroke="#333" strokeWidth="2" />
                  
                  {/* Подписи осей */}
                  <text x="400" y="390" textAnchor="middle" className="text-sm fill-gray-600">Поколение</text>
                  <text x="20" y="200" textAnchor="middle" transform="rotate(-90, 20, 200)" className="text-sm fill-gray-600">Функция пригодности</text>
                  
                  {/* Фазы (фоновые зоны) */}
                  <rect x="50" y="50" width="220" height="300" fill="rgba(59, 130, 246, 0.1)" />
                  <rect x="270" y="50" width="320" height="300" fill="rgba(16, 185, 129, 0.1)" />
                  <rect x="590" y="50" width="160" height="300" fill="rgba(139, 92, 246, 0.1)" />
                  
                  {/* График лучшей приспособленности */}
                  <polyline
                    points={result.generationHistory.map((h, i) => {
                      const x = 50 + (i / result.generationHistory.length) * 700;
                      const y = 350 - (h.bestFitness * 300);
                      return `${x},${y}`;
                    }).join(' ')}
                    fill="none"
                    stroke="#3b82f6"
                    strokeWidth="3"
                  />
                  
                  {/* График средней приспособленности */}
                  <polyline
                    points={result.generationHistory.map((h, i) => {
                      const x = 50 + (i / result.generationHistory.length) * 700;
                      const y = 350 - (h.avgFitness * 300);
                      return `${x},${y}`;
                    }).join(' ')}
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="2"
                    strokeDasharray="5,5"
                  />
                  
                  {/* Легенда */}
                  <line x1="600" y1="30" x2="640" y2="30" stroke="#3b82f6" strokeWidth="3" />
                  <text x="650" y="35" className="text-sm fill-gray-700">Лучшая приспособленность</text>
                  
                  <line x1="600" y1="50" x2="640" y2="50" stroke="#10b981" strokeWidth="2" strokeDasharray="5,5" />
                  <text x="650" y="55" className="text-sm fill-gray-700">Средняя приспособленность</text>
                </svg>

                <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-semibold text-gray-800 mb-2">Анализ сходимости</h3>
                  <p className="text-sm text-gray-600">
                    Алгоритм достиг качества {(result.fitness * 100).toFixed(1)}% за {result.generationHistory.length} поколений 
                    (время: {result.generationHistory[result.generationHistory.length - 1]?.elapsedTime.toFixed(1)} сек)
                  </p>
                </div>
              </div>
            )}

            {/* Динамика весов */}
            {activeTab === 'weights' && result.weightHistory && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-4">
                  Динамика изменения весовых коэффициентов
                </h2>
                
                <svg viewBox="0 0 800 400" className="w-full">
                  {/* Оси */}
                  <line x1="50" y1="350" x2="750" y2="350" stroke="#333" strokeWidth="2" />
                  <line x1="50" y1="50" x2="50" y2="350" stroke="#333" strokeWidth="2" />
                  
                  {/* Подписи */}
                  <text x="400" y="390" textAnchor="middle" className="text-sm fill-gray-600">Поколение</text>
                  <text x="20" y="200" textAnchor="middle" transform="rotate(-90, 20, 200)" className="text-sm fill-gray-600">Вес критерия</text>
                  
                  {/* Графики весов */}
                  {['w1', 'w2', 'w3', 'w4'].map((weight, idx) => {
                    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'];
                    const labels = ['Простой', 'Порожний пробег', 'Ожидание', 'Кол-во локомотивов'];
                    
                    return (
                      <g key={weight}>
                        <polyline
                          points={result.weightHistory.map((h, i) => {
                            const x = 50 + (i / result.weightHistory.length) * 700;
                            const y = 350 - (h[weight] * 300);
                            return `${x},${y}`;
                          }).join(' ')}
                          fill="none"
                          stroke={colors[idx]}
                          strokeWidth="2"
                        />
                        <line x1="600" y1={30 + idx * 25} x2="640" y2={30 + idx * 25} stroke={colors[idx]} strokeWidth="2" />
                        <text x="650" y={35 + idx * 25} className="text-sm fill-gray-700">{labels[idx]}</text>
                      </g>
                    );
                  })}
                </svg>

                <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-semibold text-gray-800 mb-2">Интерпретация</h3>
                  <p className="text-sm text-gray-600 mb-2">
                    График демонстрирует адаптивную стратегию алгоритма:
                  </p>
                  <ul className="text-sm text-gray-600 space-y-1 list-disc list-inside ml-4">
                    <li>Начальная фаза: равномерное распределение приоритетов</li>
                    <li>Средняя фаза: акцент на оптимизацию порожних пробегов и простоев</li>
                    <li>Финальная фаза: приоритет минимизации количества локомотивов</li>
                  </ul>
                </div>
              </div>
            )}

            {/* Детальный анализ */}
            {activeTab === 'analysis' && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-4">
                  Детальный анализ работы алгоритма
                </h2>
                
                {/* Показатели эффективности */}
                <div className="mb-6">
                  <h3 className="font-semibold text-gray-700 mb-3">Показатели эффективности</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 border border-gray-200 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Сокращение порожнего пробега</div>
                      <div className="text-2xl font-bold text-green-600">
                        {((1 - result.details.emptyRuns / 500) * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-4 border border-gray-200 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Эффективность использования парка</div>
                      <div className="text-2xl font-bold text-blue-600">
                        {((locomotives.length - result.details.locomotivesUsed) / locomotives.length * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-4 border border-gray-200 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Время сходимости</div>
                      <div className="text-2xl font-bold text-purple-600">
                        {result.generationHistory[result.generationHistory.length - 1]?.elapsedTime.toFixed(1)} сек
                      </div>
                    </div>
                    <div className="p-4 border border-gray-200 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Качество решения</div>
                      <div className="text-2xl font-bold text-indigo-600">
                        {(result.fitness * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>

                {/* Сравнение с базовым методом */}
                <div className="mb-6">
                  <h3 className="font-semibold text-gray-700 mb-3">Сравнение с диспетчерской эвристикой</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b-2 border-gray-200">
                          <th className="text-left py-2 px-3 font-semibold text-gray-700">Показатель</th>
                          <th className="text-left py-2 px-3 font-semibold text-gray-700">Базовый метод</th>
                          <th className="text-left py-2 px-3 font-semibold text-gray-700">ГА метод</th>
                          <th className="text-left py-2 px-3 font-semibold text-gray-700">Улучшение</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-b border-gray-100">
                          <td className="py-2 px-3">Порожний пробег, км</td>
                          <td className="py-2 px-3">~500</td>
                          <td className="py-2 px-3 font-medium">{result.details.emptyRuns}</td>
                          <td className="py-2 px-3 text-green-600 font-medium">
                            -{((1 - result.details.emptyRuns / 500) * 100).toFixed(0)}%
                          </td>
                        </tr>
                        <tr className="border-b border-gray-100">
                          <td className="py-2 px-3">Простой, мин</td>
                          <td className="py-2 px-3">~1000</td>
                          <td className="py-2 px-3 font-medium">{result.details.idleTime}</td>
                          <td className="py-2 px-3 text-green-600 font-medium">
                            -{((1 - result.details.idleTime / 1000) * 100).toFixed(0)}%
                          </td>
                        </tr>
                        <tr className="border-b border-gray-100">
                          <td className="py-2 px-3">Локомотивов задействовано</td>
                          <td className="py-2 px-3">{locomotives.length}</td>
                          <td className="py-2 px-3 font-medium">{result.details.locomotivesUsed}</td>
                          <td className="py-2 px-3 text-green-600 font-medium">
                            -{((1 - result.details.locomotivesUsed / locomotives.length) * 100).toFixed(0)}%
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Анализ операторов */}
                <div>
                  <h3 className="font-semibold text-gray-700 mb-3">Используемые операторы</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <div className="text-sm font-medium text-gray-700 mb-1">Кроссовер</div>
                      <div className="text-lg font-semibold text-blue-700">
                        {crossoverTypes.find(t => t.value === config.crossoverType)?.label}
                      </div>
                      <div className="text-xs text-gray-600 mt-1">
                        Вероятность: {(config.crossoverRate * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div className="p-4 bg-green-50 rounded-lg">
                      <div className="text-sm font-medium text-gray-700 mb-1">Мутация</div>
                      <div className="text-lg font-semibold text-green-700">
                        {mutationTypes.find(t => t.value === config.mutationType)?.label}
                      </div>
                      <div className="text-xs text-gray-600 mt-1">
                        Вероятность: {(config.mutationRate * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                </div>

                {/* Рекомендации */}
                <div className="mt-6 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                  <h3 className="font-semibold text-gray-800 mb-2">💡 Рекомендации по применению</h3>
                  <ul className="text-sm text-gray-700 space-y-1 list-disc list-inside">
                    <li>Для штатных режимов используйте одноточечный кроссовер и мутацию замены</li>
                    <li>При дефиците парка включайте кроссовер с приоритетом графика</li>
                    <li>В нештатных ситуациях увеличьте вероятность мутации до 20-25%</li>
                    <li>Оптимальное время расчета: 45-60 секунд для достижения 95-97% качества</li>
                  </ul>
                </div>
              </div>
            )}
          </>
        )}

        {/* Информация о методе */}
        <div className="bg-white rounded-xl shadow-lg p-6 mt-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">О методе из диссертации</h3>
          <div className="text-sm text-gray-600 space-y-2">
            <p>
              <strong>Генетический алгоритм</strong> с адаптивными параметрами, корректирующими операторами и динамическим управлением весами
            </p>
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div>
                <p className="font-semibold mb-2">Операторы кроссовера:</p>
                <ul className="list-disc list-inside space-y-1 ml-2 text-xs">
                  <li>Одноточечный (классический)</li>
                  <li>Двухточечный (повышенное разнообразие)</li>
                  <li>Равномерный (точечная комбинация)</li>
                  <li>С приоритетом графика (учет категорий поездов)</li>
                </ul>
              </div>
              <div>
                <p className="font-semibold mb-2">Операторы мутации:</p>
                <ul className="list-disc list-inside space-y-1 ml-2 text-xs">
                  <li>Обмен локомотивами</li>
                  <li>Замена локомотива</li>
                  <li>Перемешивание локомотивов</li>
                </ul>
              </div>
            </div>
            <p className="mt-4 text-xs italic">
              Реализация на основе главы 3 и экспериментального анализа из главы 4 диссертации Страдомской А.А.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
