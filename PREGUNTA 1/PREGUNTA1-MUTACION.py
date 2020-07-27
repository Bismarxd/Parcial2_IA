
import array
import random
import wwarnings
warning.simplefilter('ignore')
import numpy 
import pandas as pd
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

#LEENDO EL ARCHIVO CSV
datos = pd.read_csv('Datos.csv')
df = pd.DataFrame(datos)
fits = []

#TIPOS
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
# GENERADOR DE ATRIBUTOS
toolbox.register("attr_bool", random.randint, 0, 1)
# ESTRUCTURA DE INCIALIZADORES
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#FUNCION PARA EVALUAR AL INDIVIDUO
def evalOneMax(individual):
   return sum(individual)

#OPERADORES
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint) 
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

'''def main():
    #random.seed(64)   
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=5, 
                                   stats=stats, halloffame=hof, verbose=False)
    
    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof = main()
    #print(pop)
    #print(log)
    print(hof)'''
    
    
def main():
    pop = toolbox.population(n=300)
    #EVALUAR TODA LA POBLACION
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
#CXPB LA PROBABILIDAD DE CRUZAR A DOS INDIVIDUOS
#MUTPB LA PROBABILIDAD DE MUTAR A UN INDIVIDUO
CXPB, MUTPB = 0,5, 0,2
#EXTRAER EL FITNESS DE 3 COLUMNAS DE NUESTRO DATASET
fits1 = datos['x']
fits2 = datos['y']
fits3 = datos['FFMC']
#VARIABLE PARA GUARDAR LA GENERACION
g = 0     

#COMENZANDO LA EVOLUCIÓN   
while max(fits) < 100 and g < 1000:
    #CREAR UN NUEVA GENERACION
    g = g + 1
    print("Generacion #i" %g)
    #SELECCIONAR INDIVIDUOS PARA LA NUEVA GENERACION
    offspring = toolbox.select(pop, len(pop))
    #clonar individuos para la nueva generacion
    offspring = list(map(toolbox.clone, offspring))
    #APLICAR CROSSOVER Y MUTAR A LA NUEVA GENERACIÓN
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
         if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
             
    for mutant in offspring:
        if random.random() < MUTB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
             
    #EVALUAR A LOS INDIVIDUOS CON UN FITNESS INVALIDO
    invalid_ind = [ind for ind in offspring in not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    pop[:] = offspring
    
    #CREAR LISTA PARA IMPRIMIR
    fits1 = [ind.fitness.values[0] for ind pop]
    fits2 = [ind.fitness.values[0] for ind pop]
    fits3 = [ind.fitness.values[0] for ind pop]
    
    length = len(pop)
    mean = sum (fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5
    
    print("MIN %s" % min(fits))
    print("MAX %s" % max(fits))
    print("AVG %s" % mean)
    print("STD %s" % std)
    print("INDV %s" % pop[:1]
    
    
    
    
