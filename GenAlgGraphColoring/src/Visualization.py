from matplotlib import pyplot as plt

class Visualization:

    # Visualize the best fitness through generations
    @staticmethod
    def visualize(generation: int, fitness: list[int]) -> None:
        generations = list(range(1, generation + 1))

        plt.plot(generations, fitness)

        plt.xlabel("generation")
        plt.ylabel("best-fitness")

        plt.show()
