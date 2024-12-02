from matplotlib import pyplot as plt


class Visualization:

    # Visualize the best fitness through generations
    @staticmethod
    def visualize(
            generation: int,
            fitness: list[int],
            numberOfColors: int
    ) -> None:

        generations = list(range(1, generation + 1))

        plt.plot(generations, fitness)

        plt.title(f"{numberOfColors} colors in {generation} generations")
        plt.xlabel("generation")
        plt.ylabel("best-fitness")

        plt.show()
