import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, set1, set2):
        self.set1 = set1
        self.set2 = set2
        self.plot = None
        self.create_chart()

    def create_chart(self):
        [x1, y1] = zip(*self.set1)
        [x2, y2] = zip(*self.set2)
        plt.figure(figsize=(6, 6))
        plt.scatter(x1, y1, marker='x', color='blue')
        plt.scatter(x2, y2, marker='o', color='red')
        self.plot = plt

    def show(self):
        self.plot.show()
        self.plot.clf()

    def save(self, filename):
        self.plot.savefig(f'./outputs/{filename}.png')
        self.plot.clf()


def get_dataset():
    set1 = [(1, 1), (1, 2), (1, 3.5), (1.2, 5.5), (1.3, 6.5), (0.5, 7.5), (0.5, 9), (2, 0.5), (2, 2), (2, 3), (2, 4),
            (2.2, 6), (2.3, 7), (1.5, 8), (2.5, 9), (3, 1), (3, 2.5), (3, 4), (3.2, 5), (3.2, 6.5), (3.1, 8), (4, 0.5),
            (4, 1.5), (4, 3), (4.5, 4.5), (4.5, 7), (4.5, 9), (5.5, 0.5), (5.4, 1.5), (5.5, 2.8), (6, 4), (7, 7),
            (7.5, 1.5), (9.5, 3.5), (10.2, 4.5), (10.9, 3), (11.2, 4.5), (12, 3.5)]
    set2 = [(5, 3.5), (4.8, 6), (5.5, 5), (5.5, 7), (5.8, 8), (5.5, 9.5), (6.5, 1.5), (8.5, 0.5), (10, 0.5), (9.8, 1),
            (12, 1), (9, 1.5), (11, 2), (13.5, 1.5), (7.5, 2.5), (7, 3.5), (9.5, 2.5), (12.2, 2.2), (13.5, 3), (8.5, 3),
            (8.6, 4), (7.2, 5.8), (6.8, 6), (6.9, 9), (8.5, 5.2), (8, 6.2), (8, 8), (8.7, 9), (9.5, 5), (9.3, 6),
            (9.1, 8), (11, 6), (10.5, 7), (10.3, 8.8), (10.3, 7.5), (12.2, 5.5), (12.2, 6.8), (13.3, 4.3)]

    return Dataset(set1, set2)


