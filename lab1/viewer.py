import matplotlib.pyplot as plot


def plot_angles(x, avg, err, max_dim=500):
    fig = plot.figure(figsize=(16, 9))
    plot.errorbar(x, avg, yerr=err, fmt=',-m', ecolor='y', elinewidth=1, capsize=2)
    plot.title('kąt między dwoma wylosowanymi wektorami', size=25)
    plot.ylabel('kąt (deg)', size=15)
    plot.xlabel('liczba wymiarów', size=15)
    plot.legend(['średnia i odchylenie standardowe'], loc='upper right', prop={'size': 15})
    plot.xlim((0, max_dim))
    plot.grid()
    print('Saving PNG...')
    plot.savefig('angle.png')
    print('angle.png saved')


def plot_ratios(x, avg, err, max_dim=500):
    fig = plot.figure(figsize=(16, 9))
    plot.errorbar(x, avg, yerr=err, fmt=',-m', ecolor='y', elinewidth=1, capsize=2)
    plot.title('prawdopodobieństwo wystąpienia punktu z hipersześcianu wewnątrz kuli', size=25)
    plot.ylabel('prawdopodobieństwo (%)', size=15)
    plot.xlabel('liczba wymiarów', size=15)
    plot.legend(['średnia i odchylenie standardowe'], loc='upper right', prop={'size': 15})
    plot.xlim((0, max_dim))
    plot.grid()
    print('Saving PNG...')
    plot.savefig('inside.png')
    print('inside.png saved')


def plot_distances(x, avg, err, max_dim=500):
    fig = plot.figure(figsize=(16, 9))
    plot.errorbar(x, avg, yerr=err, fmt=',-m', ecolor='y', elinewidth=1, capsize=2)
    plot.title('stosunek różnicy odległości między dwoma punktami do średniej z tych odległości', size=25)
    plot.ylabel('stosunek (%)', size=15)
    plot.xlabel('liczba wymiarów', size=15)
    plot.legend(['średnia i odchylenie standardowe'], loc='upper right', prop={'size': 15})
    plot.xlim((0, max_dim))
    plot.grid()
    print('Saving PNG...')
    plot.savefig('ratio.png')
    print('ratio.png saved')
