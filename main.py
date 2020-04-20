# x * exp(-1j*2*x)
#   для аналитики - финитное преобразование
#   Использовать св-ва Фурье
'''
1.Реализовать одномерное финитное преобразование Фурье с помощью применения
алгоритма БПФ
2.Построить график гауссова пучка exp(−𝑠𝑥^2), s – задаваемая константа. Здесь и далее для
каждого графика следует строить отдельно графики амплитуды и фазы
3.Убедиться в правильности реализации преобразования, подав на вход гауссов пучок exp(−𝑠𝑥^2)
– собственную функцию преобразования Фурье. На выходе тоже должен
получиться гауссов пучок, но другого масштаба (построить график на правильной
области определения [−𝑏̃, 𝑏̃]). Рекомендуемая входная область: [−𝑎, 𝑎] = [−5, 5]
4.Реализовать финитное преобразование Фурье стандартным методом численного
интегрирования (например, методом прямоугольников). Важно: необходимо
вычислить интеграл для каждого дискретного значения u, чтобы получить результат
в виде вектора. На вход преобразования вновь следует подавать гауссов пучок.
5. Построить результаты двух разных реализаций преобразования на одном
изображении (одно для амплитуды, одно для фазы) и убедиться, что они совпадают.
6. Используя первую реализацию преобразования, подать на вход световое поле,
отличное от гауссова пучка, в соответствии со своим вариантом. Построить графики
самого пучка и результата преобразования.
7. Рассчитать аналитически результат преобразования своего варианта поля и
построить график на одной системе координат с результатом, полученным в
предыдущем пункте
8. Выполнить пункты 1-3 и 6-7 для двумерного случая. Графики изменятся на
двумерные изображения, одномерные функции следует заменить на двумерные,
равные произведению соответствующих одномерных функций. Например, гауссов
пучок поменяется на exp(−𝑠𝑥^2−𝑝𝑦^2), s, p – задаваемые константы.
'''

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

a = 5
N = 2 ** 10
M = 2 ** 13
b = np.square(N) / (4 * a * M)
hx = a * 2 / N
hF = b * 2 / N
s = 1
p = 1


def fin_fft_2d(hx, f):
    f = add_zeros_1d(f, M)
    f = np.fft.fftshift(f)
    F = np.fft.fft(f)
    F = F * hx
    F = np.fft.fftshift(F)
    F = get_central_part_1d(F)
    return F


def fin_fft_3d(hx, zf):
    f = add_zeros_2d(zf, M)
    f = np.fft.fftshift(f)
    F = np.fft.fft2(f)
    F = F * (hx ** 2)
    F = np.fft.fftshift(F)
    F = get_central_part_2d(F)
    return F


def core_2d(Xi, x):
    return np.exp(-2j * np.pi * x * Xi)


def int_fourier_2d(X, Xi, h, func):
    A = core_2d(Xi[:, None], X[None, :])
    f = func(X)
    res = np.dot(A, f) * h
    return res


def int_fourier_3d(X, Xi, h, func):
    A = core_2d(Xi[:, None], X[None, :])
    f = func(X)
    res = np.dot(A, f) * h
    res1 = np.copy(res)
    return np.dot(res[:, None], res1[None, :])


def gaussian(x):
    return np.array(np.exp(-s * np.square(x)), dtype=np.complex)


def input_field_2d(x):
    return x * np.exp(-2j * x)


def gaussian_3d(x1, x2):
    return np.exp(-s * np.square(x1) - p * np.square(x2))


def input_field_3d(x, y):
    return x * y * np.exp(-2j * (x + y))


def input_of_size(a, N):
    x = np.linspace(-a, a, N)
    return input_field_2d(x)


def add_zeros_1d(x, M: int):
    l = (M - N) // 2
    r = l + N
    res = np.pad(x, (l, l), 'constant', constant_values=(0, 0))
    return res


def add_zeros_2d(x, M: int):
    l = (M - N) // 2
    r = l + N
    res = np.pad(x, ((l, l), (l, l)), 'constant', constant_values=((0, 0), (0, 0)))
    return res


def get_central_part_1d(x):
    M = len(x)
    l = (M - N) // 2
    r = l + N
    return x[l:r]


def get_central_part_2d(x):
    l = (M - N) // 2
    r = l + N
    return x[l:r, l:r]


def swap_halves_1d(x):
    N = len(x)
    N_2 = N // 2
    assert N % 2 == 0
    assert N_2 % 2 == 0
    t = np.copy(x)
    x[:N_2] = t[N_2:]
    x[N_2:] = t[:N_2]
    return x

def swap_halves_2d(x):
    N_2 = len(x)//2
    t = np.copy(x)
    a = np.zeros((len(x),len(x)),dtype=np.complex)
    a[:N_2,:N_2] = t[N_2:,N_2:]
    a[N_2:,:N_2] = t[:N_2,N_2:]
    a[N_2:,N_2:] = t[:N_2,:N_2]
    a[:N_2,N_2:] = t[N_2:,:N_2]
    return a


def plot_all_2D(x, y):
    fig1 = plt.figure(1)
    labels = ["через БПФ", "через ПФ", "аналитическое"]
    marks = ['--', '-.', ':']
    for i in range(len(x)):
        label, mark = labels[i], marks[i]
        plt.plot(x[i], np.abs(y[i]), mark, label=label)
    plt.title("Амплитуда")
    plt.grid()
    plt.legend()

    fig2 = plt.figure(2)
    for i in range(len(x)):
        label, mark = labels[i], marks[i]
        plt.plot(x[i], np.angle(y[i]), mark, label=label)
    plt.title("Фаза")
    plt.grid()
    plt.legend()

    plt.show()
    plt.close()


def plot_input_2D(x, y):
    fig1 = plt.figure(1)
    plt.plot(x, np.abs(y))
    plt.title("Амплитуда")
    plt.grid()

    fig2 = plt.figure(2)
    plt.plot(x, np.angle(y))
    plt.title("Фаза")
    plt.grid()

    plt.show()
    plt.close()


def analitical_input_field_2d(x, a):
    return 1j * (2 * (np.pi * a * x + a) * np.cos(2 * a * (np.pi * x + 1)) - np.sin(2 * (np.pi * a * x + a))) / (
            2 * np.square(np.pi * x + 1))


def analitical_input_field_3d(x1, x2, a):
    return -(2 * (np.pi * a * x1 + a) * np.cos(2 * a * (np.pi * x1 + 1)) - np.sin(2 * (np.pi * a * x1 + a))) * \
           (2 * (np.pi * a * x2 + a) * np.cos(2 * a * (np.pi * x2 + 1)) - np.sin(2 * (np.pi * a * x2 + a))) \
           / (4 * np.square(np.pi * x1 + 1) * np.square(np.pi * x2 + 1))


def input_field_print_2d():
    global N, M, hx
    funct = input_field_2d
    x_f = np.linspace(-a, a, N)
    x_F = np.linspace(-b, b, N)

    f = funct(x_f)
    F = fin_fft_2d(hx, f)
    F1 = int_fourier_2d(x_f, x_F, hx, funct)
    F2 = analitical_input_field_2d(x_F, a)
    plot_all_2D((x_F, x_F, x_F), (F, F1, F2))

    plot_input_2D(x_f, f)


def plot_all_3d(xs, ys):
    labels = ["через БПФ", "через ПФ", "аналитическое"]
    ops = [np.abs, np.angle]
    axises = ["abs", "angle"]
    for i in range(len(xs)):
        label = labels[i]
        x,x = np.meshgrid(xs[i],xs[i])
        for j in range(2):
            op, axis = ops[j], axises[j]
            target = op(ys[i])
            ax = plt.axes(projection='3d')
            ax.plot_surface(X=x, Y=x.T, Z=target, cmap='plasma', vmin=np.nanmin(target), vmax=np.nanmax(target))
            ax.set_xlabel('ξ1')
            ax.set_xlabel('ξ1')
            ax.set_xlabel(axis)
            plt.savefig(label + ' ' + axis + '.png')
            plt.show()
            plt.close()


def plot_input_3D(x_f, f):
    pass


def input_field_print_3d():
    funct = input_field_3d
    x_f = np.linspace(-a, a, N)
    y_f = np.linspace(-a, a, N)
    x_F = np.linspace(-b, b, N)

    x, y = np.meshgrid(x_f, y_f)
    X, Y = np.meshgrid(x_F, x_F)
    zf = funct(x, y)
    plt.imshow(np.abs(zf),extent=(-a,a,-a,a))
    plt.show()
    plt.imshow(np.angle(zf),extent=(-a,a,-a,a))
    plt.show()
    F = fin_fft_3d(hx, zf)
    plt.imshow(np.abs(F),extent=(-b,b,-b,b))
    #plt.show()
    F1 = int_fourier_3d(x_f, x_F, hx, input_field_2d)
    plt.imshow(np.abs(F1),extent=(-b,b,-b,b))
    #plt.show()
    F2 = analitical_input_field_3d(x_F[:, None], x_F[None, :], a)
    plt.imshow(np.abs(F2),extent=(-b,b,-b,b))
    #plt.show()

    plot_all_3d((x_F, x_F, x_F), (F, F1, F2))

    # plot_input_3D(x_f, zf)


def gaussian_print_2d():
    global N, M, hx
    funct = gaussian
    x_f = np.linspace(-a, a, N)
    x_F = np.linspace(-b, b, N)

    f = funct(x_f)
    F = fin_fft_2d(hx, f)
    F1 = int_fourier_2d(x_f, x_F, hx, funct)
    plot_all_2D((x_F, x_F), (F, F1))

    plot_input_2D(x_f, f)


def gaussian_print_3d():
    funct = gaussian_3d
    x_f = np.linspace(-a, a, N)
    y_f = np.linspace(-a, a, N)
    x_F = np.linspace(-b, b, N)

    zf = funct(x_f[:, None], y_f[None, :])
    plt.imshow(zf)
    #plt.show()
    F = fin_fft_3d(hx, zf)
    plt.imshow(np.abs(F))
    #plt.show()
    F1 = int_fourier_3d(x_f, x_F, hx, gaussian)
    plt.imshow(np.abs(F1))
    #plt.show()
    x, y = np.meshgrid(x_f, x_f)
    F2 = analitical_input_field_3d(x, y, a)
    plt.imshow(np.abs(F))


def main():
    #input_field_print_3d()
    input_field_print_3d()


if __name__ == '__main__':
    main()
