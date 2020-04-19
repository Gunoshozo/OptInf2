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
import matplotlib.pyplot as plt


a = 5
N = 2 ** 10
M = 2 ** 14
b = np.square(N) / (4 * a * M)
hx = a * 2 / N
hF = b * 2 / N
s = 1
p = 1

def fin_fft_2d( hx, f, N, M):
    f = add_zeros_1d(f, M)
    f = np.fft.fftshift(f)
    F = np.fft.fft(f)
    F = F * hx
    F = np.fft.fftshift(F)
    F = get_central_part_1d(F, N)
    return F


def fin_fft_3d(hx,zf,N,M):
    f = add_zeros_2d(zf,M)
    f = np.fft.fftshift(f)
    F = np.fft.fft(f)
    F = F *(hx**2)
    F = np.fft.fftshift(F)
    F = get_central_part_2d(F, N)
    return F


def core_2d(Xi, x):
    return np.exp(-2j * np.pi * x * Xi)

def core_3d(Xi,x1,x2):
    return np.exp(-2j*np.pi*Xi*x1*x2)


def int_fourier_2d(X, Xi, h, func):
    A = core_2d(Xi[:, None], X[None, :])
    f = func(X)
    res = np.dot(A, f) * h
    return res

def int_fourier_3d(X,Xi,h,func):
    A = core_3d(Xi[:,None,None],X[None,:,None],X[None,None,:])
    f = func(X[:,None],X[None,:])
    res = np.dot(A,f)*(h**2)
    return res

def gaussian(x):
    return np.array(np.exp(-s * np.square(x)), dtype=np.complex)


def input_field_2d(x):
    return x * np.exp(-2j * x)


def gaussian_3d(x, y):
    return np.exp(-s * np.square(x) - p * np.square(y))


def input_field_3d(x, y):
    return x * y * np.exp(-2j * (x + y))


def input_of_size(a, N):
    x = np.linspace(-a, a, N)
    return input_field_2d(x)


def add_zeros_1d(x, M: int):
    l = (M - N) // 2
    r = l + N
    res = np.pad(x,(l,l),'constant',constant_values=(0,0))
    return res

def add_zeros_2d(x, M:int):
    l = (M - N) // 2
    r = l+N
    res = np.pad(x,((l,l),(l,l)),'constant', constant_values=((0,0),(0,0)))
    return res




def get_central_part_1d(x, N):
    M = len(x)
    l = (M - N) // 2
    r = l + N
    return x[l:r]


def get_central_part_2d(x,N):
    l = (M - N) // 2
    r = l + N
    return x[l:r,l:r]


def swap_halves_1d(x):
    N = len(x)
    N_2 = N // 2
    assert N % 2 == 0
    assert N_2 % 2 == 0
    t = np.copy(x)
    x[:N_2] = t[N_2:]
    x[N_2:] = t[:N_2]
    return x





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

def analitical_input_field_3d(x,a):
    return 0

def input_field_print_2d():
    global N, M, hx
    funct = input_field_2d
    x_f = np.linspace(-a, a, N)
    x_F = np.linspace(-b, b, N)

    f = funct(x_f)
    F = fin_fft_2d(hx, f, N, M)
    F1 = int_fourier_2d(x_f, x_F, hx, funct)
    F2 = analitical_input_field_2d(x_F, a)
    plot_all_2D((x_F, x_F, x_F), (F, F1, F2))

    plot_input_2D(x_f, f)


def plot_all_3D(param, param1):
    pass


def plot_input_3D(x_f, f):
    pass


def input_field_print_3d():
    funct = input_field_3d
    x_f = np.linspace(-a,a,N)
    y_f = np.linspace(-a,a,N)
    x_F = np.linspace(-b,b,N)

    x_f, y_f = np.meshgrid(x_f, y_f)

    #zf = funct(x_f,y_f)
    #F = fin_fft_3d(hx,zf,N,M)
    F1 = int_fourier_3d(x_f,x_F,hx,funct)
    F2 = analitical_input_field_3d()

    #plot_all_3D((x_F, x_F, x_F), (F, F1, F2))

    #plot_input_3D(x_f, zf)




def gaussian_print_2d():
    global N, M, hx
    funct = gaussian
    x_f = np.linspace(-a, a, N)
    x_F = np.linspace(-b, b, N)

    f = funct(x_f)
    F = fin_fft_2d(hx, f, N, M)
    F1 = int_fourier_2d(x_f, x_F, hx, funct)
    plot_all_2D((x_F, x_F), (F, F1))

    plot_input_2D(x_f, f)


def main():
   input_field_print_3d()


if __name__ == '__main__':
    main()
