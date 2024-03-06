import numpy as np

def TDMA(a, b, c, f):
    """Реализация метода прогонки для решения СЛАУ. A quick Aligorithm for solving AX=b when A is Tridiagonal matrix.

    Args:
        a (list[float]): диагональ, лежащая под главной (нумеруется: [0;n-1], a[0]=0)
        b (list[float]): диагональ, лежащая над главной (нумеруется: [0;n-1], b[n-1]=0)
        c (list[float]): главная диагональ матрицы A    (нумеруется: [0;n-1])
        f (list[float]): правая часть (столбец)         (нумеруется: [0;n-1])

    Returns:
        x (list[float]): решение, массив x будет содержать ответ (нумеруется: [0;n-1])
    """

    a, b, c, f = tuple(map(lambda k_list: list(map(float, k_list)), (a, b, c, f)))

    alpha = [-b[0] / c[0]]
    beta = [f[0] / c[0]]
    n = len(f)
    x = [0] * n

    for i in range(1, n):
        alpha.append(-b[i] / (a[i] * alpha[i - 1] + c[i]))
        beta.append((f[i] - a[i] * beta[i - 1]) / (a[i] * alpha[i - 1] + c[i]))

    x[n - 1] = beta[n - 1]

    for i in range(n - 1, -1, -1):
        x[i - 1] = alpha[i - 1] * x[i] + beta[i - 1]

    return x


def readMatrix(n: int) -> list[list[float]]:
    matrix = [list(map(lambda x: float(x), input(f'Введите {i + 1} строку (через пробелы): ').split(' '))) for i in range(n)]
    return matrix

def readSolution(n: int) -> list[float]:
    vec = list(map(lambda x: float(x), input(f'Введите строку решения: ').split(' ')))
    return vec

def getABCForTDMA(matrix: list[list[float]]) -> tuple:
    n = len(matrix)
    a = [0] + [matrix[i + 1][i] for i in range(n - 1)]  # диагональ, лежащая под главной (a[0]=0)
    b = [matrix[i][i + 1] for i in range(n - 1)] + [0]  # диагональ, лежащая над главной (b[n-1]=0)
    c = [matrix[i][i] for i in range(n)]                # главная диагональ матрицы A

    return a, b, c
    

if __name__ == '__main__':
    n = int(input('Введите размерность матрицы одним числом: '))
    
    print('\nНачался процесс считывания матрицы')
    print('Матрица:')
    matrix = readMatrix(n)
    print('\nСтолбец решения:')
    f = readSolution(n)
    print('Завершился процесс считывания матрицы\n')
    
    a, b, c = getABCForTDMA(matrix)
    
    x = TDMA(a, b, c, f)

    s = ''
    for cx in x:
        s += f'{cx} '
    print(f'Результат: [{s[:-1]}]')
    
    print('Проверка (расчитаем Ax): ')
    A = np.array(matrix)
    X = np.array(x).T
    
    print(np.dot(A, X))
    
    