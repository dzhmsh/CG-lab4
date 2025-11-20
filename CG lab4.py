import numpy as np
from PIL import Image, ImageOps

print("I'm trying my hardest")

# Создание холста для изображения размером 2000x2000 пикселей
img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)

# Создание фона
for i in range(2000):
    for j in range(2000):
        img_mat[i, j] = [0, 0, 0]


def read_obj_data(file):
    vertices = []  # вершины
    poligons = []  # полигоны (треугольники)
    meat = []  # координаты текстуры
    bones = []
    for s in file:
        s_list = s.split()
        if (s_list[0] == 'v'):
            vertices.append(
                [float(s_list[1]), float(s_list[2]), float(s_list[3])])

        if (s_list[0] == 'vt'):
            meat.append([float(s_list[1]), float(s_list[2])])

        if (s_list[0] == 'f'):
            poligons.append([int(s_list[1].split(
                '/')[0]), int(s_list[2].split('/')[0]), int(s_list[3].split('/')[0])])
            bones.append([int(s_list[1].split(
                '/')[1]), int(s_list[2].split('/')[1]), int(s_list[3].split('/')[1])])

    return vertices, poligons, meat, bones


def normal_array(poligons, vertices):
    poly_n = []
    lenth = len(poligons)

    '''
        a0 = vertices[poligons[k][0]]
        a1 = vertices[poligons[k][1]]
        a2 = vertices[poligons[k][2]]
    '''
    for k in range(lenth):
        a0 = vertices[poligons[k][0]-1]
        a1 = vertices[poligons[k][1]-1]
        a2 = vertices[poligons[k][2]-1]
        poly_n.append(np.cross([a1[0]-a2[0], a1[1]-a2[1], a1[2]-a2[2]],
                               [a1[0]-a0[0], a1[1]-a0[1], a1[2]-a0[2]]))

        poly_n[k] /= np.linalg.norm(poly_n[k])

    vertex_n = np.zeros((lenth, 3), dtype=np.float32)
    for k in range(lenth):
        v0 = poligons[k][0]-1
        v1 = poligons[k][1]-1
        v2 = poligons[k][2]-1
        vertex_n[v0] += poly_n[k]
        vertex_n[v1] += poly_n[k]
        vertex_n[v2] += poly_n[k]

    for k in range(len(vertex_n)):
        vertex_n[k] /= np.linalg.norm(vertex_n[k])

    return vertex_n

# Создание матрицы поворота по осям X, Y, Z


def rotation(rt_x, rt_y, rt_z):
    Rx = np.array([[1, 0, 0], [0, np.cos(rt_x), np.sin(rt_x)],
                  [0, -np.sin(rt_x), np.cos(rt_x)]])
    Ry = np.array([[np.cos(rt_y), 0, np.sin(rt_y)], [
                  0, 1, 0], [-np.sin(rt_y), 0, np.cos(rt_y)]])
    Rz = np.array([[np.cos(rt_z), np.sin(rt_z), 0],
                  [-np.sin(rt_z), np.cos(rt_z), 0], [0, 0, 1]])

    R = Rx @ Ry @ Rz

    return R

# Применение преобразований (поворот и смещение) к вершинам


def apply_transformations(R, vertices):

    for i in range(len(vertices)):
        vertices[i] = R@vertices[i] + [0, -0.04, 0.15]

# Вычисление барицентрических координат для точки (x,y) относительно треугольника


def barycentric_coords(x, y, x0, y0, x1, y1, x2, y2):

    denominator = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    if denominator == 0:
        return [1, 1, 1]

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denominator
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denominator
    lambda2 = 1.0 - lambda0 - lambda1

    return [lambda0, lambda1, lambda2]

# Основная функция отрисовки 3D-модели


def render_model(vertices, poligons, vertex_n, meat, skin, bones, n):

    for k in range(len(poligons)):
        # Получение координат вершин треугольника
        x0 = vertices[poligons[k][0]-1][0]
        y0 = vertices[poligons[k][0]-1][1]
        z0 = vertices[poligons[k][0]-1][2]
        x1 = vertices[poligons[k][1]-1][0]
        y1 = vertices[poligons[k][1]-1][1]
        z1 = vertices[poligons[k][1]-1][2]
        x2 = vertices[poligons[k][2]-1][0]
        y2 = vertices[poligons[k][2]-1][1]
        z2 = vertices[poligons[k][2]-1][2]

        x0t = meat[bones[k][0]-1][0]
        y0t = meat[bones[k][0]-1][1]
        x1t = meat[bones[k][1]-1][0]
        y1t = meat[bones[k][1]-1][1]
        x2t = meat[bones[k][2]-1][0]
        y2t = meat[bones[k][2]-1][1]

        I0 = np.dot(vertex_n[poligons[k][0]-1], [0, 0, 1])
        I1 = np.dot(vertex_n[poligons[k][1]-1], [0, 0, 1])
        I2 = np.dot(vertex_n[poligons[k][2]-1], [0, 0, 1])
        pain_triangle(n, x0, y0, z0, x1, y1, z1, x2,
                      y2, z2, x0t, y0t, x1t, y1t, x2t, y2t, skin, [0, 200, 200], I0, I1, I2)

# Вычисление минимального и максимального значения с ограничением снизу нулем


def get_bounds(a1, a2, a3):
    fmax = max(a1, a2, a3)
    fmin = min(a1, a2, a3)
    if (fmax < 0):
        fmax = 0
    if (fmin < 0):
        fmin = 0
    return [int(fmin), int(fmax)]

# Вычисление нормали к треугольнику и косинуса угла с направлением взгляда


def normal(a0, a1, a2):

    n = np.cross([a1[0]-a2[0], a1[1]-a2[1], a1[2]-a2[2]],
                 [a1[0]-a0[0], a1[1]-a0[1], a1[2]-a0[2]])

    normal_vec = n / np.linalg.norm(n)
    view_dir = [0, 0, 1]  # направление взгляда (по оси Z)
    cos_angle = np.dot(normal_vec, view_dir)
    return cos_angle

# Цвет пикселя


def get_color(x, y, cos_angle):
    contrast = cos_angle  # Кубическое усиление контраста

    # Красный канал - сильно зависит от угла
    '''r = max(0, min(255, int(-255 * contrast)))

    # Зеленый канал - паттерн + угол
    g = max(0, min(255, int((y + x) ** abs(contrast) % 256)))

    # Синий канал - добавляем глубину и зависимость от координат
    b = max(0, min(255, int(128 * abs(contrast) + (x * y) % 128)))
'''
    r = 255*cos_angle
    b = 0*cos_angle
    g = 120*cos_angle
    return [r, g, b]

# Отрисовка одного треугольника с учетом перспективы и z-буфера


def pain_triangle(n, x0, y0, z0, x1, y1, z1, x2,
                  y2, z2, x0t, y0t, x1t, y1t, x2t, y2t, skin, color, I0, I1, I2):

    # Центр изображения для смещения
    center_y = img_mat.shape[0] * 0.5
    center_x = img_mat.shape[1] * 0.5

    # Проекция вершин на экран с перспективой
    x0s = x0 * n / z0 + center_x
    x1s = x1 * n / z1 + center_x
    x2s = x2 * n / z2 + center_x
    y0s = y0 * n / z0 + center_y
    y1s = y1 * n / z1 + center_y
    y2s = y2 * n / z2 + center_y

    # Определение ограничивающего прямоугольника для треугольника
    x_bounds = get_bounds(x0s, x1s, x2s)
    y_bounds = get_bounds(y0s, y1s, y2s)

    # Вычисление нормали для отбраковки невидимых граней
    cos_angle = normal([x0, y0, z0], [x1, y1, z1], [x2, y2, z2])

    # Отбраковка задних граней
    if cos_angle > 0:
        return
    # Растеризация треугольника
    for x in range(x_bounds[0], x_bounds[1] + 1):
        for y in range(y_bounds[0], y_bounds[1] + 1):
            [l1, l2, l3] = barycentric_coords(
                x, y, x0s, y0s, x1s, y1s, x2s, y2s)
            if l1 >= 0 and l2 >= 0 and l3 >= 0:
                # Интерполяция z-координаты
                z_interpolated = l1 * z0 + l2 * z1 + l3 * z2
                I_interpolated = (l1 * I0 + l2 * I1 + l3 * I2)
                # Проверка z-буфера
                if z_interpolated < zbuf[y, x]:
                    cell = [1024*(l1*x0t + l2 * x1t + l3 * x2t),
                            1024 * (l1*y0t + l2*y1t + l3*y2t)]
                    '''img_mat[y, x] = get_color(x, y, I_interpolated)'''
                    img_mat[y, x] = np.array(skin.getpixel(
                        (cell[0], 1023 - cell[1])))*(-I_interpolated)
                    zbuf[y, x] = z_interpolated


# Основная программа
file1 = open('model_1.obj')
skin = Image.open('voda.jpg')
vertices, poligons, meat, bones = read_obj_data(file1)

print(poligons[:20])
# Применение поворота и смещения
R = rotation(0, 2.6, 0)
apply_transformations(R, vertices)

# Инициализация z-буфера
zbuf = np.full((2000, 2000), np.inf)

vertex_n = normal_array(poligons, vertices)

# Отрисовка модели
render_model(vertices, poligons, vertex_n, meat, skin, bones, 2000)


# Сохранение изображения
img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)  # Отражаем по вертикали
img.save('img.png')
print("I drawt this beautiful rabbit")

file1.close()
