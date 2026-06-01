from math import atan2, cos, pi, sin, sqrt
import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk

# Лабораторная работа 3.

SAMPLE_COUNT = 100000
RANDOM_SEED = 20260504
EPS = 1e-9
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRIANGLE_A = (0.0, 0.0, 0.0)
TRIANGLE_B = (2.0, 0.0, 0.0)
TRIANGLE_C = (0.5, 1.8, 0.7)

CIRCLE_CENTER = (-1.0, 0.5, 0.8)
CIRCLE_NORMAL = (0.2, -0.3, 1.0)
CIRCLE_RADIUS = 1.2

SPHERE_Z_PARTS = 2
SPHERE_PHI_SECTORS = 4

COSINE_NORMAL = (0.0, 0.0, 1.0)


def parse_triplet(text):
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Нужно ввести три числа через запятую")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def parse_positive_int(text):
    value = int(text)
    if value <= 0:
        raise ValueError("Количество выборок должно быть положительным")
    return value


def parse_positive_float(text):
    value = float(text)
    if value <= 0:
        raise ValueError("Радиус должен быть положительным")
    return value


def add_vec(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub_vec(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale_vec(v, k):
    return (v[0] * k, v[1] * k, v[2] * k)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(v):
    return sqrt(dot(v, v))


def normalize(v):
    v_len = length(v)
    if v_len < EPS:
        raise ValueError("Нельзя нормализовать нулевой вектор")
    return scale_vec(v, 1.0 / v_len)


def distance(a, b):
    return length(sub_vec(a, b))


def mean_point(points):
    total = (0.0, 0.0, 0.0)
    for point in points:
        total = add_vec(total, point)
    return scale_vec(total, 1.0 / len(points))


def get_basis_by_normal(normal):
    n = normalize(normal)
    if abs(n[0]) < 0.9:
        helper = (1.0, 0.0, 0.0)
    else:
        helper = (0.0, 1.0, 0.0)

    tangent = normalize(cross(helper, n))
    bitangent = cross(n, tangent)
    return tangent, bitangent, n


def generate_triangle_points(count):
    points = []
    uv_values = []
    ab = sub_vec(TRIANGLE_B, TRIANGLE_A)
    ac = sub_vec(TRIANGLE_C, TRIANGLE_A)

    for _ in range(count):
        u = random.random()
        v = random.random()

        # Метод из пособия: если точка попала во вторую половину параллелограмма,
        # отражаем ее внутрь треугольника.
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v

        point = add_vec(TRIANGLE_A, add_vec(scale_vec(ab, u), scale_vec(ac, v)))
        points.append(point)
        uv_values.append((u, v))

    return points, uv_values


def generate_circle_points(count):
    points = []
    local_xy_values = []
    tangent, bitangent, _ = get_basis_by_normal(CIRCLE_NORMAL)

    for _ in range(count):
        r = CIRCLE_RADIUS * sqrt(random.random())
        phi = 2.0 * pi * random.random()
        x = r * cos(phi)
        y = r * sin(phi)
        point = add_vec(
            CIRCLE_CENTER,
            add_vec(scale_vec(tangent, x), scale_vec(bitangent, y)),
        )
        points.append(point)
        local_xy_values.append((x, y))

    return points, local_xy_values


def generate_sphere_directions(count):
    directions = []
    for _ in range(count):
        z = 1.0 - 2.0 * random.random()
        phi = 2.0 * pi * random.random()
        radius = sqrt(max(0.0, 1.0 - z * z))
        directions.append((radius * cos(phi), radius * sin(phi), z))
    return directions


def generate_cosine_directions(count, normal):
    directions = []
    tangent, bitangent, n = get_basis_by_normal(normal)

    for _ in range(count):
        r = sqrt(random.random())
        phi = 2.0 * pi * random.random()
        x = r * cos(phi)
        y = r * sin(phi)
        z = sqrt(max(0.0, 1.0 - r * r))
        direction = add_vec(add_vec(scale_vec(tangent, x), scale_vec(bitangent, y)), scale_vec(n, z))
        directions.append(normalize(direction))

    return directions


def count_triangle_rectangles(uv_values):
    rectangles = [
        ("T1", 0.00, 0.20, 0.00, 0.20),
        ("T2", 0.20, 0.40, 0.00, 0.20),
        ("T3", 0.60, 0.80, 0.00, 0.20),
        ("T4", 0.00, 0.20, 0.20, 0.40),
        ("T5", 0.20, 0.40, 0.20, 0.40),
        ("T6", 0.40, 0.60, 0.20, 0.40),
        ("T7", 0.20, 0.40, 0.40, 0.60),
        ("T8", 0.00, 0.20, 0.60, 0.80),
    ]
    expected_count = len(uv_values) * 0.2 * 0.2 / 0.5
    return count_rectangles(uv_values, rectangles, expected_count)


def count_circle_sectors(local_xy_values, sector_count=8):
    counts = [0 for _ in range(sector_count)]
    sector_width = 2.0 * pi / sector_count

    for x, y in local_xy_values:
        phi = atan2(y, x)
        if phi < 0.0:
            phi += 2.0 * pi
        index = int(phi / sector_width)
        if index >= sector_count:
            index = sector_count - 1
        counts[index] += 1

    expected_count = len(local_xy_values) / sector_count
    rows = []
    for index, count in enumerate(counts):
        phi_min = index * 360.0 / sector_count
        phi_max = (index + 1) * 360.0 / sector_count
        rows.append((f"C{index + 1}", phi_min, phi_max, count, expected_count, count - expected_count))
    return rows


def count_rectangles(points_2d, rectangles, expected_count):
    rows = []
    for name, x_min, x_max, y_min, y_max in rectangles:
        count = 0
        for x, y in points_2d:
            if x_min <= x < x_max and y_min <= y < y_max:
                count += 1
        rows.append((name, x_min, x_max, y_min, y_max, count, expected_count, count - expected_count))
    return rows


def relative_std(values):
    average = sum(values) / len(values)
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return sqrt(variance) / average


def histogram_density(values, left, right, bin_count):
    counts = [0 for _ in range(bin_count)]
    width = (right - left) / bin_count
    for value in values:
        index = int((value - left) / width)
        if index < 0:
            index = 0
        if index >= bin_count:
            index = bin_count - 1
        counts[index] += 1

    density = [count / (len(values) * width) for count in counts]
    centers = [left + (i + 0.5) * width for i in range(bin_count)]
    return centers, density


def histogram_counts(values, left, right, bin_count):
    counts = [0 for _ in range(bin_count)]
    width = (right - left) / bin_count
    for value in values:
        index = int((value - left) / width)
        if index < 0:
            index = 0
        if index >= bin_count:
            index = bin_count - 1
        counts[index] += 1

    rows = []
    for index, count in enumerate(counts):
        bin_left = left + index * width
        bin_right = bin_left + width
        rows.append((bin_left, bin_right, count))
    return rows


def average_absolute_error(values_a, values_b):
    return sum(abs(a - b) for a, b in zip(values_a, values_b)) / len(values_a)


def check_triangle(points, uv_values):
    normal = normalize(cross(sub_vec(TRIANGLE_B, TRIANGLE_A), sub_vec(TRIANGLE_C, TRIANGLE_A)))
    inside_count = 0
    plane_error = 0.0
    for point, (u, v) in zip(points, uv_values):
        if u >= -EPS and v >= -EPS and u + v <= 1.0 + EPS:
            inside_count += 1
        plane_error += abs(dot(sub_vec(point, TRIANGLE_A), normal))

    expected_center = scale_vec(add_vec(add_vec(TRIANGLE_A, TRIANGLE_B), TRIANGLE_C), 1.0 / 3.0)
    rectangles = count_triangle_rectangles(uv_values)
    return {
        "inside_ratio": inside_count / len(points),
        "constraint_error": plane_error / len(points),
        "mean_error": distance(mean_point(points), expected_center),
        "uniformity_score": relative_std([row[5] for row in rectangles]),
        "rectangles": rectangles,
    }


def check_circle(points, local_xy_values):
    normal = normalize(CIRCLE_NORMAL)
    inside_count = 0
    plane_error = 0.0
    for point, (x, y) in zip(points, local_xy_values):
        if x * x + y * y <= CIRCLE_RADIUS * CIRCLE_RADIUS + EPS:
            inside_count += 1
        plane_error += abs(dot(sub_vec(point, CIRCLE_CENTER), normal))

    sectors = count_circle_sectors(local_xy_values)
    return {
        "inside_ratio": inside_count / len(points),
        "constraint_error": plane_error / len(points),
        "mean_error": distance(mean_point(points), CIRCLE_CENTER),
        "uniformity_score": relative_std([row[3] for row in sectors]),
        "sectors": sectors,
    }


def check_sphere(directions):
    z_values = [direction[2] for direction in directions]
    centers, density = histogram_density(z_values, -1.0, 1.0, 20)
    expected_density = [0.5 for _ in centers]
    parts = count_sphere_parts(directions)
    inside_count = sum(1 for direction in directions if abs(length(direction) - 1.0) <= 1e-8)

    return {
        "inside_ratio": inside_count / len(directions),
        "constraint_error": sum(abs(length(direction) - 1.0) for direction in directions) / len(directions),
        "mean_error": length(mean_point(directions)),
        "uniformity_score": relative_std([row[5] for row in parts]),
        "histogram": (centers, density, expected_density),
        "parts": parts,
    }


def count_sphere_parts(directions):
    z_width = 2.0 / SPHERE_Z_PARTS
    phi_width = 360.0 / SPHERE_PHI_SECTORS
    z_parts = [(-1.0 + i * z_width, -1.0 + (i + 1) * z_width) for i in range(SPHERE_Z_PARTS)]
    phi_parts = [(i * phi_width, (i + 1) * phi_width) for i in range(SPHERE_PHI_SECTORS)]
    counts = [[0 for _ in phi_parts] for _ in z_parts]

    for x, y, z in directions:
        z_index = int((z + 1.0) / z_width)
        if z_index < 0:
            z_index = 0
        if z_index >= len(z_parts):
            z_index = len(z_parts) - 1
        phi = atan2(y, x)
        if phi < 0.0:
            phi += 2.0 * pi
        phi_deg = phi * 180.0 / pi
        phi_index = int(phi_deg / phi_width)
        if phi_index >= len(phi_parts):
            phi_index = len(phi_parts) - 1
        counts[z_index][phi_index] += 1

    expected_count = len(directions) / (len(z_parts) * len(phi_parts))
    rows = []
    for z_index, (z_min, z_max) in enumerate(z_parts):
        for phi_index, (phi_min, phi_max) in enumerate(phi_parts):
            count = counts[z_index][phi_index]
            rows.append(
                (
                    f"S{z_index * len(phi_parts) + phi_index + 1}",
                    z_min,
                    z_max,
                    phi_min,
                    phi_max,
                    count,
                    expected_count,
                    count - expected_count,
                )
            )
    return rows


def check_cosine(directions, normal):
    n = normalize(normal)
    mu_values = [dot(direction, n) for direction in directions]
    centers, density = histogram_density(mu_values, 0.0, 1.0, 20)
    expected_density = [2.0 * center for center in centers]
    mu_bins = []
    for mu_left, mu_right, count in histogram_counts(mu_values, 0.0, 1.0, 20):
        expected_count = len(directions) * (mu_right * mu_right - mu_left * mu_left)
        mu_bins.append((mu_left, mu_right, count, expected_count, count - expected_count))
    inside_count = sum(1 for mu in mu_values if mu >= -EPS)

    return {
        "inside_ratio": inside_count / len(directions),
        "constraint_error": sum(abs(length(direction) - 1.0) for direction in directions) / len(directions),
        "mean_error": abs(sum(mu_values) / len(mu_values) - 2.0 / 3.0),
        "uniformity_score": average_absolute_error(density, expected_density),
        "histogram": (centers, density, expected_density),
        "bins": mu_bins,
    }


def fmt(value):
    return f"{value:.6f}"


def fmt_sci(value):
    return f"{value:.6e}"


def add_metric_row(rows, name, metrics):
    rows.append(
        f"{name};{fmt(metrics['inside_ratio'])};{fmt_sci(metrics['constraint_error'])};"
        f"{fmt_sci(metrics['mean_error'])};{fmt_sci(metrics['uniformity_score'])}"
    )


def add_rectangle_rows(rows, title, rectangles):
    rows.append("")
    rows.append(title)
    rows.append("Прямоугольник;x_min;x_max;y_min;y_max;Число точек;Ожидаемое среднее;Ошибка")
    for name, x_min, x_max, y_min, y_max, count, expected_count, error in rectangles:
        rows.append(
            f"{name};{x_min:.2f};{x_max:.2f};{y_min:.2f};{y_max:.2f};"
            f"{count};{expected_count:.2f};{error:.2f}"
        )


def add_interval_rows(rows, title, left_name, right_name, intervals):
    rows.append("")
    rows.append(title)
    rows.append(f"{left_name};{right_name};Число точек;Ожидаемое среднее;Ошибка")
    for left, right, count, expected_count, error in intervals:
        rows.append(f"{left:.2f};{right:.2f};{count};{expected_count:.2f};{error:.2f}")


def add_sector_rows(rows, title, sectors):
    rows.append("")
    rows.append(title)
    rows.append("Сектор;phi_min_deg;phi_max_deg;Число точек;Ожидаемое среднее;Ошибка")
    for name, phi_min, phi_max, count, expected_count, error in sectors:
        rows.append(f"{name};{phi_min:.2f};{phi_max:.2f};{count};{expected_count:.2f};{error:.2f}")


def add_sphere_part_rows(rows, title, parts):
    rows.append("")
    rows.append(title)
    rows.append("Часть;z_min;z_max;phi_min_deg;phi_max_deg;Число точек;Ожидаемое среднее;Ошибка")
    for name, z_min, z_max, phi_min, phi_max, count, expected_count, error in parts:
        rows.append(
            f"{name};{z_min:.2f};{z_max:.2f};{phi_min:.2f};{phi_max:.2f};"
            f"{count};{expected_count:.2f};{error:.2f}"
        )


def build_result_rows(triangle_metrics, circle_metrics, sphere_metrics, cosine_metrics):
    rows = ["Распределение;Inside ratio;Constraint error;Mean error;Uniformity score"]
    add_metric_row(rows, "Треугольник", triangle_metrics)
    add_metric_row(rows, "Круг", circle_metrics)
    add_metric_row(rows, "Сфера", sphere_metrics)
    add_metric_row(rows, "Косинусное распределение", cosine_metrics)
    add_rectangle_rows(rows, "Проверка треугольника одинаковыми прямоугольниками", triangle_metrics["rectangles"])
    add_sector_rows(rows, "Проверка круга равными секторами", circle_metrics["sectors"])
    add_sphere_part_rows(rows, "Проверка сферы равными частями", sphere_metrics["parts"])
    add_interval_rows(rows, "Проверка косинусного распределения интервалами по mu", "mu_min", "mu_max", cosine_metrics["bins"])
    return rows


def save_results(triangle_metrics, circle_metrics, sphere_metrics, cosine_metrics):
    rows = build_result_rows(triangle_metrics, circle_metrics, sphere_metrics, cosine_metrics)

    with open(os.path.join(BASE_DIR, "results_lr3.txt"), "w", encoding="utf-8") as file:
        file.write("\n".join(rows) + "\n")

    return "\n".join(rows)


def save_plots(
    triangle_points,
    triangle_uv,
    circle_points,
    circle_xy,
    sphere_directions,
    cosine_directions,
    sphere_metrics,
    cosine_metrics,
):
    output_dir = os.path.join(BASE_DIR, "lab3_plots")
    os.makedirs(output_dir, exist_ok=True)

    plot_count = min(5000, len(triangle_points))

    plt.figure(figsize=(7, 6))
    xs = [uv[0] for uv in triangle_uv[:plot_count]]
    ys = [uv[1] for uv in triangle_uv[:plot_count]]
    plt.scatter(xs, ys, s=2, alpha=0.5)
    plt.xlabel("u")
    plt.ylabel("v")
    plt.title("Равномерные точки в треугольнике")
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/triangle.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    xs = [xy[0] for xy in circle_xy[:plot_count]]
    ys = [xy[1] for xy in circle_xy[:plot_count]]
    plt.scatter(xs, ys, s=2, alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Равномерные точки в круге")
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/circle.png", dpi=200)
    plt.close()

    centers, density, expected = sphere_metrics["histogram"]
    plt.figure(figsize=(8, 5))
    plt.bar(centers, density, width=0.09, alpha=0.7, label="Выборка")
    plt.plot(centers, expected, color="black", label="Теория: p(z)=0.5")
    plt.xlabel("z")
    plt.ylabel("Плотность")
    plt.title("Равномерные направления на сфере")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sphere_z.png", dpi=200)
    plt.close()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Равномерные направления на сфере")

    surface_steps = 40
    theta_values = [i * pi / (surface_steps - 1) for i in range(surface_steps)]
    phi_values = [i * (2 * pi) / (surface_steps - 1) for i in range(surface_steps)]
    surface_x = []
    surface_y = []
    surface_z = []
    for theta in theta_values:
        row_x = []
        row_y = []
        row_z = []
        for phi in phi_values:
            row_x.append(sin(theta) * cos(phi))
            row_y.append(sin(theta) * sin(phi))
            row_z.append(cos(theta))
        surface_x.append(row_x)
        surface_y.append(row_y)
        surface_z.append(row_z)

    ax.plot_surface(
        np.array(surface_x),
        np.array(surface_y),
        np.array(surface_z),
        color="#4aa3df",
        alpha=0.14,
        linewidth=0,
    )

    plot_count = min(6000, len(sphere_directions))
    xs = [direction[0] for direction in sphere_directions[:plot_count]]
    ys = [direction[1] for direction in sphere_directions[:plot_count]]
    zs = [direction[2] for direction in sphere_directions[:plot_count]]
    ax.scatter(xs, ys, zs, s=3, alpha=0.38, color="#2374ab")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.view_init(elev=23, azim=-55)
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sphere_3d.png", dpi=200)
    plt.close()

    centers, density, expected = cosine_metrics["histogram"]
    plt.figure(figsize=(8, 5))
    plt.bar(centers, density, width=0.045, alpha=0.7, label="Выборка")
    plt.plot(centers, expected, color="black", label="Теория: p(mu)=2mu")
    plt.xlabel("mu = cos(theta)")
    plt.ylabel("Плотность")
    plt.title("Косинусное распределение направлений")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cosine_mu.png", dpi=200)
    plt.close()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Косинусное распределение направлений")

    theta_values = [i * (pi / 2) / (surface_steps - 1) for i in range(surface_steps)]
    phi_values = [i * (2 * pi) / (surface_steps - 1) for i in range(surface_steps)]
    surface_x = []
    surface_y = []
    surface_z = []
    for theta in theta_values:
        row_x = []
        row_y = []
        row_z = []
        for phi in phi_values:
            row_x.append(sin(theta) * cos(phi))
            row_y.append(sin(theta) * sin(phi))
            row_z.append(cos(theta))
        surface_x.append(row_x)
        surface_y.append(row_y)
        surface_z.append(row_z)

    ax.plot_surface(
        np.array(surface_x),
        np.array(surface_y),
        np.array(surface_z),
        color="#48bff7",
        alpha=0.22,
        linewidth=0,
    )

    plot_count = min(6000, len(cosine_directions))
    xs = [direction[0] for direction in cosine_directions[:plot_count]]
    ys = [direction[1] for direction in cosine_directions[:plot_count]]
    zs = [direction[2] for direction in cosine_directions[:plot_count]]
    ax.scatter(xs, ys, zs, s=3, alpha=0.45, color="#24b9f2")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(0.0, 1.05)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1, 1, 0.65))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cosine_3d.png", dpi=200)
    plt.close()


def build_input_summary(sample_count, seed):
    lines = [
        "Input data",
        f"N = {sample_count}",
        f"seed = {seed}",
        f"Triangle A = {TRIANGLE_A}",
        f"Triangle B = {TRIANGLE_B}",
        f"Triangle C = {TRIANGLE_C}",
        f"Circle center = {CIRCLE_CENTER}",
        f"Circle normal = {CIRCLE_NORMAL}",
        f"Circle radius = {CIRCLE_RADIUS}",
        f"Sphere z parts = {SPHERE_Z_PARTS}",
        f"Sphere phi sectors = {SPHERE_PHI_SECTORS}",
        f"Cosine normal = {COSINE_NORMAL}",
    ]
    return "\n".join(lines)


def run_experiment(sample_count=SAMPLE_COUNT, seed=RANDOM_SEED):
    random.seed(seed)

    triangle_points, triangle_uv = generate_triangle_points(sample_count)
    circle_points, circle_xy = generate_circle_points(sample_count)
    sphere_directions = generate_sphere_directions(sample_count)
    cosine_directions = generate_cosine_directions(sample_count, COSINE_NORMAL)

    triangle_metrics = check_triangle(triangle_points, triangle_uv)
    circle_metrics = check_circle(circle_points, circle_xy)
    sphere_metrics = check_sphere(sphere_directions)
    cosine_metrics = check_cosine(cosine_directions, COSINE_NORMAL)

    result_text = save_results(triangle_metrics, circle_metrics, sphere_metrics, cosine_metrics)
    save_plots(
        triangle_points,
        triangle_uv,
        circle_points,
        circle_xy,
        sphere_directions,
        cosine_directions,
        sphere_metrics,
        cosine_metrics,
    )
    return build_input_summary(sample_count, seed) + "\n\n" + result_text


def apply_config(config):
    global SAMPLE_COUNT
    global RANDOM_SEED
    global TRIANGLE_A
    global TRIANGLE_B
    global TRIANGLE_C
    global CIRCLE_CENTER
    global CIRCLE_NORMAL
    global CIRCLE_RADIUS
    global SPHERE_Z_PARTS
    global SPHERE_PHI_SECTORS
    global COSINE_NORMAL

    SAMPLE_COUNT = config["sample_count"]
    RANDOM_SEED = config["seed"]
    TRIANGLE_A = config["triangle_a"]
    TRIANGLE_B = config["triangle_b"]
    TRIANGLE_C = config["triangle_c"]
    CIRCLE_CENTER = config["circle_center"]
    CIRCLE_NORMAL = config["circle_normal"]
    CIRCLE_RADIUS = config["circle_radius"]
    SPHERE_Z_PARTS = config["sphere_z_parts"]
    SPHERE_PHI_SECTORS = config["sphere_phi_sectors"]
    COSINE_NORMAL = config["cosine_normal"]


def launch_gui():
    root = tk.Tk()
    root.title("ЛР3 МОИ - распределения случайных величин")
    root.geometry("1360x880")

    left = tk.Frame(root, padx=12, pady=12, width=350)
    left.pack(side=tk.LEFT, fill=tk.Y)
    left.pack_propagate(False)
    right = tk.Frame(root, padx=12, pady=12)
    right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    tk.Label(left, text="Входные данные", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 8))
    entries = {}

    def add_entry(parent, key, label, default_value):
        tk.Label(parent, text=label).pack(anchor="w")
        entry = tk.Entry(parent, width=38)
        entry.pack(anchor="w", pady=(0, 6))
        entry.insert(0, default_value)
        entries[key] = entry

    common_section = tk.LabelFrame(left, text="Общие параметры", padx=8, pady=8)
    common_section.pack(fill=tk.X, pady=(0, 8))
    add_entry(common_section, "sample_count", "Количество выборок", str(SAMPLE_COUNT))
    add_entry(common_section, "seed", "Seed", str(RANDOM_SEED))

    triangle_section = tk.LabelFrame(left, text="Треугольник", padx=8, pady=8)
    triangle_section.pack(fill=tk.X, pady=(0, 8))
    add_entry(triangle_section, "triangle_a", "A", ", ".join(str(v) for v in TRIANGLE_A))
    add_entry(triangle_section, "triangle_b", "B", ", ".join(str(v) for v in TRIANGLE_B))
    add_entry(triangle_section, "triangle_c", "C", ", ".join(str(v) for v in TRIANGLE_C))

    circle_section = tk.LabelFrame(left, text="Круг", padx=8, pady=8)
    circle_section.pack(fill=tk.X, pady=(0, 8))
    add_entry(circle_section, "circle_center", "Центр C", ", ".join(str(v) for v in CIRCLE_CENTER))
    add_entry(circle_section, "circle_normal", "Нормаль N", ", ".join(str(v) for v in CIRCLE_NORMAL))
    add_entry(circle_section, "circle_radius", "Радиус Rc", str(CIRCLE_RADIUS))

    sphere_section = tk.LabelFrame(left, text="Сфера", padx=8, pady=8)
    sphere_section.pack(fill=tk.X, pady=(0, 8))
    add_entry(sphere_section, "sphere_z_parts", "Частей по z", str(SPHERE_Z_PARTS))
    add_entry(sphere_section, "sphere_phi_sectors", "Секторов по phi", str(SPHERE_PHI_SECTORS))

    cosine_section = tk.LabelFrame(left, text="Косинусное распределение", padx=8, pady=8)
    cosine_section.pack(fill=tk.X, pady=(0, 8))
    add_entry(cosine_section, "cosine_normal", "Нормаль N", ", ".join(str(v) for v in COSINE_NORMAL))

    notebook = ttk.Notebook(right)
    notebook.pack(fill=tk.BOTH, expand=True)

    plot_3d_tab = tk.Frame(notebook, padx=10, pady=10)
    plots_tab = tk.Frame(notebook, padx=10, pady=10)
    result_tab = tk.Frame(notebook, padx=10, pady=10)
    notebook.add(plot_3d_tab, text="3D-график")
    notebook.add(plots_tab, text="2D-графики")
    notebook.add(result_tab, text="Таблицы")

    plot_3d_grid = tk.Frame(plot_3d_tab)
    plot_3d_grid.pack(fill=tk.BOTH, expand=True)
    plot_3d_images = {}
    plot_3d_labels = {}
    plot_3d_items = [
        ("sphere_3d", "Сфера", os.path.join(BASE_DIR, "lab3_plots", "sphere_3d.png")),
        ("cosine_3d", "Косинусное", os.path.join(BASE_DIR, "lab3_plots", "cosine_3d.png")),
    ]
    for index, (key, title, _) in enumerate(plot_3d_items):
        panel = tk.LabelFrame(plot_3d_grid, text=title, padx=6, pady=6)
        panel.grid(row=0, column=index, padx=8, pady=8, sticky="nsew")
        label = tk.Label(panel, text="3D-график пока не создан", bg="white", bd=1, relief=tk.SOLID)
        label.pack(fill=tk.BOTH, expand=True)
        plot_3d_labels[key] = label

    plot_3d_grid.rowconfigure(0, weight=1)
    for column in range(2):
        plot_3d_grid.columnconfigure(column, weight=1)

    plots_grid = tk.Frame(plots_tab)
    plots_grid.pack(fill=tk.BOTH, expand=True)
    plot_images = {}
    plot_labels = {}
    plot_items = [
        ("triangle", "Треугольник", os.path.join(BASE_DIR, "lab3_plots", "triangle.png")),
        ("circle", "Круг", os.path.join(BASE_DIR, "lab3_plots", "circle.png")),
        ("sphere", "Сфера: p(z)", os.path.join(BASE_DIR, "lab3_plots", "sphere_z.png")),
        ("cosine", "Косинус: p(mu)", os.path.join(BASE_DIR, "lab3_plots", "cosine_mu.png")),
    ]
    for index, (key, title, _) in enumerate(plot_items):
        panel = tk.LabelFrame(plots_grid, text=title, padx=6, pady=6)
        panel.grid(row=index // 2, column=index % 2, padx=8, pady=8, sticky="nsew")
        label = tk.Label(panel, text="График пока не создан", bg="white", bd=1, relief=tk.SOLID)
        label.pack(fill=tk.BOTH, expand=True)
        plot_labels[key] = label

    for row in range(2):
        plots_grid.rowconfigure(row, weight=1)
    for column in range(2):
        plots_grid.columnconfigure(column, weight=1)

    def load_preview_image(path):
        image = Image.open(path).convert("RGB")
        image.thumbnail((500, 330), Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    def update_3d_preview():
        for key, title, path in plot_3d_items:
            try:
                image = Image.open(path).convert("RGB")
                image.thumbnail((520, 660), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                plot_3d_images[key] = photo
                plot_3d_labels[key].configure(image=photo, text="")
            except Exception:
                plot_3d_labels[key].configure(image="", text="3D-график пока не создан")

    def update_plot_previews():
        update_3d_preview()
        for key, title, path in plot_items:
            try:
                image = load_preview_image(path)
                plot_images[key] = image
                plot_labels[key].configure(image=image, text="")
            except Exception:
                plot_labels[key].configure(image="", text="Файл пока не создан")

    tk.Label(result_tab, text="Таблицы для Excel", font=("Arial", 14, "bold")).pack(anchor="w")
    text_frame = tk.Frame(result_tab)
    text_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
    result_text = tk.Text(text_frame, wrap="none", font=("Courier", 13))
    result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    y_scroll = tk.Scrollbar(text_frame, orient="vertical", command=result_text.yview)
    y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    result_text.configure(yscrollcommand=y_scroll.set)
    x_scroll = tk.Scrollbar(result_tab, orient="horizontal", command=result_text.xview)
    x_scroll.pack(fill=tk.X)
    result_text.configure(xscrollcommand=x_scroll.set)

    def select_all_results(event=None):
        result_text.tag_add(tk.SEL, "1.0", tk.END)
        result_text.mark_set(tk.INSERT, "1.0")
        result_text.see(tk.INSERT)
        return "break"

    result_text.bind("<Control-a>", select_all_results)
    result_text.bind("<Command-a>", select_all_results)

    def collect_config():
        return {
            "sample_count": parse_positive_int(entries["sample_count"].get()),
            "seed": int(entries["seed"].get()),
            "triangle_a": parse_triplet(entries["triangle_a"].get()),
            "triangle_b": parse_triplet(entries["triangle_b"].get()),
            "triangle_c": parse_triplet(entries["triangle_c"].get()),
            "circle_center": parse_triplet(entries["circle_center"].get()),
            "circle_normal": parse_triplet(entries["circle_normal"].get()),
            "circle_radius": parse_positive_float(entries["circle_radius"].get()),
            "sphere_z_parts": parse_positive_int(entries["sphere_z_parts"].get()),
            "sphere_phi_sectors": parse_positive_int(entries["sphere_phi_sectors"].get()),
            "cosine_normal": parse_triplet(entries["cosine_normal"].get()),
        }

    def calculate_and_show(show_message=True):
        try:
            config = collect_config()
            apply_config(config)
            output = run_experiment(SAMPLE_COUNT, RANDOM_SEED)
            result_text.delete("1.0", tk.END)
            result_text.insert("1.0", output)
            update_plot_previews()
            if show_message:
                messagebox.showinfo(
                    "Готово",
                    "Расчет выполнен.\nФайлы обновлены: results_lr3.txt и lab3_plots/*.png",
                )
        except Exception as error:
            messagebox.showerror("Ошибка", str(error))

    def save_result_as():
        path = filedialog.asksaveasfilename(
            title="Сохранить результаты",
            defaultextension=".txt",
            initialfile="results_lr3.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        text = result_text.get("1.0", tk.END).strip()
        if not text:
            calculate_and_show(show_message=False)
            text = result_text.get("1.0", tk.END).strip()
        with open(path, "w", encoding="utf-8") as file:
            file.write(text + "\n")
        messagebox.showinfo("Сохранено", f"Результаты сохранены в файл:\n{path}")

    def copy_selected_results():
        try:
            text = result_text.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            messagebox.showinfo("Ничего не выделено", "Сначала выделите нужные строки или ячейки в таблице")
            return
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        messagebox.showinfo("Скопировано", "Выделенный текст скопирован в буфер обмена")

    buttons = tk.Frame(left, pady=8)
    buttons.pack(fill=tk.X)
    tk.Button(buttons, text="Посчитать", command=calculate_and_show, width=16).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(buttons, text="Сохранить как", command=save_result_as, width=16).pack(side=tk.LEFT)

    table_buttons = tk.Frame(left)
    table_buttons.pack(fill=tk.X, pady=(0, 8))
    tk.Button(table_buttons, text="Копировать выделенное", command=copy_selected_results, width=34).pack(fill=tk.X)

    calculate_and_show(show_message=False)
    root.mainloop()


def main():
    print("1  Использовать пример из файла")
    print("2  Открыть окно для ввода параметров")
    mode = input("Выберите режим [2]: ").strip() or "2"

    if mode == "2":
        launch_gui()
        return

    print(run_experiment(SAMPLE_COUNT, RANDOM_SEED))


if __name__ == "__main__":
    main()
